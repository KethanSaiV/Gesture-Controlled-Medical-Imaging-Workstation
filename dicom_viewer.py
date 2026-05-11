"""
Phase 1 - Basic DICOM Viewer
Gesture-Controlled Medical Imaging Workstation
"""

import sys
import os
import numpy as np
import pydicom
import SimpleITK as sitk

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
    QStatusBar, QGroupBox, QGridLayout, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QPalette, QColor


# ──────────────────────────────────────────────
#  DICOM Loader (runs in background thread)
# ──────────────────────────────────────────────
class DicomLoader(QThread):
    loaded = pyqtSignal(object, dict)   # emits (volume_array, metadata)
    error  = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            # ── Single .dcm file ──────────────────────────────
            if os.path.isfile(self.path):
                ds = pydicom.dcmread(self.path)
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]          # make it (1, H, W)
                meta = self._extract_meta(ds)
                self.loaded.emit(arr, meta)

            # ── Directory of .dcm files ───────────────────────
            elif os.path.isdir(self.path):
                reader = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(self.path)
                if not series_ids:
                    self.error.emit("No DICOM series found in folder.")
                    return
                files = reader.GetGDCMSeriesFileNames(self.path, series_ids[0])
                reader.SetFileNames(files)
                image = reader.Execute()
                arr = sitk.GetArrayFromImage(image).astype(np.float32)
                # SimpleITK gives (Z, Y, X) — that's (slices, H, W) ✓
                ds   = pydicom.dcmread(files[0])
                meta = self._extract_meta(ds)
                self.loaded.emit(arr, meta)
            else:
                self.error.emit("Invalid path.")
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _extract_meta(ds):
        def safe(tag, default="N/A"):
            return str(getattr(ds, tag, default))
        return {
            "Patient":    safe("PatientName"),
            "Modality":   safe("Modality"),
            "Study Date": safe("StudyDate"),
            "Rows":       safe("Rows"),
            "Columns":    safe("Columns"),
            "Institution":safe("InstitutionName"),
        }


# ──────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────
class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume   = None          # numpy array (slices, H, W)
        self.current  = 0             # current slice index
        self.ww       = 400.0         # window width
        self.wl       = 40.0          # window level (centre)
        self._drag_start = None

        self._setup_ui()

    # ── UI Layout ─────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("DICOM Viewer — Phase 1")
        self.setMinimumSize(1000, 700)
        self._apply_dark_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(10, 10, 10, 10)

        # ── Left panel (controls + metadata) ─
        left = QVBoxLayout()
        left.setSpacing(8)

        # Open buttons
        btn_file   = QPushButton("📂  Open DICOM File")
        btn_folder = QPushButton("📁  Open DICOM Folder")
        btn_file.clicked.connect(self.open_file)
        btn_folder.clicked.connect(self.open_folder)
        for b in (btn_file, btn_folder):
            b.setFixedHeight(38)
            b.setStyleSheet(self._btn_style())
        left.addWidget(btn_file)
        left.addWidget(btn_folder)

        # Metadata box
        meta_group = QGroupBox("Patient Info")
        meta_group.setStyleSheet(self._group_style())
        meta_layout = QGridLayout(meta_group)
        self.meta_labels = {}
        for i, key in enumerate(["Patient","Modality","Study Date","Rows","Columns","Institution"]):
            lbl_key = QLabel(f"{key}:")
            lbl_key.setStyleSheet("color:#888; font-size:11px;")
            lbl_val = QLabel("—")
            lbl_val.setStyleSheet("color:#eee; font-size:11px;")
            lbl_val.setWordWrap(True)
            meta_layout.addWidget(lbl_key, i, 0)
            meta_layout.addWidget(lbl_val, i, 1)
            self.meta_labels[key] = lbl_val
        left.addWidget(meta_group)

        # Windowing controls
        win_group = QGroupBox("Windowing")
        win_group.setStyleSheet(self._group_style())
        win_layout = QVBoxLayout(win_group)

        self.lbl_wl = QLabel("Window Level (WL): 40")
        self.lbl_ww = QLabel("Window Width  (WW): 400")
        self.lbl_wl.setStyleSheet("color:#ccc; font-size:11px;")
        self.lbl_ww.setStyleSheet("color:#ccc; font-size:11px;")

        self.slider_wl = QSlider(Qt.Horizontal)
        self.slider_wl.setRange(-1000, 3000)
        self.slider_wl.setValue(40)
        self.slider_wl.valueChanged.connect(self._on_wl_change)

        self.slider_ww = QSlider(Qt.Horizontal)
        self.slider_ww.setRange(1, 4000)
        self.slider_ww.setValue(400)
        self.slider_ww.valueChanged.connect(self._on_ww_change)

        for w in (self.lbl_wl, self.slider_wl, self.lbl_ww, self.slider_ww):
            win_layout.addWidget(w)

        # Presets
        presets_layout = QHBoxLayout()
        for name, (wl, ww) in [("Brain",(40,80)),("Lung",(-600,1500)),("Bone",(400,1800))]:
            pb = QPushButton(name)
            pb.setFixedHeight(28)
            pb.setStyleSheet(self._btn_style(small=True))
            pb.clicked.connect(lambda _, l=wl, w=ww: self._apply_preset(l, w))
            presets_layout.addWidget(pb)
        win_layout.addLayout(presets_layout)
        left.addWidget(win_group)

        # Slice info
        self.lbl_slice = QLabel("Slice: — / —")
        self.lbl_slice.setStyleSheet("color:#aaa; font-size:12px; qproperty-alignment: AlignCenter;")
        left.addWidget(self.lbl_slice)

        left.addStretch()

        # Reset button
        btn_reset = QPushButton("↺  Reset View")
        btn_reset.setFixedHeight(34)
        btn_reset.setStyleSheet(self._btn_style())
        btn_reset.clicked.connect(self._reset_view)
        left.addWidget(btn_reset)

        root.addLayout(left, 1)

        # ── Centre: image display ─────────────
        img_panel = QVBoxLayout()

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#111; border:1px solid #333; border-radius:4px;")
        self.image_label.setMinimumSize(600, 500)
        self.image_label.setText("Open a DICOM file or folder to begin")
        self.image_label.setFont(QFont("Courier New", 13))
        self.image_label.setStyleSheet(
            "background:#0d0d0d; color:#444; border:1px solid #2a2a2a; border-radius:6px;"
        )

        # Slice slider (vertical)
        self.slice_slider = QSlider(Qt.Vertical)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slice_change)
        self.slice_slider.setStyleSheet("QSlider::handle:vertical{background:#3af;}")

        img_row = QHBoxLayout()
        img_row.addWidget(self.image_label)
        img_row.addWidget(self.slice_slider)
        img_panel.addLayout(img_row)

        # Navigation buttons
        nav = QHBoxLayout()
        btn_prev = QPushButton("◀  Prev")
        btn_next = QPushButton("Next  ▶")
        for b in (btn_prev, btn_next):
            b.setFixedHeight(34)
            b.setStyleSheet(self._btn_style())
        btn_prev.clicked.connect(self._prev_slice)
        btn_next.clicked.connect(self._next_slice)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        img_panel.addLayout(nav)

        root.addLayout(img_panel, 3)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("color:#888; background:#1a1a1a;")
        self.status.showMessage("Ready — open a DICOM file or folder to begin.")

    # ── Open handlers ─────────────────────────
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM (*.dcm);;All Files (*)")
        if path:
            self._load(path)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if path:
            self._load(path)

    def _load(self, path):
        self.status.showMessage("Loading…")
        self.loader = DicomLoader(path)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.error.connect(self._on_error)
        self.loader.start()

    def _on_loaded(self, volume, meta):
        self.volume  = volume
        self.current = volume.shape[0] // 2
        self.slice_slider.setRange(0, volume.shape[0] - 1)
        self.slice_slider.setValue(self.current)
        self._update_meta(meta)
        self._reset_windowing()
        self._show_slice()
        self.status.showMessage(f"Loaded  {volume.shape[0]} slices  —  {volume.shape[2]}×{volume.shape[1]} px")

    def _on_error(self, msg):
        self.status.showMessage(f"Error: {msg}")
        self.image_label.setText(f"⚠  {msg}")

    # ── Slice display ─────────────────────────
    def _show_slice(self):
        if self.volume is None:
            return
        raw = self.volume[self.current]
        img = self._apply_window(raw)
        h, w = img.shape
        qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix  = QPixmap.fromImage(qimg)
        pix  = pix.scaled(
            self.image_label.width() - 4,
            self.image_label.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pix)
        total = self.volume.shape[0]
        self.lbl_slice.setText(f"Slice:  {self.current + 1} / {total}")

    def _apply_window(self, raw):
        lo = self.wl - self.ww / 2
        hi = self.wl + self.ww / 2
        img = np.clip(raw, lo, hi)
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        return img

    # ── Slice navigation ──────────────────────
    def _on_slice_change(self, val):
        self.current = val
        self._show_slice()

    def _prev_slice(self):
        if self.volume is not None and self.current > 0:
            self.current -= 1
            self.slice_slider.setValue(self.current)

    def _next_slice(self):
        if self.volume is not None and self.current < self.volume.shape[0] - 1:
            self.current += 1
            self.slice_slider.setValue(self.current)

    def wheelEvent(self, event):
        if self.volume is None:
            return
        if event.angleDelta().y() > 0:
            self._prev_slice()
        else:
            self._next_slice()

    # ── Windowing ─────────────────────────────
    def _on_wl_change(self, val):
        self.wl = float(val)
        self.lbl_wl.setText(f"Window Level (WL): {val}")
        self._show_slice()

    def _on_ww_change(self, val):
        self.ww = float(val)
        self.lbl_ww.setText(f"Window Width  (WW): {val}")
        self._show_slice()

    def _apply_preset(self, wl, ww):
        self.slider_wl.setValue(wl)
        self.slider_ww.setValue(ww)

    def _reset_windowing(self):
        if self.volume is not None:
            self.wl = float(np.median(self.volume))
            self.ww = float(self.volume.max() - self.volume.min())
            self.slider_wl.setValue(int(self.wl))
            self.slider_ww.setValue(int(self.ww))

    def _reset_view(self):
        if self.volume is not None:
            self.current = self.volume.shape[0] // 2
            self.slice_slider.setValue(self.current)
            self._reset_windowing()

    # ── Metadata ──────────────────────────────
    def _update_meta(self, meta):
        for key, val in meta.items():
            if key in self.meta_labels:
                self.meta_labels[key].setText(val)

    # ── Resize redraws image ──────────────────
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_slice()

    # ── Styles ────────────────────────────────
    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1a1a1a;
                color: #dddddd;
                font-family: 'Segoe UI', sans-serif;
                font-size: 12px;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #333; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3af; width: 14px; height: 14px;
                margin: -5px 0; border-radius: 7px;
            }
            QSlider::sub-page:horizontal { background: #3af; border-radius: 2px; }
            QSlider::groove:vertical {
                width: 4px; background: #333; border-radius: 2px;
            }
            QSlider::handle:vertical {
                background: #3af; width: 14px; height: 14px;
                margin: 0 -5px; border-radius: 7px;
            }
            QSlider::sub-page:vertical { background: #3af; border-radius: 2px; }
            QStatusBar { background: #151515; color: #666; }
        """)

    def _btn_style(self, small=False):
        pad = "4px 8px" if small else "6px 12px"
        return f"""
            QPushButton {{
                background:#252525; color:#ccc; border:1px solid #383838;
                border-radius:4px; padding:{pad};
            }}
            QPushButton:hover {{ background:#2e2e2e; color:#fff; border-color:#3af; }}
            QPushButton:pressed {{ background:#1a1a1a; }}
        """

    def _group_style(self):
        return """
            QGroupBox {
                border:1px solid #2e2e2e; border-radius:6px;
                margin-top:8px; padding:8px;
                color:#888; font-size:11px;
            }
            QGroupBox::title { subcontrol-origin:margin; left:8px; color:#3af; }
        """


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    viewer = DicomViewer()
    viewer.show()
    sys.exit(app.exec_())