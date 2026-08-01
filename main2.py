"""
main.py
Gesture-Controlled Medical Imaging Workstation

Wires all modules together:
  - ModeBar (top strip)
  - GestureEngine → GestureDispatcher → actions on WorkstationState
  - Left panel swaps content on mode change
  - DicomLoader, SegmentationWorker, PreprocessingEngine
  - Viewer renders from WorkstationState on every state_changed signal
"""

import sys
import os
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import time
import tempfile
import shutil
import subprocess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QFileDialog, QStatusBar,
    QGroupBox, QGridLayout, QProgressBar, QScrollArea,
    QComboBox, QStackedWidget, QSizePolicy, QLineEdit, QTextEdit,
    QListWidget, QListWidgetItem, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

from viewer_state import state, Mode, MODE_LABELS, MODE_ICONS
from mode_bar import ModeBar
from gesture_engine import GestureEngine
from gesture_dispatcher import GestureDispatcher
from preprocessing_engine import PreprocessingEngine
from registration_worker import RegistrationWorker, VIEW_MODES
from path_planning_worker import PathPlanWorker, RIBS_LABEL, save_paths_to_fcsv


# ══════════════════════════════════════════════
#  Organ colour map (BGR)
# ══════════════════════════════════════════════
ORGAN_COLOURS = {
    1:  ("Spleen",              (0,   255,  80)),
    2:  ("Kidney Right",        (0,   120, 255)),
    3:  ("Kidney Left",         (0,   200, 255)),
    4:  ("Gallbladder",         (0,   255, 220)),
    5:  ("Liver",               (0,    60, 255)),
    6:  ("Stomach",             (255, 180,   0)),
    7:  ("Pancreas",            (0,   255, 255)),
    8:  ("Adrenal Right",       (255, 255,   0)),
    9:  ("Adrenal Left",        (200, 255,   0)),
    10: ("Aorta",               (0,     0, 255)),
    11: ("Inferior Vena Cava",  (180,   0, 255)),
    12: ("Portal/Splenic Vein", (255,   0, 180)),
    13: ("Iliac Artery Right",  (0,    80, 200)),
    14: ("Iliac Artery Left",   (0,   100, 180)),
    15: ("Iliac Vein Right",    (140,   0, 200)),
    16: ("Iliac Vein Left",     (120,   0, 180)),
    17: ("Lung Right",          (100, 180, 255)),
    18: ("Lung Left",           (60,  140, 255)),
    19: ("Heart",               (0,    80, 255)),
    20: ("Esophagus",           (255, 160, 100)),
    21: ("Urinary Bladder",     (255, 100, 100)),
    22: ("Prostate/Uterus",     (255, 150, 200)),
    23: ("Colon",               (140, 200,  80)),
    24: ("Duodenum",            (255, 200, 100)),
    25: ("Small Bowel",         (120, 180,  60)),
    26: ("Sacrum",              (255, 240, 180)),
    27: ("Femur Right",         (200, 150, 255)),
    28: ("Femur Left",          (150, 100, 255)),
    29: ("Hip Right",           (220, 170, 255)),
    30: ("Hip Left",            (170, 120, 255)),
    31: ("Vertebrae",           (255, 220, 150)),
    32: ("Intervertebral Discs",(200, 200, 140)),
    33: ("Gluteus Max Right",   (180, 255, 180)),
    34: ("Gluteus Max Left",    (140, 220, 140)),
    35: ("Gluteus Med Right",   (160, 240, 160)),
    36: ("Gluteus Med Left",    (120, 200, 120)),
    37: ("Gluteus Min Right",   (140, 220, 100)),
    38: ("Gluteus Min Left",    (100, 180,  80)),
    39: ("Iliopsoas Right",     (255, 180, 140)),
    40: ("Iliopsoas Left",      (220, 140, 100)),
    41: ("Autochthon Right",    (200, 255, 220)),
    42: ("Autochthon Left",     (160, 220, 180)),
    43: ("Spinal Cord",         (240, 240, 100)),
    44: ("Scapula Right",       (230, 210, 255)),
    45: ("Scapula Left",        (200, 180, 255)),
    46: ("Clavicula Right",     (210, 200, 255)),
    47: ("Clavicula Left",      (180, 170, 255)),
    48: ("Humerus Right",       (255, 210, 210)),
    49: ("Humerus Left",        (220, 180, 180)),
    50: ("Brain",               (200, 220, 255)),
    RIBS_LABEL: ("Ribs",        (180, 180, 255)),
}

_RIB_NAMES = {f"rib_left_{i}": RIBS_LABEL for i in range(1, 13)}
_RIB_NAMES.update({f"rib_right_{i}": RIBS_LABEL for i in range(1, 13)})

_NAME_TO_ID = {
    "spleen": 1, "kidney_right": 2, "kidney_left": 3,
    "gallbladder": 4, "liver": 5, "stomach": 6, "pancreas": 7,
    "adrenal_gland_right": 8, "adrenal_gland_left": 9,
    "aorta": 10, "inferior_vena_cava": 11,
    "portal_vein_and_splenic_vein": 12,
    "iliac_artery_right": 13, "iliac_artery_left": 14,
    "iliac_vena_right": 15, "iliac_vena_left": 16,
    "iliac_vein_right": 15, "iliac_vein_left": 16,
    "lung_right": 17, "lung_upper_lobe_right": 17,
    "lung_lower_lobe_right": 17, "lung_middle_lobe_right": 17,
    "lung_left": 18, "lung_upper_lobe_left": 18,
    "lung_lower_lobe_left": 18,
    "heart": 19, "esophagus": 20, "urinary_bladder": 21,
    "prostate": 22, "uterus": 22, "colon": 23,
    "duodenum": 24, "small_bowel": 25, "sacrum": 26,
    "femur_right": 27, "femur_left": 28,
    "hip_right": 29, "hip_left": 30,
    "vertebrae": 31, "intervertebral_discs": 32,
    "gluteus_maximus_right": 33, "gluteus_maximus_left": 34,
    "gluteus_medius_right": 35, "gluteus_medius_left": 36,
    "gluteus_minimus_right": 37, "gluteus_minimus_left": 38,
    "iliopsoas_right": 39, "iliopsoas_left": 40,
    "autochthon_right": 41, "autochthon_left": 42,
    "spinal_cord": 43, "scapula_right": 44, "scapula_left": 45,
    "clavicula_right": 46, "clavicula_left": 47,
    "humerus_right": 48, "humerus_left": 49, "brain": 50,
}
_NAME_TO_ID.update(_RIB_NAMES)


def _pick_task(modality: str) -> str:
    return "total_mr" if modality.upper() in ("MR", "MRI") else "total"


# ══════════════════════════════════════════════
#  Background workers
# ══════════════════════════════════════════════

class DicomLoader(QThread):
    loaded = pyqtSignal(object, object, dict)
    error  = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            if os.path.isfile(self.path):
                ds  = pydicom.dcmread(self.path)
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                self.loaded.emit(arr, sitk.GetImageFromArray(arr), self._meta(ds))
            elif os.path.isdir(self.path):
                reader    = sitk.ImageSeriesReader()
                dicom_dir = self.path
                ids       = reader.GetGDCMSeriesIDs(dicom_dir)
                if not ids:
                    for sub in sorted(os.listdir(dicom_dir)):
                        sp = os.path.join(dicom_dir, sub)
                        if os.path.isdir(sp):
                            ids = reader.GetGDCMSeriesIDs(sp)
                            if ids:
                                dicom_dir = sp
                                break
                if not ids:
                    self.error.emit("No DICOM series found.")
                    return
                files = reader.GetGDCMSeriesFileNames(dicom_dir, ids[0])
                reader.SetFileNames(files)
                img  = reader.Execute()
                arr  = sitk.GetArrayFromImage(img).astype(np.float32)
                ds   = pydicom.dcmread(files[0])
                self.loaded.emit(arr, img, self._meta(ds))
            else:
                self.error.emit("Invalid path.")
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _meta(ds):
        def s(t, d="N/A"): return str(getattr(ds, t, d))
        return {
            "Patient"    : s("PatientName"),
            "Modality"   : s("Modality"),
            "Study Date" : s("StudyDate"),
            "Rows"       : s("Rows"),
            "Columns"    : s("Columns"),
            "Institution": s("InstitutionName"),
        }


class SegmentationWorker(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)
    error    = pyqtSignal(str)

    def __init__(self, dicom_path, modality):
        super().__init__()
        self.dicom_path = dicom_path
        self.modality   = modality.upper()

    def run(self):
        tmp_dir = None
        try:
            task    = _pick_task(self.modality)
            tmp_dir = tempfile.mkdtemp(prefix="totalseg_")
            nii_in  = os.path.join(tmp_dir, "input.nii.gz")
            out_dir = os.path.join(tmp_dir, "segs")
            os.makedirs(out_dir, exist_ok=True)

            self.progress.emit(f"Converting DICOM → NIfTI ({task})...")
            reader    = sitk.ImageSeriesReader()
            dicom_dir = self.dicom_path
            if os.path.isdir(self.dicom_path):
                ids = reader.GetGDCMSeriesIDs(dicom_dir)
                if not ids:
                    for sub in sorted(os.listdir(dicom_dir)):
                        sp = os.path.join(dicom_dir, sub)
                        if os.path.isdir(sp):
                            ids = reader.GetGDCMSeriesIDs(sp)
                            if ids:
                                dicom_dir = sp
                                break
                if not ids:
                    self.error.emit("No DICOM series found.")
                    return
                files = reader.GetGDCMSeriesFileNames(dicom_dir, ids[0])
            else:
                files = [self.dicom_path]
            reader.SetFileNames(files)
            sitk.WriteImage(reader.Execute(), nii_in)

            self.progress.emit(f"Running TotalSegmentator ({task}, CPU fast)...")
            proc = subprocess.Popen(
                [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
                 "-i", nii_in, "-o", out_dir, "-ta", task, "-d", "cpu", "--fast"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                line = line.strip()
                if line and not line.startswith(("W0", "E0", "===")):
                    self.progress.emit(line[:100])
            proc.wait()
            if proc.returncode != 0:
                self.error.emit(f"TotalSegmentator failed (code {proc.returncode}).")
                return

            self.progress.emit("Merging masks...")
            seg_files = [f for f in os.listdir(out_dir)
                         if f.endswith(".nii.gz") or f.endswith(".nii")]
            if not seg_files:
                self.error.emit("No output files from TotalSegmentator.")
                return

            label_vol = None
            merged    = 0
            for fname in sorted(seg_files):
                stem = fname.replace(".nii.gz", "").replace(".nii", "").lower()
                lid  = _NAME_TO_ID.get(stem)
                if lid is None:
                    continue
                arr = sitk.GetArrayFromImage(
                    sitk.ReadImage(os.path.join(out_dir, fname))
                ).astype(np.uint8)
                if label_vol is None:
                    label_vol = np.zeros_like(arr, dtype=np.int32)
                label_vol[arr > 0] = lid
                merged += 1

            if label_vol is None or label_vol.max() == 0:
                self.error.emit("Mask empty — no organs matched.")
                return

            self.progress.emit(f"Done — {merged} organs merged.")
            self.finished.emit(label_vol)
        except Exception as e:
            import traceback
            self.error.emit(f"{e}\n{traceback.format_exc()}")
        finally:
            if tmp_dir:
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ══════════════════════════════════════════════
#  Legend widget
# ══════════════════════════════════════════════

class LegendWidget(QWidget):
    def __init__(self, active_labels=None):
        super().__init__()
        self.active_labels = active_labels or []
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Organ Legend")
        title.setStyleSheet("color:#3af; font-size:11px; font-weight:bold;")
        layout.addWidget(title)
        for label, (name, bgr) in ORGAN_COLOURS.items():
            if self.active_labels and label not in self.active_labels:
                continue
            r, g, b = bgr[2], bgr[1], bgr[0]
            row     = QHBoxLayout()
            sw      = QLabel(); sw.setFixedSize(16, 12)
            sw.setStyleSheet(f"background:rgb({r},{g},{b}); border-radius:2px;")
            nl = QLabel(name); nl.setStyleSheet("color:#ccc; font-size:10px;")
            row.addWidget(sw); row.addWidget(nl); row.addStretch()
            layout.addLayout(row)
        layout.addStretch()


# ══════════════════════════════════════════════
#  Left panel pages (one per mode)
# ══════════════════════════════════════════════

def _group(title):
    g = QGroupBox(title)
    g.setStyleSheet(
        "QGroupBox{border:1px solid #252525;border-radius:6px;"
        "margin-top:8px;padding:5px;color:#555;font-size:10px;}"
        "QGroupBox::title{subcontrol-origin:margin;left:8px;color:#3af;}"
    )
    return g

def _btn(text, highlight=False, small=False):
    b  = QPushButton(text)
    p  = "3px 6px" if small else "5px 10px"
    bg = "#1a3a1a" if highlight else "#222"
    bc = "#3a3"    if highlight else "#333"
    hb = "#2a4a2a" if highlight else "#2a2a2a"
    b.setStyleSheet(
        f"QPushButton{{background:{bg};color:#ccc;border:1px solid {bc};"
        f"border-radius:4px;padding:{p};}}"
        f"QPushButton:hover{{background:{hb};color:#fff;border-color:#3af;}}"
        f"QPushButton:pressed{{background:#111;}}"
        f"QPushButton:disabled{{background:#181818;color:#444;border-color:#222;}}"
    )
    return b

def _lbl(text, style="color:#bbb; font-size:10px;"):
    l = QLabel(text); l.setStyleSheet(style); return l


class ClickableLabel(QLabel):
    """QLabel that reports the pixel position (relative to itself) of a
    left-click — used to let the user mark the tumour point in Path Plan
    mode by clicking directly on the displayed slice."""
    clicked = pyqtSignal(int, int)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos().x(), event.pos().y())
        super().mousePressEvent(event)


class DicomPanel(QWidget):
    open_file_requested   = pyqtSignal()
    open_folder_requested = pyqtSignal()
    wl_changed            = pyqtSignal(int)
    ww_changed            = pyqtSignal(int)
    preset_requested      = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setSpacing(5)

        for text, sig in [("📂 Open DICOM File",   self.open_file_requested),
                           ("📁 Open DICOM Folder", self.open_folder_requested)]:
            b = _btn(text); b.setFixedHeight(32)
            b.clicked.connect(sig); v.addWidget(b)

        # Metadata
        mg = _group("Patient Info")
        gl = QGridLayout(mg)
        self.meta = {}
        for i, k in enumerate(["Patient","Modality","Study Date","Rows","Columns","Institution"]):
            lk = _lbl(f"{k}:", "color:#555; font-size:10px;")
            lv = _lbl("—")
            gl.addWidget(lk, i, 0); gl.addWidget(lv, i, 1)
            self.meta[k] = lv
        lv.setWordWrap(True)
        v.addWidget(mg)

        # Windowing
        wg  = _group("Windowing")
        wgl = QVBoxLayout(wg)
        self.lbl_wl = _lbl("WL: 40"); self.lbl_ww = _lbl("WW: 400")
        self.sld_wl = QSlider(Qt.Horizontal); self.sld_wl.setRange(-1000, 3000); self.sld_wl.setValue(40)
        self.sld_ww = QSlider(Qt.Horizontal); self.sld_ww.setRange(1, 4000);     self.sld_ww.setValue(400)
        self.sld_wl.valueChanged.connect(self.wl_changed)
        self.sld_ww.valueChanged.connect(self.ww_changed)
        for w in (self.lbl_wl, self.sld_wl, self.lbl_ww, self.sld_ww):
            wgl.addWidget(w)
        pr = QHBoxLayout()
        for name, (l, w) in [("Brain",(40,80)),("Lung",(-600,1500)),("Bone",(400,1800))]:
            pb = _btn(name, small=True); pb.setFixedHeight(22)
            pb.clicked.connect(lambda _, lv=l, wv=w: self.preset_requested.emit(lv, wv))
            pr.addWidget(pb)
        wgl.addLayout(pr)
        v.addWidget(wg)
        v.addStretch()

    def update_meta(self, meta: dict):
        for k, lbl in self.meta.items():
            lbl.setText(meta.get(k, "—"))

    def set_wl(self, val):
        self.sld_wl.blockSignals(True); self.sld_wl.setValue(int(val)); self.sld_wl.blockSignals(False)
        self.lbl_wl.setText(f"WL: {int(val)}")

    def set_ww(self, val):
        self.sld_ww.blockSignals(True); self.sld_ww.setValue(int(val)); self.sld_ww.blockSignals(False)
        self.lbl_ww.setText(f"WW: {int(val)}")


class FilterPanel(QWidget):
    filter_changed   = pyqtSignal(str)
    strength_changed = pyqtSignal(float)
    apply_requested  = pyqtSignal()
    reset_requested  = pyqtSignal()

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setSpacing(5)
        g = _group("Preprocessing Filter"); gl = QVBoxLayout(g)

        gl.addWidget(_lbl("Filter:"))
        self.combo = QComboBox()
        self.combo.addItems(PreprocessingEngine.FILTERS)
        self.combo.setStyleSheet(
            "QComboBox{background:#222;color:#ddd;border:1px solid #333;"
            "border-radius:3px;padding:2px;font-size:10px;}"
            "QComboBox QAbstractItemView{background:#222;color:#ddd;"
            "selection-background-color:#3af;}"
        )
        self.combo.currentTextChanged.connect(self.filter_changed)
        gl.addWidget(self.combo)

        self.lbl_strength = _lbl("Strength: 1.0")
        gl.addWidget(self.lbl_strength)
        self.sld = QSlider(Qt.Horizontal); self.sld.setRange(1, 20); self.sld.setValue(10)
        self.sld.valueChanged.connect(lambda v: self.strength_changed.emit(v / 10.0))
        gl.addWidget(self.sld)

        row = QHBoxLayout()
        ba  = _btn("Apply", highlight=True, small=True); ba.clicked.connect(self.apply_requested)
        br  = _btn("Reset", small=True);                 br.clicked.connect(self.reset_requested)
        row.addWidget(ba); row.addWidget(br)
        gl.addLayout(row)
        v.addWidget(g)

        # Gesture guide
        gg = _group("Gesture Guide (Filter Mode)"); ggl = QVBoxLayout(gg)
        for line in ["Swipe UP/DOWN  → Prev/Next filter",
                     "Pinch          → Strength up",
                     "Fist           → Strength down",
                     "Open Palm      → Apply filter"]:
            ggl.addWidget(_lbl(line, "color:#444; font-size:9px;"))
        v.addWidget(gg)
        v.addStretch()

    def set_strength_label(self, val: float):
        self.lbl_strength.setText(f"Strength: {val:.1f}")


class SegmentPanel(QWidget):
    run_requested    = pyqtSignal()
    toggle_requested = pyqtSignal()
    opacity_changed  = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setSpacing(5)
        g = _group("AI Segmentation (TotalSegmentator)"); gl = QVBoxLayout(g)

        self.lbl_hint = _lbl("Load DICOM first", "color:#555; font-size:9px; font-style:italic;")
        gl.addWidget(self.lbl_hint)

        self.btn_run = _btn("🤖 Run Segmentation", highlight=True); self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self.run_requested)
        gl.addWidget(self.btn_run)

        self.btn_tog = _btn("👁 Toggle Overlay"); self.btn_tog.setFixedHeight(26)
        self.btn_tog.clicked.connect(self.toggle_requested)
        self.btn_tog.setEnabled(False)
        gl.addWidget(self.btn_tog)

        gl.addWidget(_lbl("Overlay Opacity:"))
        self.sld_op = QSlider(Qt.Horizontal); self.sld_op.setRange(10, 90); self.sld_op.setValue(45)
        self.sld_op.valueChanged.connect(self.opacity_changed)
        gl.addWidget(self.sld_op)

        self.prog = QProgressBar(); self.prog.setRange(0, 0); self.prog.setVisible(False)
        self.prog.setFixedHeight(6)
        self.prog.setStyleSheet(
            "QProgressBar{border:none;background:#222;border-radius:3px;}"
            "QProgressBar::chunk{background:#3af;border-radius:3px;}"
        )
        gl.addWidget(self.prog)

        self.lbl_status = _lbl("No segmentation loaded"); self.lbl_status.setWordWrap(True)
        gl.addWidget(self.lbl_status)
        v.addWidget(g)

        # Legend scroll
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border:none; background:#161616;")
        self.legend = LegendWidget()
        self.scroll.setWidget(self.legend)
        v.addWidget(self.scroll)

    def set_task_hint(self, modality: str):
        task = _pick_task(modality)
        self.lbl_hint.setText(f"Detected: {modality} → task='{task}'")
        self.lbl_hint.setStyleSheet("color:#3af; font-size:9px;")

    def set_progress_visible(self, visible: bool):
        self.prog.setVisible(visible)

    def set_status(self, msg: str):
        self.lbl_status.setText(msg)

    def set_seg_done(self, active_labels: list):
        self.btn_tog.setEnabled(True)
        self.prog.setVisible(False)
        self.legend = LegendWidget(active_labels=active_labels)
        self.scroll.setWidget(self.legend)


class RegistrationPanel(QWidget):
    """Left-panel content for Register mode: CT/MRI folder pickers, run
    button, view-mode toggle (CT / MRI Registered / Overlay), progress log
    and quality metrics — ported from the Phase-4 registration prototype."""

    run_requested   = pyqtSignal()
    view_mode_requested = pyqtSignal(int)
    load_requested  = pyqtSignal()

    METRIC_ROWS = [
        ("NMI",       "nmi_rigid_cropped"),
        ("Edge Dice", "edge_dice_rigid_cropped"),
        ("NCC",       "ncc_rigid_cropped"),
        ("MAD",       "mad_rigid_cropped"),
        ("Mismatch",  "mismatch_rigid_cropped"),
        ("FoV Ratio", "fov_overlap_ratio"),
    ]

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setSpacing(5)

        g = _group("Registration Setup"); gl = QVBoxLayout(g)
        self.edit_ct  = self._path_row(gl, "CT DICOM Folder (fixed):")
        self.edit_mri = self._path_row(gl, "MRI DICOM Folder (moving):")
        self.edit_out = self._path_row(gl, "Output Directory:")
        gl.addWidget(_lbl("Patient ID:"))
        self.edit_pid = QLineEdit("Patient01")
        self.edit_pid.setStyleSheet(
            "QLineEdit{background:#181818;color:#ccc;border:1px solid #333;"
            "border-radius:3px;padding:2px;font-size:10px;}")
        gl.addWidget(self.edit_pid)
        v.addWidget(g)

        self.btn_run = _btn("🔗 Run Rigid Registration", highlight=True)
        self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self.run_requested)
        v.addWidget(self.btn_run)

        vm = _group("View Mode"); vml = QHBoxLayout(vm)
        self.btn_vm = []
        for i, label in enumerate(["CT", "MRI Reg.", "Overlay"]):
            b = _btn(label, small=True); b.setFixedHeight(24)
            b.clicked.connect(lambda _, idx=i: self.view_mode_requested.emit(idx))
            vml.addWidget(b); self.btn_vm.append(b)
        v.addWidget(vm)

        lg = _group("Registration Log"); lgl = QVBoxLayout(lg)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(130)
        self.log.setStyleSheet(
            "QTextEdit{background:#0d0d0d; color:#8f8; font-family:'Courier New';"
            "font-size:9px; border:1px solid #1a1a1a; border-radius:3px;}")
        self.prog = QProgressBar(); self.prog.setRange(0, 0); self.prog.setVisible(False)
        self.prog.setFixedHeight(5)
        self.prog.setStyleSheet(
            "QProgressBar{border:none;background:#222;border-radius:2px;}"
            "QProgressBar::chunk{background:#fa0;border-radius:2px;}")
        lgl.addWidget(self.log); lgl.addWidget(self.prog)
        v.addWidget(lg)

        mg = _group("Registration Metrics"); mgl = QGridLayout(mg)
        self.metric_labels = {}
        for i, (name, key) in enumerate(self.METRIC_ROWS):
            lk = _lbl(f"{name}:", "color:#555; font-size:10px;")
            lv = _lbl("—", "color:#fa0; font-size:10px; font-weight:bold;")
            mgl.addWidget(lk, i, 0); mgl.addWidget(lv, i, 1)
            self.metric_labels[key] = lv
        v.addWidget(mg)

        self.btn_load = _btn("📥 Load Registered Volumes into Viewer")
        self.btn_load.setFixedHeight(28)
        self.btn_load.setEnabled(False)
        self.btn_load.clicked.connect(self.load_requested)
        v.addWidget(self.btn_load)
        v.addStretch()

    def _path_row(self, parent_layout, label_text):
        parent_layout.addWidget(_lbl(label_text, "color:#aaa; font-size:10px;"))
        row = QHBoxLayout()
        edit = QLineEdit(); edit.setPlaceholderText("Select folder...")
        edit.setStyleSheet(
            "QLineEdit{background:#181818;color:#ccc;border:1px solid #333;"
            "border-radius:3px;padding:2px;font-size:10px;}")
        btn = _btn("…", small=True); btn.setFixedWidth(28); btn.setFixedHeight(22)
        btn.clicked.connect(lambda: self._browse(edit))
        row.addWidget(edit); row.addWidget(btn)
        parent_layout.addLayout(row)
        return edit

    def _browse(self, edit):
        path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if path:
            edit.setText(path)

    def get_params(self):
        return (self.edit_ct.text().strip(), self.edit_mri.text().strip(),
                self.edit_out.text().strip(), self.edit_pid.text().strip() or "Patient01")

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.prog.setVisible(running)
        if running:
            self.log.clear()

    def append_log(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def set_metrics(self, metrics: dict):
        for key, lbl in self.metric_labels.items():
            val = metrics.get(key)
            lbl.setText(f"{val:.4f}" if val is not None else "—")

    def enable_load(self, enabled: bool):
        self.btn_load.setEnabled(enabled)

    def highlight_view_mode(self, idx: int):
        for i, b in enumerate(self.btn_vm):
            b.setStyleSheet("")  # reset then re-apply via helper below
        for i, b in enumerate(self.btn_vm):
            hi = (i == idx)
            p = "3px 6px"; bg = "#1a3a1a" if hi else "#222"; bc = "#3a3" if hi else "#333"
            hb = "#2a4a2a" if hi else "#2a2a2a"
            b.setStyleSheet(
                f"QPushButton{{background:{bg};color:#ccc;border:1px solid {bc};"
                f"border-radius:4px;padding:{p};}}"
                f"QPushButton:hover{{background:{hb};color:#fff;border-color:#3af;}}")


class PathPlanPanel(QWidget):
    """Left-panel content for Path Plan mode: mark a tumour point on the
    viewer, tune constraints, run the collision-free needle-path search
    (reusing the TotalSegmentator mask from Segment mode), and browse the
    ranked results."""

    mark_tumor_toggled  = pyqtSignal(bool)
    clear_tumor_requested = pyqtSignal()
    run_requested        = pyqtSignal()
    path_selected        = pyqtSignal(int)
    export_requested      = pyqtSignal()

    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setSpacing(5)

        g = _group("Tumour Target"); gl = QVBoxLayout(g)
        gl.addWidget(_lbl(
            "1. Load a CT + run Segmentation (Segment tab) with ribs "
            "included, then mark the tumour on the slice where it's "
            "visible.", "color:#777; font-size:9px;"))
        self.btn_mark = _btn("🎯 Mark Tumour Point (click image)")
        self.btn_mark.setCheckable(True)
        self.btn_mark.setFixedHeight(30)
        self.btn_mark.toggled.connect(self.mark_tumor_toggled)
        gl.addWidget(self.btn_mark)
        self.lbl_point = _lbl("Tumour point: not set", "color:#fa0; font-size:10px;")
        gl.addWidget(self.lbl_point)
        self.btn_clear = _btn("Clear Point", small=True); self.btn_clear.setFixedHeight(22)
        self.btn_clear.clicked.connect(self.clear_tumor_requested)
        gl.addWidget(self.btn_clear)
        v.addWidget(g)

        pg = _group("Constraints"); pgl = QGridLayout(pg)
        self.spin_angle  = self._spin(pgl, 0, "Max Angle (°)",  0, 90, 60)
        self.spin_minlen = self._spin(pgl, 1, "Min Length (mm)", 1, 300, 20)
        self.spin_maxlen = self._spin(pgl, 2, "Max Length (mm)", 1, 400, 150)
        self.spin_cone   = self._spin(pgl, 3, "Cone Angle (°)",  5, 60, 30)
        v.addWidget(pg)

        self.btn_run = _btn("🧭 Run Path Planning", highlight=True)
        self.btn_run.setFixedHeight(32)
        self.btn_run.clicked.connect(self.run_requested)
        v.addWidget(self.btn_run)

        lg = _group("Planning Log"); lgl = QVBoxLayout(lg)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(90)
        self.log.setStyleSheet(
            "QTextEdit{background:#0d0d0d; color:#8cf; font-family:'Courier New';"
            "font-size:9px; border:1px solid #1a1a1a; border-radius:3px;}")
        self.prog = QProgressBar(); self.prog.setRange(0, 0); self.prog.setVisible(False)
        self.prog.setFixedHeight(5)
        self.prog.setStyleSheet(
            "QProgressBar{border:none;background:#222;border-radius:2px;}"
            "QProgressBar::chunk{background:#3af;border-radius:2px;}")
        lgl.addWidget(self.log); lgl.addWidget(self.prog)
        v.addWidget(lg)

        rg = _group("Ranked Paths (shortest first)"); rgl = QVBoxLayout(rg)
        self.list_paths = QListWidget()
        self.list_paths.setFixedHeight(140)
        self.list_paths.setStyleSheet(
            "QListWidget{background:#181818;color:#ccc;border:1px solid #333;"
            "font-size:9px;} QListWidget::item:selected{background:#1a3a5a;color:#3af;}")
        self.list_paths.currentRowChanged.connect(self.path_selected)
        rgl.addWidget(self.list_paths)
        v.addWidget(rg)

        self.btn_export = _btn("💾 Export Ranked Paths (.fcsv)")
        self.btn_export.setFixedHeight(26)
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_requested)
        v.addWidget(self.btn_export)
        v.addStretch()

    def _spin(self, grid, row, label, lo, hi, default):
        grid.addWidget(_lbl(label, "color:#aaa; font-size:10px;"), row, 0)
        s = QDoubleSpinBox(); s.setRange(lo, hi); s.setValue(default); s.setDecimals(0)
        s.setStyleSheet(
            "QDoubleSpinBox{background:#181818;color:#ccc;border:1px solid #333;"
            "border-radius:3px;padding:2px;font-size:10px;}")
        grid.addWidget(s, row, 1)
        return s

    def get_params(self):
        return (self.spin_angle.value(), self.spin_minlen.value(),
                self.spin_maxlen.value(), self.spin_cone.value())

    def set_marking(self, on: bool):
        self.btn_mark.blockSignals(True)
        self.btn_mark.setChecked(on)
        self.btn_mark.blockSignals(False)
        self.btn_mark.setText("🎯 Marking… click the image" if on else "🎯 Mark Tumour Point (click image)")

    def set_point_label(self, mm_point):
        if mm_point is None:
            self.lbl_point.setText("Tumour point: not set")
        else:
            x, y, z = mm_point
            self.lbl_point.setText(f"Tumour point: ({x:.1f}, {y:.1f}, {z:.1f}) mm")

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.prog.setVisible(running)
        if running:
            self.log.clear()
            self.list_paths.clear()
            self.btn_export.setEnabled(False)

    def append_log(self, msg: str):
        self.log.append(msg)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def set_results(self, ranked: list):
        self.list_paths.clear()
        for i, p in enumerate(ranked):
            item = QListWidgetItem(
                f"#{i+1}  len={p['length']:.1f}mm  angle={p['angle_deg']:.1f}°  "
                f"organ≥{p['organ_clearance']:.1f}mm  rib≥{p['rib_clearance']:.1f}mm")
            self.list_paths.addItem(item)
        self.btn_export.setEnabled(bool(ranked))


class GestureInfoPanel(QWidget):
    """Shared gesture log shown in every mode's panel."""
    def __init__(self):
        super().__init__()
        v = QVBoxLayout(self); v.setSpacing(4); v.setContentsMargins(0,0,0,0)
        g = _group("Gesture Control"); gl = QVBoxLayout(g)
        self.lbl_gesture = QLabel("Waiting...")
        self.lbl_gesture.setStyleSheet(
            "color:#3af; font-size:12px; font-weight:bold; qproperty-alignment:AlignCenter;")
        gl.addWidget(self.lbl_gesture)
        self.lbl_lock = QLabel("🔓 UNLOCKED")
        self.lbl_lock.setStyleSheet(
            "color:#666; font-size:11px; qproperty-alignment:AlignCenter;")
        gl.addWidget(self.lbl_lock)
        v.addWidget(g)

    def set_gesture(self, token: str, mode: str, action: str):
        self.lbl_gesture.setText(f"{token}\n{action}")

    def set_lock(self, locked: bool):
        if locked:
            self.lbl_lock.setText("🔒 LOCKED")
            self.lbl_lock.setStyleSheet(
                "color:#f33; font-size:11px; font-weight:bold; qproperty-alignment:AlignCenter;")
        else:
            self.lbl_lock.setText("🔓 UNLOCKED")
            self.lbl_lock.setStyleSheet(
                "color:#666; font-size:11px; qproperty-alignment:AlignCenter;")


# ══════════════════════════════════════════════
#  Viewer canvas (centre)
# ══════════════════════════════════════════════

def apply_overlay(scan_slice, mask_slice, opacity):
    bgr = cv2.cvtColor(scan_slice, cv2.COLOR_GRAY2BGR)
    ov  = bgr.copy()
    for label, (_, colour) in ORGAN_COLOURS.items():
        m = (mask_slice == label)
        if m.any():
            ov[m] = colour
    return cv2.addWeighted(ov, opacity, bgr, 1 - opacity, 0)


def _crop_zoom_pan(arr2d, zoom, pan_x, pan_y, interp=cv2.INTER_LINEAR):
    """Apply the same zoom/pan crop-and-resize used everywhere in the
    viewer, to an arbitrary single-channel or 3-channel 2-D array."""
    if zoom == 1.0 and pan_x == 0 and pan_y == 0:
        return arr2d
    h, w = arr2d.shape[:2]
    new_h = max(1, int(h / zoom)); new_w = max(1, int(w / zoom))
    cy = h // 2 + pan_y; cx = w // 2 + pan_x
    y1 = max(0, cy - new_h // 2); y2 = min(h, y1 + new_h)
    x1 = max(0, cx - new_w // 2); x2 = min(w, x1 + new_w)
    if y2 > h: y1 = max(0, h - new_h); y2 = h
    if x2 > w: x1 = max(0, w - new_w); x2 = w
    return cv2.resize(arr2d[y1:y2, x1:x2], (w, h), interpolation=interp)


def _normalize_to_255(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)


def _draw_path_markers(bgr: np.ndarray) -> np.ndarray:
    """Draw a crosshair for the tumour point / selected needle entry point
    on whichever slice they fall on. bgr is modified and returned."""
    if state.raw_volume is None:
        return bgr

    def marker(voxel_zyx, colour, label):
        if voxel_zyx is None:
            return
        z, y, x = voxel_zyx
        if int(round(z)) != state.current_slice:
            return
        h, w = bgr.shape[:2]
        px = int(round(x)); py = int(round(y))
        if not (0 <= px < w and 0 <= py < h):
            return
        cv2.drawMarker(bgr, (px, py), colour, markerType=cv2.MARKER_CROSS,
                        markerSize=16, thickness=2)
        cv2.putText(bgr, label, (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)

    marker(state.tumor_voxel, (0, 220, 255), "Tumor")

    if state.selected_path_idx is not None and state.path_ranked and state.sitk_image is not None:
        p = state.path_ranked[state.selected_path_idx]
        try:
            ex, ey, ez = state.sitk_image.TransformPhysicalPointToIndex(p["entry"])
            marker((ez, ey, ex), (255, 120, 0), "Entry")
        except Exception:
            pass
    return bgr


def render_slice() -> QImage:
    """
    Render the current state into a QImage.
    Applies windowing, preprocessing filter, zoom+pan, overlay, and (in
    Register mode) CT / MRI-registered / Overlay view-mode blending plus
    needle-path markers.
    """
    if state.raw_volume is None:
        return None

    # ── Register-mode view blending (MRI / Overlay) ────────────
    if state.view_mode in (1, 2) and state.reg_mri_arr is not None:
        mz = state.reg_mri_arr.shape[0]
        vz = state.raw_volume.shape[0]
        mi = min(int(state.current_slice * mz / vz), mz - 1)

        if state.view_mode == 1:
            raw_mri = state.reg_mri_arr[mi]
            mri_u8  = _normalize_to_255(raw_mri)
            mri_u8  = _crop_zoom_pan(mri_u8, state.zoom, state.pan_x, state.pan_y)
            h, w    = mri_u8.shape
            return QImage(mri_u8.tobytes(), w, h, w, QImage.Format_Grayscale8)

        # Overlay: CT = blue, MRI = red, blend = green
        raw_ct  = state.raw_volume[state.current_slice]
        raw_mri = state.reg_mri_arr[mi]
        ct_u8   = _normalize_to_255(raw_ct)
        mri_u8  = _normalize_to_255(raw_mri)
        if ct_u8.shape != mri_u8.shape:
            mri_u8 = cv2.resize(mri_u8, (ct_u8.shape[1], ct_u8.shape[0]),
                                 interpolation=cv2.INTER_LINEAR)
        blended = cv2.addWeighted(ct_u8, 0.5, mri_u8, 0.5, 0)
        bgr = np.zeros((*blended.shape, 3), dtype=np.uint8)
        bgr[..., 0] = (ct_u8  * 0.6).astype(np.uint8)
        bgr[..., 2] = (mri_u8 * 0.6).astype(np.uint8)
        bgr[..., 1] = (blended * 0.4).astype(np.uint8)
        bgr = _crop_zoom_pan(bgr, state.zoom, state.pan_x, state.pan_y)
        h, w = bgr.shape[:2]
        return QImage(bgr.tobytes(), w, h, w * 3, QImage.Format_BGR888)

    raw = state.raw_volume[state.current_slice]
    lo  = state.wl - state.ww / 2
    hi  = state.wl + state.ww / 2
    img = ((np.clip(raw, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)

    # Preprocessing
    from preprocessing_engine import PreprocessingEngine
    eng = PreprocessingEngine()
    eng.active_filter = state.active_filter
    eng.strength      = state.filter_strength
    img = eng.apply(img)

    # Zoom + pan
    if state.zoom != 1.0 or state.pan_x != 0 or state.pan_y != 0:
        h, w  = img.shape
        new_h = int(h / state.zoom); new_w = int(w / state.zoom)
        cy = h // 2 + state.pan_y; cx = w // 2 + state.pan_x
        y1 = max(0, cy - new_h // 2); y2 = min(h, y1 + new_h)
        x1 = max(0, cx - new_w // 2); x2 = min(w, x1 + new_w)
        if y2 > h: y1 = max(0, h - new_h); y2 = h
        if x2 > w: x1 = max(0, w - new_w); x2 = w
        img = cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

    # Overlay
    if state.show_overlay and state.seg_mask is not None:
        mask = state.current_mask_slice
        raw_h, raw_w = raw.shape
        if mask.shape != (raw_h, raw_w):
            mask = cv2.resize(mask.astype(np.float32), (raw_w, raw_h),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)
        if state.zoom != 1.0 or state.pan_x != 0 or state.pan_y != 0:
            h0, w0 = mask.shape
            nh = int(h0 / state.zoom); nw = int(w0 / state.zoom)
            cy = h0 // 2 + state.pan_y; cx = w0 // 2 + state.pan_x
            y1 = max(0, cy - nh // 2); y2 = min(h0, y1 + nh)
            x1 = max(0, cx - nw // 2); x2 = min(w0, x1 + nw)
            if y2 > h0: y1 = max(0, h0 - nh); y2 = h0
            if x2 > w0: x1 = max(0, w0 - nw); x2 = w0
            mask = cv2.resize(mask[y1:y2, x1:x2].astype(np.float32),
                              (w0, h0), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        if mask.shape != img.shape:
            mask = cv2.resize(mask.astype(np.float32),
                              (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(np.int32)
        disp = apply_overlay(img, mask, state.opacity)
        disp = _draw_path_markers(disp)
        h, w = disp.shape[:2]
        return QImage(disp.tobytes(), w, h, w * 3, QImage.Format_BGR888)
    else:
        if state.tumor_voxel is not None or state.selected_path_idx is not None:
            disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            disp = _draw_path_markers(disp)
            h, w = disp.shape[:2]
            return QImage(disp.tobytes(), w, h, w * 3, QImage.Format_BGR888)
        h, w = img.shape
        return QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)


# ══════════════════════════════════════════════
#  Main Window
# ══════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture-Controlled Medical Imaging Workstation")
        self.setMinimumSize(1500, 860)
        self._apply_theme()

        self._preprocessor = PreprocessingEngine()
        self._dicom_path   = None
        self._marking_tumor = False
        self.reg_worker  = None
        self.path_worker = None

        self._build_ui()
        self._wire_signals()
        self._start_engine()

    # ── UI Construction ────────────────────────

    def _build_ui(self):
        root_widget = QWidget()
        self.setCentralWidget(root_widget)
        root_v = QVBoxLayout(root_widget)
        root_v.setSpacing(0)
        root_v.setContentsMargins(0, 0, 0, 0)

        # Mode bar (top strip)
        self.mode_bar = ModeBar()
        root_v.addWidget(self.mode_bar)

        # Main content row
        content = QWidget()
        content_h = QHBoxLayout(content)
        content_h.setSpacing(8)
        content_h.setContentsMargins(8, 8, 8, 8)
        root_v.addWidget(content)

        # ── LEFT: stacked mode panels ──────────
        self.left_stack = QStackedWidget()
        self.left_stack.setFixedWidth(260)

        self.panel_dicom  = DicomPanel()
        self.panel_filter = FilterPanel()
        self.panel_seg    = SegmentPanel()
        self.panel_reg    = RegistrationPanel()
        self.panel_path   = PathPlanPanel()

        def _placeholder(label):
            ph = QWidget()
            v  = QVBoxLayout(ph)
            v.addWidget(_lbl(f"{label} mode — coming soon",
                             "color:#555; font-size:11px;"))
            v.addStretch()
            return ph

        self.panel_measure = _placeholder("Measure")
        self.panel_review  = _placeholder("Review")

        # Insert panels at the exact stack index matching each Mode's
        # enum value, so left_stack index == mode.value - 1 always holds
        # regardless of Mode enum ordering/growth.
        mode_panels = {
            Mode.DICOM    : self.panel_dicom,
            Mode.FILTER   : self.panel_filter,
            Mode.SEGMENT  : self.panel_seg,
            Mode.REGISTER : self.panel_reg,
            Mode.PATH_PLAN: self.panel_path,
            Mode.MEASURE  : self.panel_measure,
            Mode.REVIEW   : self.panel_review,
        }
        for mode in Mode:
            self.left_stack.insertWidget(mode.value - 1, mode_panels[mode])

        content_h.addWidget(self.left_stack)

        # ── CENTRE: viewer ─────────────────────
        centre_v = QVBoxLayout()

        self.lbl_view_mode = QLabel("Mode: CT")
        self.lbl_view_mode.setAlignment(Qt.AlignCenter)
        self.lbl_view_mode.setStyleSheet(
            "color:#fa0; background:#111; font-size:11px; font-weight:bold;"
            "border:1px solid #252525; border-radius:4px; padding:3px;")
        centre_v.addWidget(self.lbl_view_mode)

        self.dicom_label = ClickableLabel()
        self.dicom_label.setAlignment(Qt.AlignCenter)
        self.dicom_label.setMinimumSize(700, 560)
        self.dicom_label.setText("Open a DICOM file or folder")
        self.dicom_label.setFont(QFont("Courier New", 12))
        self.dicom_label.setStyleSheet(
            "background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;")
        self.dicom_label.clicked.connect(self._on_viewer_clicked)

        self.slice_slider = QSlider(Qt.Vertical)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slider_slice)

        img_row = QHBoxLayout()
        img_row.addWidget(self.dicom_label)
        img_row.addWidget(self.slice_slider)
        centre_v.addLayout(img_row)

        nav_row = QHBoxLayout()
        self.btn_prev = _btn("◀  Prev"); self.btn_prev.setFixedHeight(28)
        self.btn_next = _btn("Next  ▶"); self.btn_next.setFixedHeight(28)
        self.btn_prev.clicked.connect(self._prev_slice)
        self.btn_next.clicked.connect(self._next_slice)
        self.lbl_slice = _lbl("Slice: — / —",
            "color:#aaa; font-size:11px; qproperty-alignment:AlignCenter;")
        nav_row.addWidget(self.btn_prev)
        nav_row.addWidget(self.lbl_slice)
        nav_row.addWidget(self.btn_next)
        centre_v.addLayout(nav_row)
        content_h.addLayout(centre_v, 3)

        # ── RIGHT: camera + gesture info ───────
        right_v = QVBoxLayout(); right_v.setSpacing(6)

        cam_title = QLabel("✋ Gesture Camera")
        cam_title.setStyleSheet(
            "color:#3af; font-size:12px; font-weight:bold;")
        cam_title.setAlignment(Qt.AlignCenter)
        right_v.addWidget(cam_title)

        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setFixedSize(360, 290)
        self.cam_label.setStyleSheet(
            "background:#0a0a0a; border:1px solid #2a2a2a; border-radius:6px;")
        right_v.addWidget(self.cam_label)

        self.lbl_cam_status = _lbl("● Camera: connecting...", "color:#888; font-size:10px;")
        right_v.addWidget(self.lbl_cam_status)

        self.gesture_info = GestureInfoPanel()
        right_v.addWidget(self.gesture_info)
        right_v.addStretch()

        btn_reset = _btn("↺  Reset View"); btn_reset.setFixedHeight(28)
        btn_reset.clicked.connect(self._reset_view)
        right_v.addWidget(btn_reset)

        content_h.addLayout(right_v, 1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("color:#666; background:#111;")
        self.status.showMessage("Ready — open a DICOM file to begin.")

    # ── Signal wiring ──────────────────────────

    def _wire_signals(self):
        # Mode bar
        self.mode_bar.mode_changed.connect(self._on_mode_changed)

        # DicomPanel
        self.panel_dicom.open_file_requested.connect(self._open_file)
        self.panel_dicom.open_folder_requested.connect(self._open_folder)
        self.panel_dicom.wl_changed.connect(self._on_wl)
        self.panel_dicom.ww_changed.connect(self._on_ww)
        self.panel_dicom.preset_requested.connect(self._preset)

        # FilterPanel
        self.panel_filter.filter_changed.connect(self._on_filter_change)
        self.panel_filter.strength_changed.connect(self._on_strength_change)
        self.panel_filter.apply_requested.connect(self._apply_filter)
        self.panel_filter.reset_requested.connect(self._reset_filter)

        # SegmentPanel
        self.panel_seg.run_requested.connect(self._run_segmentation)
        self.panel_seg.toggle_requested.connect(self._toggle_overlay)
        self.panel_seg.opacity_changed.connect(lambda v: self._set_opacity(v / 100.0))

        # RegistrationPanel
        self.panel_reg.run_requested.connect(self._run_registration)
        self.panel_reg.view_mode_requested.connect(self._set_view_mode)
        self.panel_reg.load_requested.connect(self._load_registered_into_viewer)

        # PathPlanPanel
        self.panel_path.mark_tumor_toggled.connect(self._mark_tumor_toggle)
        self.panel_path.clear_tumor_requested.connect(self._clear_tumor_point)
        self.panel_path.run_requested.connect(self._run_path_planning)
        self.panel_path.path_selected.connect(self._on_path_selected)
        self.panel_path.export_requested.connect(self._export_paths_fcsv)

    def _start_engine(self):
        # Gesture engine
        self.engine = GestureEngine()
        self.engine.frame_ready.connect(self._on_cam_frame)
        self.engine.lock_toggled.connect(self._toggle_lock)
        self.engine.reset_triggered.connect(self._reset_view)
        self.engine.overlay_toggled.connect(self._toggle_overlay)
        self.engine.seg_triggered.connect(self._run_segmentation)
        self.engine.window_changed.connect(self._on_window_gesture)
        self.engine.cycle_view_mode.connect(self._cycle_view_mode)

        # Dispatcher
        self.dispatcher = GestureDispatcher()
        self.engine.gesture_token.connect(self.dispatcher.on_gesture)
        self.engine.pan_moved.connect(self.dispatcher.on_pan)
        self.dispatcher.state_changed.connect(self._on_state_changed)
        self.dispatcher.gesture_executed.connect(self._on_gesture_executed)
        self.dispatcher.pan_executed.connect(
            lambda dx, dy: self.status.showMessage(f"Pan {dx:+d},{dy:+d}px"))

        self.engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")

    # ── Mode switching ─────────────────────────

    def _on_mode_changed(self, mode: Mode):
        # Map mode to stack index (0-based, matching Mode enum order)
        idx = mode.value - 1
        self.left_stack.setCurrentIndex(idx)
        self.status.showMessage(f"Mode: {MODE_ICONS[mode]} {MODE_LABELS[mode]}")
        self._refresh_viewer()

    # ── State change handler ───────────────────

    def _on_state_changed(self):
        """Called after any dispatcher action — re-render everything."""
        self._refresh_viewer()
        self.mode_bar.update_lock()
        self.gesture_info.set_lock(state.view_locked)
        self.panel_dicom.set_wl(state.wl)
        self.panel_dicom.set_ww(state.ww)

        # Check for special requests set by mode_maps actions
        if getattr(state, "_request_segmentation", False):
            state._request_segmentation = False
            self._run_segmentation()

    def _on_gesture_executed(self, token: str, mode: str, action: str):
        self.gesture_info.set_gesture(token, mode, action)
        self.status.showMessage(f"[{mode}] {token} → {action}")

    # ── Viewer refresh ─────────────────────────

    def _refresh_viewer(self):
        qimg = render_slice()
        if qimg is None:
            return

        # Draw lock border
        border = "border:1px solid #222;"
        if state.view_locked:
            border = "border:3px solid #f33;"
        elif state.ai_mode if hasattr(state, 'ai_mode') else False:
            border = "border:2px solid #f90;"
        self.dicom_label.setStyleSheet(
            f"background:#0d0d0d; color:#333; {border} border-radius:6px;")

        pix = QPixmap.fromImage(qimg).scaled(
            self.dicom_label.width() - 4,
            self.dicom_label.height() - 4,
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.dicom_label.setPixmap(pix)
        self.lbl_slice.setText(f"Slice: {state.current_slice+1} / {state.num_slices}")
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(state.current_slice)
        self.slice_slider.blockSignals(False)

    # ── Camera ─────────────────────────────────

    def _on_cam_frame(self, frame: np.ndarray):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    # ── DICOM loading ──────────────────────────

    def _open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open DICOM File", "", "DICOM (*.dcm);;All (*)")
        if path:
            self._dicom_path = path
            self._load(path)

    def _open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if path:
            self._dicom_path = path
            self._load(path)

    def _load(self, path):
        self.status.showMessage("Loading DICOM...")
        self.loader = DicomLoader(path)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.error.connect(lambda e: self.status.showMessage(f"Error: {e}"))
        self.loader.start()

    def _on_loaded(self, volume, sitk_image, meta):
        state.raw_volume    = volume
        state.seg_mask      = None
        state.show_overlay  = False
        state.modality      = meta.get("Modality", "CT").upper()
        state.current_slice = volume.shape[0] // 2
        state.zoom  = 1.0; state.pan_x = 0; state.pan_y = 0
        state.wl    = float(np.median(volume))
        state.ww    = float(volume.max() - volume.min())
        state.metadata = meta
        state.sitk_image = sitk_image
        # Registration / path-planning results belong to a previously
        # loaded volume - clear them on a fresh load.
        state.reg_ct_arr = state.reg_mri_arr = state.reg_ct_sitk = None
        state.reg_metrics = {}
        state.view_mode = 0
        state.tumor_point_mm = state.tumor_voxel = None
        state.path_candidates = []; state.path_ranked = []; state.selected_path_idx = None

        self.slice_slider.setRange(0, volume.shape[0] - 1)
        self.slice_slider.setValue(state.current_slice)
        self.panel_dicom.update_meta(meta)
        self.panel_dicom.set_wl(state.wl)
        self.panel_dicom.set_ww(state.ww)
        self.panel_seg.set_task_hint(state.modality)
        self.panel_reg.enable_load(False)
        self.panel_reg.set_metrics({})
        self.panel_reg.highlight_view_mode(0)
        self.panel_path.set_point_label(None)
        self.panel_path.set_results([])
        self._update_view_mode_label()

        self._refresh_viewer()
        self.status.showMessage(
            f"Loaded {volume.shape[0]} slices | {state.modality} | "
            f"{volume.shape[2]}×{volume.shape[1]}px"
        )

    # ── Windowing ──────────────────────────────

    def _on_wl(self, val):
        state.wl = float(val)
        self.panel_dicom.lbl_wl.setText(f"WL: {val}")
        self._refresh_viewer()

    def _on_ww(self, val):
        state.ww = float(val)
        self.panel_dicom.lbl_ww.setText(f"WW: {val}")
        self._refresh_viewer()

    def _preset(self, wl: int, ww: int):
        state.wl = float(wl); state.ww = float(ww)
        self.panel_dicom.set_wl(wl); self.panel_dicom.set_ww(ww)
        self._refresh_viewer()

    def _on_window_gesture(self, dwl: float, dww: float):
        state.adjust_wl(dwl)
        state.adjust_ww(dww)
        self.panel_dicom.set_wl(state.wl)
        self.panel_dicom.set_ww(state.ww)
        self._refresh_viewer()

    # ── Navigation ─────────────────────────────

    def _on_slider_slice(self, val):
        state.current_slice = val
        self._refresh_viewer()

    def _prev_slice(self):
        state.prev_slice(); self._refresh_viewer()

    def _next_slice(self):
        state.next_slice(); self._refresh_viewer()

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            state.prev_slice()
        else:
            state.next_slice()
        self._refresh_viewer()

    # ── Zoom / pan ─────────────────────────────

    def _reset_view(self):
        state.reset_all()
        self.mode_bar.update_lock()
        self.gesture_info.set_lock(False)
        self.panel_dicom.set_wl(state.wl)
        self.panel_dicom.set_ww(state.ww)
        self._refresh_viewer()
        self.status.showMessage("View reset")

    def _toggle_lock(self):
        state.toggle_lock()
        self.mode_bar.update_lock()
        self.gesture_info.set_lock(state.view_locked)
        msg = "🔒 View LOCKED" if state.view_locked else "🔓 View UNLOCKED"
        self.status.showMessage(msg)
        self._refresh_viewer()

    # ── Filter ─────────────────────────────────

    def _on_filter_change(self, name: str):
        state.active_filter = name
        self._refresh_viewer()
        self.status.showMessage(f"Filter: {name}")

    def _on_strength_change(self, val: float):
        state.filter_strength = val
        self.panel_filter.set_strength_label(val)
        self._refresh_viewer()

    def _apply_filter(self):
        if state.raw_volume is None:
            return
        self.status.showMessage("Applying filter to all slices...")
        eng = PreprocessingEngine()
        eng.active_filter = state.active_filter
        eng.strength      = state.filter_strength
        lo = state.wl - state.ww / 2; hi = state.wl + state.ww / 2
        windowed = ((np.clip(state.raw_volume, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
        state.filtered_volume = np.stack(
            [eng.apply(sl) for sl in windowed], axis=0)
        self.status.showMessage(
            f"Filter '{state.active_filter}' applied to {state.num_slices} slices.")

    def _reset_filter(self):
        state.active_filter   = "None (Original)"
        state.filter_strength = 1.0
        state.filtered_volume = None
        self.panel_filter.combo.setCurrentIndex(0)
        self.panel_filter.sld.setValue(10)
        self._refresh_viewer()
        self.status.showMessage("Filter reset")

    # ── Segmentation ───────────────────────────

    def _run_segmentation(self):
        if self._dicom_path is None or state.raw_volume is None:
            self.status.showMessage("Open a DICOM first.")
            return
        self.panel_seg.btn_run.setEnabled(False)
        self.panel_seg.set_progress_visible(True)
        self.panel_seg.set_status("Starting TotalSegmentator...")
        self.status.showMessage("Segmentation running...")

        self.seg_worker = SegmentationWorker(self._dicom_path, state.modality)
        self.seg_worker.progress.connect(self.panel_seg.set_status)
        self.seg_worker.finished.connect(self._on_seg_done)
        self.seg_worker.error.connect(self._on_seg_error)
        self.seg_worker.start()

    def _on_seg_done(self, mask):
        state.seg_mask     = mask
        state.show_overlay = True
        active = [l for l in ORGAN_COLOURS if np.any(mask == l)]
        self.panel_seg.set_seg_done(active)
        self.panel_seg.btn_run.setEnabled(True)
        self.panel_seg.set_status(f"✅ Done — {len(active)} structures found.")
        self.status.showMessage(f"Segmentation complete — {len(active)} organs")
        self._refresh_viewer()

    def _on_seg_error(self, msg):
        self.panel_seg.set_progress_visible(False)
        self.panel_seg.btn_run.setEnabled(True)
        self.panel_seg.set_status(f"❌ {msg}")
        self.status.showMessage("Segmentation error — see panel")

    def _toggle_overlay(self):
        state.toggle_overlay()
        self._refresh_viewer()

    def _set_opacity(self, val: float):
        state.opacity = val
        self._refresh_viewer()

    # ── Registration ───────────────────────────

    def _run_registration(self):
        ct_dir, mri_dir, out_dir, pid = self.panel_reg.get_params()
        if not ct_dir or not mri_dir or not out_dir:
            self.status.showMessage("Please fill in CT dir, MRI dir, and output dir.")
            return
        self.panel_reg.set_running(True)
        self.panel_reg.append_log("Starting registration pipeline...")
        self.status.showMessage("Registration running...")

        self.reg_worker = RegistrationWorker(ct_dir, mri_dir, out_dir, pid)
        self.reg_worker.progress.connect(self.panel_reg.append_log)
        self.reg_worker.progress.connect(lambda m: self.status.showMessage(m.split("\n")[0]))
        self.reg_worker.finished.connect(self._on_reg_done)
        self.reg_worker.error.connect(self._on_reg_error)
        self.reg_worker.start()

    def _on_reg_done(self, result):
        self.panel_reg.set_running(False)
        self.panel_reg.enable_load(True)
        state.reg_ct_arr  = result["ct_arr"]
        state.reg_mri_arr = result["mri_arr"]
        state.reg_ct_sitk = result.get("ct_sitk")
        state.reg_metrics = result["metrics"]
        state.reg_out_dir = result["out_dir"]
        self.panel_reg.set_metrics(result["metrics"])
        self.panel_reg.append_log(f"\nDone! Output: {result['out_dir']}")
        self.status.showMessage(f"Registration complete → {result['out_dir']}")
        self._load_registered_into_viewer()

    def _on_reg_error(self, msg):
        self.panel_reg.set_running(False)
        self.panel_reg.append_log(f"\n❌ {msg}")
        self.status.showMessage("Registration error — see Register panel log")

    def _load_registered_into_viewer(self):
        if state.reg_ct_arr is None:
            self.status.showMessage("No registration result yet.")
            return
        state.raw_volume   = state.reg_ct_arr
        state.sitk_image   = state.reg_ct_sitk
        state.seg_mask      = None
        state.show_overlay  = False
        state.current_slice = state.reg_ct_arr.shape[0] // 2
        state.zoom = 1.0; state.pan_x = 0; state.pan_y = 0
        state.wl = float(np.median(state.reg_ct_arr))
        state.ww = float(state.reg_ct_arr.max() - state.reg_ct_arr.min())
        state.view_mode = 0
        self.panel_reg.highlight_view_mode(0)
        self._update_view_mode_label()

        self.slice_slider.setRange(0, state.reg_ct_arr.shape[0] - 1)
        self.slice_slider.setValue(state.current_slice)
        self.panel_dicom.set_wl(state.wl)
        self.panel_dicom.set_ww(state.ww)
        self._refresh_viewer()
        self.status.showMessage(
            f"Registered volumes loaded — {state.reg_ct_arr.shape[0]} slices | "
            f"Mode: {VIEW_MODES[state.view_mode]}")

    def _set_view_mode(self, idx: int):
        if idx in (1, 2) and state.reg_mri_arr is None:
            self.status.showMessage("Run registration first to view registered MRI / overlay.")
            return
        state.view_mode = idx
        self.panel_reg.highlight_view_mode(idx)
        self._update_view_mode_label()
        self._refresh_viewer()

    def _cycle_view_mode(self):
        """Gesture: BOTH_PALM held ~1s → cycle CT → MRI Registered → Overlay."""
        next_mode = (state.view_mode + 1) % len(VIEW_MODES)
        if next_mode != 0 and state.reg_mri_arr is None:
            self.status.showMessage("No registered MRI — run registration first.")
            return
        state.view_mode = next_mode
        self.panel_reg.highlight_view_mode(next_mode)
        self._update_view_mode_label()
        self._refresh_viewer()
        self.gesture_info.set_gesture("BOTH_PALM", "REGISTER", f"View → {VIEW_MODES[next_mode]}")

    def _update_view_mode_label(self):
        colors = ["#3af", "#fa0", "#3f3"]
        icons  = ["🔵", "🟡", "🟢"]
        mode = state.view_mode
        self.lbl_view_mode.setText(f"{icons[mode]}  Mode: {VIEW_MODES[mode]}")
        self.lbl_view_mode.setStyleSheet(
            f"color:{colors[mode]}; background:#111; font-size:11px; font-weight:bold;"
            f"border:1px solid #252525; border-radius:4px; padding:3px;")

    # ── Path Planning ──────────────────────────

    def _mark_tumor_toggle(self, on: bool):
        self._marking_tumor = on
        self.panel_path.set_marking(on)
        if on:
            self.status.showMessage("Click on the image to mark the tumour point.")

    def _label_pos_to_volume_xy(self, lx: int, ly: int):
        """Convert a click position on self.dicom_label (widget pixels) to
        (x, y) pixel coordinates in the *original* current slice, accounting
        for the letterboxed/aspect-fit pixmap and the current zoom/pan crop.
        Returns None if the click landed outside the actual image or no
        volume is loaded."""
        if state.raw_volume is None:
            return None
        pix = self.dicom_label.pixmap()
        if pix is None or pix.isNull():
            return None

        raw_h, raw_w = state.raw_volume[state.current_slice].shape
        # pixmap was scaled with KeepAspectRatio to fit the label -> letterboxed
        lbl_w, lbl_h = self.dicom_label.width(), self.dicom_label.height()
        pix_w, pix_h = pix.width(), pix.height()
        off_x = (lbl_w - pix_w) / 2.0
        off_y = (lbl_h - pix_h) / 2.0
        px, py = lx - off_x, ly - off_y
        if not (0 <= px < pix_w and 0 <= py < pix_h):
            return None

        # pixmap dimensions correspond 1:1 to the (zoomed/panned) rendered
        # image, which itself has shape (raw_h, raw_w) - undo scaling.
        disp_x = px / pix_w * raw_w
        disp_y = py / pix_h * raw_h

        # Undo the zoom/pan crop-and-resize to get back to raw pixel coords.
        if state.zoom != 1.0 or state.pan_x != 0 or state.pan_y != 0:
            new_h = max(1, int(raw_h / state.zoom)); new_w = max(1, int(raw_w / state.zoom))
            cy = raw_h // 2 + state.pan_y; cx = raw_w // 2 + state.pan_x
            y1 = max(0, cy - new_h // 2); y2 = min(raw_h, y1 + new_h)
            x1 = max(0, cx - new_w // 2); x2 = min(raw_w, x1 + new_w)
            if y2 > raw_h: y1 = max(0, raw_h - new_h); y2 = raw_h
            if x2 > raw_w: x1 = max(0, raw_w - new_w); x2 = raw_w
            orig_x = x1 + disp_x * (x2 - x1) / raw_w
            orig_y = y1 + disp_y * (y2 - y1) / raw_h
        else:
            orig_x, orig_y = disp_x, disp_y

        return int(round(orig_x)), int(round(orig_y))

    def _on_viewer_clicked(self, lx: int, ly: int):
        if not self._marking_tumor:
            return
        xy = self._label_pos_to_volume_xy(lx, ly)
        if xy is None:
            return
        x, y = xy
        z = state.current_slice
        state.tumor_voxel = (z, y, x)

        if state.sitk_image is not None:
            try:
                mm = state.sitk_image.TransformIndexToPhysicalPoint((int(x), int(y), int(z)))
            except Exception:
                mm = (float(x), float(y), float(z))
        else:
            mm = (float(x), float(y), float(z))
        state.tumor_point_mm = mm

        self.panel_path.set_point_label(mm)
        self.status.showMessage(f"Tumour point set at slice {z+1}, ({x},{y}) → {mm} mm")
        self._refresh_viewer()

    def _clear_tumor_point(self):
        state.tumor_point_mm = None
        state.tumor_voxel = None
        self.panel_path.set_point_label(None)
        self._refresh_viewer()

    def _run_path_planning(self):
        if state.raw_volume is None:
            self.status.showMessage("Open a CT volume first.")
            return
        if state.modality.upper() not in ("CT",):
            self.status.showMessage("Path planning requires a CT volume.")
            return
        if state.seg_mask is None:
            self.status.showMessage("Run Segmentation first (Segment tab) — path planning "
                                     "reuses the organ/rib mask.")
            return
        if state.tumor_point_mm is None:
            self.status.showMessage("Mark the tumour point first.")
            return
        if state.sitk_image is None:
            self.status.showMessage("No geometry information for the loaded volume.")
            return

        max_angle, min_len, max_len, cone_angle = self.panel_path.get_params()
        self.panel_path.set_running(True)
        self.panel_path.append_log("Starting path planning...")
        self.status.showMessage("Path planning running...")

        self.path_worker = PathPlanWorker(
            state.raw_volume, state.seg_mask, state.sitk_image, state.tumor_point_mm,
            max_angle_deg=max_angle, min_length_mm=min_len, max_length_mm=max_len,
            cone_angle_deg=cone_angle, top_k=5)
        self.path_worker.progress.connect(self.panel_path.append_log)
        self.path_worker.progress.connect(lambda m: self.status.showMessage(m.split("\n")[0]))
        self.path_worker.finished.connect(self._on_pathplan_done)
        self.path_worker.error.connect(self._on_pathplan_error)
        self.path_worker.start()

    def _on_pathplan_done(self, result):
        self.panel_path.set_running(False)
        state.path_candidates = result["valid"]
        state.path_ranked = result["ranked"]
        state.selected_path_idx = 0 if result["ranked"] else None
        self.panel_path.set_results(result["ranked"])
        if result["ranked"]:
            self.panel_path.list_paths.setCurrentRow(0)
        self.status.showMessage(
            f"Path planning complete — {len(result['ranked'])} ranked path(s), "
            f"{len(result['valid'])} total collision-free.")
        self._refresh_viewer()

    def _on_pathplan_error(self, msg):
        self.panel_path.set_running(False)
        self.panel_path.append_log(f"\n❌ {msg}")
        self.status.showMessage("Path planning error — see Path Plan panel log")

    def _on_path_selected(self, idx: int):
        if 0 <= idx < len(state.path_ranked):
            state.selected_path_idx = idx
        else:
            state.selected_path_idx = None
        self._refresh_viewer()

    def _export_paths_fcsv(self):
        if not state.path_ranked:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Ranked Paths", "ranked_needle_paths.fcsv", "FCSV (*.fcsv)")
        if not path:
            return
        try:
            save_paths_to_fcsv(state.path_ranked, path, list_name="RankedNeedlePaths")
            self.status.showMessage(f"Exported {len(state.path_ranked)} paths → {path}")
        except Exception as e:
            self.status.showMessage(f"Export failed: {e}")

    # ── Resize ─────────────────────────────────

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_viewer()

    def closeEvent(self, event):
        self.engine.stop()
        self.engine.wait()
        event.accept()

    # ── Theme ──────────────────────────────────

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #1a1a1a; color: #ddd;
                font-family: 'Segoe UI', sans-serif; font-size: 11px;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #2a2a2a; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #3af; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }
            QSlider::sub-page:horizontal { background: #3af; border-radius: 2px; }
            QSlider::groove:vertical {
                width: 4px; background: #2a2a2a; border-radius: 2px;
            }
            QSlider::handle:vertical {
                background: #3af; width: 12px; height: 12px;
                margin: 0 -4px; border-radius: 6px;
            }
            QSlider::sub-page:vertical { background: #3af; border-radius: 2px; }
            QStatusBar { background: #111; color: #555; }
            QScrollArea { border: none; }
            QComboBox { background: #222; color: #ddd; border: 1px solid #333; }
        """)


# ══════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())