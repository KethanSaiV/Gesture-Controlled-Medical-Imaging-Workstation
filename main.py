"""
Phase 3 - GUI Integration
Gesture-Controlled Medical Imaging Workstation

Combines:
- Phase 1: DICOM Viewer
- Phase 2: Gesture Recognition

Layout:
┌─────────────────┬──────────────────┬─────────────────┐
│   Left Panel    │   DICOM Viewer   │  Webcam Feed    │
│  (controls +    │   (MRI/CT scan)  │ (gesture cam)   │
│   metadata)     │                  │                 │
└─────────────────┴──────────────────┴─────────────────┘
"""

import sys
import os
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import time
import threading

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
    QStatusBar, QGroupBox, QGridLayout
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
GESTURE_COOLDOWN = 1.0   # seconds

COL_GREEN  = (0, 255, 120)
COL_YELLOW = (0, 220, 255)
COL_RED    = (0, 80, 255)
COL_WHITE  = (255, 255, 255)
COL_BLUE   = (255, 180, 0)

GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom In/Out",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Activate AI",
    "POINT_UP"   : "Scroll Up",
    "PEACE"      : "Scroll Down",
}


# ──────────────────────────────────────────────
#  DICOM Loader Thread
# ──────────────────────────────────────────────
class DicomLoader(QThread):
    loaded = pyqtSignal(object, dict)
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
                self.loaded.emit(arr, self._meta(ds))

            elif os.path.isdir(self.path):
                reader     = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(self.path)
                if not series_ids:
                    self.error.emit("No DICOM series found.")
                    return
                files = reader.GetGDCMSeriesFileNames(self.path, series_ids[0])
                reader.SetFileNames(files)
                image = reader.Execute()
                arr   = sitk.GetArrayFromImage(image).astype(np.float32)
                ds    = pydicom.dcmread(files[0])
                self.loaded.emit(arr, self._meta(ds))
            else:
                self.error.emit("Invalid path.")
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _meta(ds):
        def s(tag, d="N/A"): return str(getattr(ds, tag, d))
        return {
            "Patient"   : s("PatientName"),
            "Modality"  : s("Modality"),
            "Study Date": s("StudyDate"),
            "Rows"      : s("Rows"),
            "Columns"   : s("Columns"),
            "Institution": s("InstitutionName"),
        }


# ──────────────────────────────────────────────
#  Gesture Engine (runs in background thread)
# ──────────────────────────────────────────────
class GestureEngine(QThread):
    gesture_detected = pyqtSignal(str)          # gesture name
    frame_ready      = pyqtSignal(np.ndarray)   # annotated webcam frame

    def __init__(self):
        super().__init__()
        self.running        = True
        self.prev_positions = []
        self.smooth_window  = 5
        self.last_time      = {}
        self._zoom_factor   = 1.0

    def run(self):
        model_path = self._get_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        options    = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            gesture   = None
            landmarks = []

            if result.hand_landmarks:
                hand_lms  = result.hand_landmarks[0]
                landmarks = [(lm.x, lm.y) for lm in hand_lms]
                self._draw_landmarks(frame, hand_lms, w, h)
                gesture = self._classify(landmarks)

            if gesture:
                self.gesture_detected.emit(gesture)

            # FPS
            now       = time.time()
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            frame = self._draw_overlay(frame, gesture, landmarks, fps, w, h)
            self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False

    # ── Drawing ───────────────────────────────
    def _draw_landmarks(self, frame, hand_lms, w, h):
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17),
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 180), -1)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(frame, pts[tip], 7, COL_YELLOW, -1)

    def _draw_overlay(self, frame, gesture, landmarks, fps, w, h):
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GREEN, 2)

        # Gesture banner
        if gesture:
            action = GESTURE_ACTIONS.get(gesture, gesture)
            cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
            cv2.putText(frame, action, (10, h - 25),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, COL_YELLOW, 2)
            cv2.putText(frame, gesture, (10, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GREEN, 1)
        else:
            cv2.rectangle(frame, (0, h - 30), (w, h), (15, 15, 15), -1)
            cv2.putText(frame, "Show hand to camera", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

        # Pinch line
        if landmarks:
            tx, ty = landmarks[4]
            ix, iy = landmarks[8]
            pt1  = (int(tx * w), int(ty * h))
            pt2  = (int(ix * w), int(iy * h))
            dist = np.hypot(tx - ix, ty - iy)
            col  = COL_RED if dist < 0.06 else COL_WHITE
            cv2.line(frame, pt1, pt2, col, 2)

        return frame

    # ── Gesture Classification ─────────────────
    def _classify(self, lm):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)

        wrist = lm[0]
        self.prev_positions.append(wrist)
        if len(self.prev_positions) > self.smooth_window:
            self.prev_positions.pop(0)

        gesture = None

        if num_up == 5:
            gesture = "OPEN_PALM"
        elif num_up == 0:
            gesture = "FIST"
        elif fingers_up == [0, 1, 0, 0, 0]:
            gesture = "POINT_UP"
        elif fingers_up == [0, 1, 1, 0, 0]:
            gesture = "PEACE"
        elif self._is_pinch(lm):
            gesture = "PINCH"
        elif len(self.prev_positions) >= self.smooth_window:
            dy = self.prev_positions[-1][1] - self.prev_positions[0][1]
            if dy < -0.06:
                gesture = "SWIPE_UP"
            elif dy > 0.06:
                gesture = "SWIPE_DOWN"

        if gesture:
            now  = time.time()
            last = self.last_time.get(gesture, 0)
            if now - last < GESTURE_COOLDOWN:
                gesture = None
            else:
                self.last_time[gesture] = now

        return gesture

    def _fingers_up(self, lm):
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        fingers = [1 if lm[tips[0]][0] < lm[pip[0]][0] else 0]
        for i in range(1, 5):
            fingers.append(1 if lm[tips[i]][1] < lm[pip[i]][1] else 0)
        return fingers

    def _is_pinch(self, lm):
        tx, ty = lm[4]
        ix, iy = lm[8]
        return np.hypot(tx - ix, ty - iy) < 0.04

    @staticmethod
    def _get_model():
        import urllib.request
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmark model...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
        return model_path


# ──────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume      = None
        self.current     = 0
        self.ww          = 400.0
        self.wl          = 40.0
        self.zoom        = 1.0
        self.ai_mode     = False
        self.gesture_engine = None

        self._setup_ui()
        self._start_gesture_engine()

    # ── UI ────────────────────────────────────
    def _setup_ui(self):
        self.setWindowTitle("Gesture-Controlled Medical Imaging Workstation — Phase 3")
        self.setMinimumSize(1400, 750)
        self._apply_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── LEFT PANEL ────────────────────────
        left = QVBoxLayout()
        left.setSpacing(6)

        # Open buttons
        btn_file   = QPushButton("📂  Open DICOM File")
        btn_folder = QPushButton("📁  Open DICOM Folder")
        btn_file.clicked.connect(self.open_file)
        btn_folder.clicked.connect(self.open_folder)
        for b in (btn_file, btn_folder):
            b.setFixedHeight(36)
            b.setStyleSheet(self._btn_style())
        left.addWidget(btn_file)
        left.addWidget(btn_folder)

        # Metadata
        meta_group = QGroupBox("Patient Info")
        meta_group.setStyleSheet(self._group_style())
        mg = QGridLayout(meta_group)
        self.meta_labels = {}
        for i, key in enumerate(["Patient","Modality","Study Date","Rows","Columns","Institution"]):
            lk = QLabel(f"{key}:")
            lk.setStyleSheet("color:#777; font-size:11px;")
            lv = QLabel("—")
            lv.setStyleSheet("color:#ddd; font-size:11px;")
            lv.setWordWrap(True)
            mg.addWidget(lk, i, 0)
            mg.addWidget(lv, i, 1)
            self.meta_labels[key] = lv
        left.addWidget(meta_group)

        # Windowing
        win_group = QGroupBox("Windowing")
        win_group.setStyleSheet(self._group_style())
        wl = QVBoxLayout(win_group)

        self.lbl_wl = QLabel("Window Level (WL): 40")
        self.lbl_ww = QLabel("Window Width  (WW): 400")
        for l in (self.lbl_wl, self.lbl_ww):
            l.setStyleSheet("color:#bbb; font-size:11px;")

        self.slider_wl = QSlider(Qt.Horizontal)
        self.slider_wl.setRange(-1000, 3000)
        self.slider_wl.setValue(40)
        self.slider_wl.valueChanged.connect(self._on_wl)

        self.slider_ww = QSlider(Qt.Horizontal)
        self.slider_ww.setRange(1, 4000)
        self.slider_ww.setValue(400)
        self.slider_ww.valueChanged.connect(self._on_ww)

        for w in (self.lbl_wl, self.slider_wl, self.lbl_ww, self.slider_ww):
            wl.addWidget(w)

        presets = QHBoxLayout()
        for name, (l, w) in [("Brain",(40,80)),("Lung",(-600,1500)),("Bone",(400,1800))]:
            pb = QPushButton(name)
            pb.setFixedHeight(26)
            pb.setStyleSheet(self._btn_style(small=True))
            pb.clicked.connect(lambda _, lv=l, wv=w: self._preset(lv, wv))
            presets.addWidget(pb)
        wl.addLayout(presets)
        left.addWidget(win_group)

        # Gesture log
        gesture_group = QGroupBox("Gesture Control")
        gesture_group.setStyleSheet(self._group_style())
        gl = QVBoxLayout(gesture_group)

        self.lbl_gesture = QLabel("Waiting for gesture...")
        self.lbl_gesture.setStyleSheet(
            "color:#3af; font-size:13px; font-weight:bold; qproperty-alignment:AlignCenter;"
        )
        self.lbl_gesture.setWordWrap(True)
        gl.addWidget(self.lbl_gesture)

        self.lbl_ai_mode = QLabel("AI Mode: OFF")
        self.lbl_ai_mode.setStyleSheet(
            "color:#888; font-size:11px; qproperty-alignment:AlignCenter;"
        )
        gl.addWidget(self.lbl_ai_mode)

        # Gesture reference
        guide_lines = [
            "Swipe Up   → Prev Slice",
            "Swipe Down → Next Slice",
            "Pinch      → Zoom",
            "Open Palm  → Reset",
            "Fist       → AI Mode",
            "Point Up   → Scroll Up",
            "Peace      → Scroll Down",
        ]
        for line in guide_lines:
            lbl = QLabel(line)
            lbl.setStyleSheet("color:#555; font-size:10px;")
            gl.addWidget(lbl)

        left.addWidget(gesture_group)

        # Slice info
        self.lbl_slice = QLabel("Slice: — / —")
        self.lbl_slice.setStyleSheet(
            "color:#aaa; font-size:12px; qproperty-alignment:AlignCenter;"
        )
        left.addWidget(self.lbl_slice)
        left.addStretch()

        btn_reset = QPushButton("↺  Reset View")
        btn_reset.setFixedHeight(32)
        btn_reset.setStyleSheet(self._btn_style())
        btn_reset.clicked.connect(self._reset_view)
        left.addWidget(btn_reset)

        root.addLayout(left, 1)

        # ── CENTRE: DICOM viewer ──────────────
        centre = QVBoxLayout()

        self.dicom_label = QLabel()
        self.dicom_label.setAlignment(Qt.AlignCenter)
        self.dicom_label.setMinimumSize(600, 520)
        self.dicom_label.setText("Open a DICOM file or folder")
        self.dicom_label.setFont(QFont("Courier New", 12))
        self.dicom_label.setStyleSheet(
            "background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;"
        )

        self.slice_slider = QSlider(Qt.Vertical)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slice_change)

        img_row = QHBoxLayout()
        img_row.addWidget(self.dicom_label)
        img_row.addWidget(self.slice_slider)
        centre.addLayout(img_row)

        nav = QHBoxLayout()
        btn_prev = QPushButton("◀  Prev")
        btn_next = QPushButton("Next  ▶")
        for b in (btn_prev, btn_next):
            b.setFixedHeight(32)
            b.setStyleSheet(self._btn_style())
        btn_prev.clicked.connect(self._prev_slice)
        btn_next.clicked.connect(self._next_slice)
        nav.addWidget(btn_prev)
        nav.addWidget(btn_next)
        centre.addLayout(nav)

        root.addLayout(centre, 3)

        # ── RIGHT: Webcam feed ────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        cam_title = QLabel("✋  Gesture Camera")
        cam_title.setStyleSheet("color:#3af; font-size:13px; font-weight:bold;")
        cam_title.setAlignment(Qt.AlignCenter)
        right.addWidget(cam_title)

        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setFixedSize(380, 300)
        self.cam_label.setStyleSheet(
            "background:#0a0a0a; border:1px solid #2a2a2a; border-radius:6px;"
        )
        self.cam_label.setText("Starting camera...")
        right.addWidget(self.cam_label)

        # Status indicators
        self.lbl_cam_status = QLabel("● Camera: connecting...")
        self.lbl_cam_status.setStyleSheet("color:#888; font-size:11px;")
        right.addWidget(self.lbl_cam_status)

        right.addStretch()
        root.addLayout(right, 1)

        # Status bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("color:#777; background:#151515;")
        self.status.showMessage("Ready — open a DICOM file to begin.")

    # ── Gesture Engine ────────────────────────
    def _start_gesture_engine(self):
        self.gesture_engine = GestureEngine()
        self.gesture_engine.gesture_detected.connect(self._on_gesture)
        self.gesture_engine.frame_ready.connect(self._on_cam_frame)
        self.gesture_engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:11px;")

    # ── Gesture Handler ───────────────────────
    def _on_gesture(self, gesture):
        action = GESTURE_ACTIONS.get(gesture, gesture)
        self.lbl_gesture.setText(f"{gesture}\n{action}")
        self.status.showMessage(f"Gesture: {gesture}  →  {action}")

        if gesture == "SWIPE_UP":
            self._prev_slice()
        elif gesture == "SWIPE_DOWN":
            self._next_slice()
        elif gesture == "OPEN_PALM":
            self._reset_view()
        elif gesture == "POINT_UP":
            self._prev_slice()
        elif gesture == "PEACE":
            self._next_slice()
        elif gesture == "PINCH":
            self._zoom_in()
        elif gesture == "FIST":
            self._toggle_ai_mode()

    def _toggle_ai_mode(self):
        self.ai_mode = not self.ai_mode
        if self.ai_mode:
            self.lbl_ai_mode.setText("AI Mode: ON 🤖")
            self.lbl_ai_mode.setStyleSheet(
                "color:#f90; font-size:11px; font-weight:bold; qproperty-alignment:AlignCenter;"
            )
            self.dicom_label.setStyleSheet(
                "background:#0d0d0d; color:#333; border:2px solid #f90; border-radius:6px;"
            )
            self.status.showMessage("AI Mode ACTIVATED — Phase 4 will add segmentation here")
        else:
            self.lbl_ai_mode.setText("AI Mode: OFF")
            self.lbl_ai_mode.setStyleSheet(
                "color:#888; font-size:11px; qproperty-alignment:AlignCenter;"
            )
            self.dicom_label.setStyleSheet(
                "background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;"
            )

    def _zoom_in(self):
        self.zoom = min(self.zoom + 0.1, 3.0)
        self._show_slice()

    # ── Webcam frame → Qt label ───────────────
    def _on_cam_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg)
        pix  = pix.scaled(
            self.cam_label.width(),
            self.cam_label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    # ── DICOM Loading ─────────────────────────
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM (*.dcm);;All (*)")
        if path: self._load(path)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if path: self._load(path)

    def _load(self, path):
        self.status.showMessage("Loading DICOM...")
        self.loader = DicomLoader(path)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.error.connect(lambda e: self.status.showMessage(f"Error: {e}"))
        self.loader.start()

    def _on_loaded(self, volume, meta):
        self.volume  = volume
        self.current = volume.shape[0] // 2
        self.zoom    = 1.0
        self.slice_slider.setRange(0, volume.shape[0] - 1)
        self.slice_slider.setValue(self.current)
        self._update_meta(meta)
        self._reset_windowing()
        self._show_slice()
        self.status.showMessage(
            f"Loaded  {volume.shape[0]} slices  —  {volume.shape[2]}×{volume.shape[1]} px  |  Use gestures to navigate"
        )

    # ── Slice Display ─────────────────────────
    def _show_slice(self):
        if self.volume is None:
            return
        raw = self.volume[self.current]
        img = self._apply_window(raw)

        # Zoom
        if self.zoom != 1.0:
            h, w   = img.shape
            new_h  = int(h / self.zoom)
            new_w  = int(w / self.zoom)
            cy, cx = h // 2, w // 2
            y1     = max(0, cy - new_h // 2)
            y2     = min(h, cy + new_h // 2)
            x1     = max(0, cx - new_w // 2)
            x2     = min(w, cx + new_w // 2)
            img    = img[y1:y2, x1:x2]
            img    = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        h, w  = img.shape
        qimg  = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)
        pix   = QPixmap.fromImage(qimg)
        pix   = pix.scaled(
            self.dicom_label.width()  - 4,
            self.dicom_label.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.dicom_label.setPixmap(pix)
        self.lbl_slice.setText(f"Slice:  {self.current + 1} / {self.volume.shape[0]}")

    def _apply_window(self, raw):
        lo  = self.wl - self.ww / 2
        hi  = self.wl + self.ww / 2
        img = np.clip(raw, lo, hi)
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        return img

    # ── Navigation ────────────────────────────
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
        if event.angleDelta().y() > 0:
            self._prev_slice()
        else:
            self._next_slice()

    # ── Windowing ─────────────────────────────
    def _on_wl(self, val):
        self.wl = float(val)
        self.lbl_wl.setText(f"Window Level (WL): {val}")
        self._show_slice()

    def _on_ww(self, val):
        self.ww = float(val)
        self.lbl_ww.setText(f"Window Width  (WW): {val}")
        self._show_slice()

    def _preset(self, wl, ww):
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
            self.zoom    = 1.0
            self.slice_slider.setValue(self.current)
            self._reset_windowing()
            self.status.showMessage("View reset.")

    # ── Metadata ──────────────────────────────
    def _update_meta(self, meta):
        for k, v in meta.items():
            if k in self.meta_labels:
                self.meta_labels[k].setText(v)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_slice()

    def closeEvent(self, event):
        if self.gesture_engine:
            self.gesture_engine.stop()
            self.gesture_engine.wait()
        event.accept()

    # ── Styles ────────────────────────────────
    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background:#1a1a1a; color:#ddd;
                font-family:'Segoe UI', sans-serif; font-size:12px;
            }
            QSlider::groove:horizontal {
                height:4px; background:#2a2a2a; border-radius:2px;
            }
            QSlider::handle:horizontal {
                background:#3af; width:13px; height:13px;
                margin:-5px 0; border-radius:7px;
            }
            QSlider::sub-page:horizontal { background:#3af; border-radius:2px; }
            QSlider::groove:vertical {
                width:4px; background:#2a2a2a; border-radius:2px;
            }
            QSlider::handle:vertical {
                background:#3af; width:13px; height:13px;
                margin:0 -5px; border-radius:7px;
            }
            QSlider::sub-page:vertical { background:#3af; border-radius:2px; }
            QStatusBar { background:#111; color:#666; }
        """)

    def _btn_style(self, small=False):
        p = "3px 8px" if small else "5px 12px"
        return f"""
            QPushButton {{
                background:#222; color:#ccc; border:1px solid #333;
                border-radius:4px; padding:{p};
            }}
            QPushButton:hover {{ background:#2a2a2a; color:#fff; border-color:#3af; }}
            QPushButton:pressed {{ background:#111; }}
        """

    def _group_style(self):
        return """
            QGroupBox {
                border:1px solid #2a2a2a; border-radius:6px;
                margin-top:8px; padding:6px; color:#666; font-size:11px;
            }
            QGroupBox::title { subcontrol-origin:margin; left:8px; color:#3af; }
        """


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())