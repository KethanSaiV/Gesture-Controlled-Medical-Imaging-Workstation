"""
Phase 4 - AI Segmentation Integration
Gesture-Controlled Medical Imaging Workstation

Segmentation backend: TotalSegmentator v2
  CT  → task="total"     (117 structures)
  MRI → task="total_mr"  ( 50 structures)

CONFIRMED WORKING: tested on pelvic MRI, 49 organs detected in 23s on CPU.
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

from scipy.ndimage import gaussian_filter, median_filter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
    QStatusBar, QGroupBox, QGridLayout, QProgressBar,
    QScrollArea, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont


# ──────────────────────────────────────────────
#  Organ Colour Map  (BGR)
# ──────────────────────────────────────────────
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
    22: ("Prostate",            (255, 150, 200)),
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
}

# ── TotalSegmentator filename stem → our label ID ──
# Confirmed from actual output of total_mr task
_NAME_TO_ID = {
    "spleen"                      : 1,
    "kidney_right"                : 2,
    "kidney_left"                 : 3,
    "gallbladder"                 : 4,
    "liver"                       : 5,
    "stomach"                     : 6,
    "pancreas"                    : 7,
    "adrenal_gland_right"         : 8,
    "adrenal_gland_left"          : 9,
    "aorta"                       : 10,
    "inferior_vena_cava"          : 11,
    "portal_vein_and_splenic_vein": 12,
    "iliac_artery_right"          : 13,
    "iliac_artery_left"           : 14,
    "iliac_vena_right"            : 15,   # confirmed spelling from output
    "iliac_vena_left"             : 16,   # confirmed spelling from output
    "iliac_vein_right"            : 15,   # fallback
    "iliac_vein_left"             : 16,
    "lung_right"                  : 17,
    "lung_upper_lobe_right"       : 17,
    "lung_lower_lobe_right"       : 17,
    "lung_middle_lobe_right"      : 17,
    "lung_left"                   : 18,
    "lung_upper_lobe_left"        : 18,
    "lung_lower_lobe_left"        : 18,
    "heart"                       : 19,
    "esophagus"                   : 20,
    "urinary_bladder"             : 21,
    "prostate"                    : 22,
    "uterus"                      : 22,
    "colon"                       : 23,
    "duodenum"                    : 24,
    "small_bowel"                 : 25,
    "sacrum"                      : 26,
    "femur_right"                 : 27,
    "femur_left"                  : 28,
    "hip_right"                   : 29,
    "hip_left"                    : 30,
    "vertebrae"                   : 31,
    "intervertebral_discs"        : 32,
    "gluteus_maximus_right"       : 33,
    "gluteus_maximus_left"        : 34,
    "gluteus_medius_right"        : 35,
    "gluteus_medius_left"         : 36,
    "gluteus_minimus_right"       : 37,
    "gluteus_minimus_left"        : 38,
    "iliopsoas_right"             : 39,
    "iliopsoas_left"              : 40,
    "autochthon_right"            : 41,
    "autochthon_left"             : 42,
    "spinal_cord"                 : 43,
    "scapula_right"               : 44,
    "scapula_left"                : 45,
    "clavicula_right"             : 46,
    "clavicula_left"              : 47,
    "humerus_right"               : 48,
    "humerus_left"                : 49,
    "brain"                       : 50,
}

# Per-gesture cooldowns (seconds)
GESTURE_COOLDOWN = {
    "SWIPE_UP"   : 0.5,   # fast slice nav
    "SWIPE_DOWN" : 0.5,
    "PINCH"      : 0.8,
    "FIST"       : 0.8,
    "OPEN_PALM"  : 1.2,
    "POINT_UP"   : 0.1,   # pan needs fast updates
    "PEACE"      : 0.8,
}

GESTURE_ACTIONS = {
    "SWIPE_UP"        : "Previous Slice",
    "SWIPE_DOWN"      : "Next Slice",
    "PINCH"           : "Zoom In",
    "FIST"            : "Zoom Out",
    "OPEN_PALM"       : "Reset View",
    "POINT_UP"        : "Pan Mode",
    # Two-hand gestures
    "BOTH_PEACE"      : "Lock View",
    "BOTH_L"          : "Unlock View",
    "BOTH_FIST"       : "Reset View",
    "BOTH_PALM"       : "Toggle Overlay",
}

COL_GREEN  = (0, 255, 120)
COL_YELLOW = (0, 220, 255)
COL_ORANGE = (0, 165, 255)


def _pick_task(modality: str) -> str:
    return "total_mr" if modality.upper() in ("MR", "MRI") else "total"


import cv2
import numpy as np
import time
import math
from collections import deque

from scipy.ndimage import gaussian_filter, median_filter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PyQt5.QtCore import QThread, pyqtSignal


COL_GREEN  = (0, 255, 120)
COL_YELLOW = (0, 220, 255)
COL_ORANGE = (0, 165, 255)
COL_RED    = (0,  60, 220)
COL_CYAN   = (255, 220, 0)
COL_WHITE  = (255, 255, 255)


class GestureEngine(QThread):
    """
    Medical-grade gesture engine implementing the 12-gesture chart.

    Signals:
        slice_navigate(int)    : +1 next, -1 prev
        zoom_changed(float)    : zoom delta (+0.05 or -0.05)
        pan_moved(float,float) : dx, dy normalised
        window_changed(float,float): dwl, dww
        lock_toggled()         : lock/unlock view
        reset_triggered()      : reset view
        segmentation_triggered(): run segmentation
        overlay_intensity(int) : +1 or -1
        recalibrate()          : recalibrate
        frame_ready(ndarray)   : camera frame
    """

    slice_navigate          = pyqtSignal(int)
    zoom_changed            = pyqtSignal(float)
    pan_moved               = pyqtSignal(float, float)
    window_changed          = pyqtSignal(float, float)
    lock_toggled            = pyqtSignal()
    reset_triggered         = pyqtSignal()
    segmentation_triggered  = pyqtSignal()
    overlay_intensity       = pyqtSignal(int)
    recalibrate             = pyqtSignal()
    frame_ready             = pyqtSignal(np.ndarray)

    # ── Tuning ─────────────────────────────────────────────────
    SWIPE_THRESHOLD     = 0.04   # normalised y movement to fire swipe
    SWIPE_COOLDOWN      = 0.35   # seconds between slice steps
    ZOOM_SENSITIVITY    = 2.5    # rotation degrees per zoom step
    PAN_DEADZONE        = 0.004  # min movement to pan (lowered)
    PAN_SCALE           = 1.5    # pan speed (increased — normalised coords are tiny)
    WINDOW_SCALE        = 0.015  # windowing sensitivity (unused — see _detect_windowing)
    LOCK_HOLD           = 2.0    # seconds to hold L-shape
    SEG_HOLD            = 3.0    # seconds to hold both thumbs up
    OVERLAY_COOLDOWN    = 0.3    # seconds between overlay steps

    def __init__(self):
        super().__init__()
        self.running = True

        # State
        self._last_swipe        = 0.0
        self._last_overlay      = 0.0

        # Rotation (zoom)
        self._prev_wrist_angle  = None
        self._rotation_accum    = 0.0

        # Pan
        self._pan_locked        = False   # True = index held 2s, panning active
        self._pan_point_start   = 0.0
        self._pan_prev_pos      = None
        self._pan_hold_done     = False

        # Lock
        self._lock_start        = 0.0
        self._lock_held         = False
        self._lock_fired        = False

        # Segmentation (both thumbs up 3s)
        self._seg_start         = 0.0
        self._seg_held          = False
        self._seg_fired         = False

        # Windowing (two index fingers)
        self._win_prev_dist     = None

        # Overlay (thumb + two fingers)
        self._overlay_prev_x    = None

        # Recalibrate (A-shape / two index tips meeting)
        self._recal_fired       = False
        self._recal_cooldown    = 0.0

        # Status display
        self._status_msg        = ""
        self._status_time       = 0.0

    def run(self):
        import os, urllib.request
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path
            )

        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options   = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        prev_time = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]
            now   = time.time()

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            # ── Parse hands ────────────────────────────────────
            left_lm  = None
            right_lm = None

            if result.hand_landmarks and result.handedness:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    lm       = [(lm.x, lm.y) for lm in hand_lms]
                    side     = result.handedness[i][0].category_name
                    color    = (0, 200, 100) if side == "Right" else (200, 100, 0)
                    self._draw_hand(frame, hand_lms, w, h, color)
                    if side == "Left":
                        left_lm = lm
                    else:
                        right_lm = lm

            # Use dominant hand for single-hand gestures
            # Priority: whichever is detected; if both, right = control
            ctrl_lm  = right_lm or left_lm
            other_lm = left_lm if right_lm else None

            # ── GESTURE 11: Swipe up/down → navigate slices ────
            if ctrl_lm:
                self._detect_swipe(ctrl_lm, now)

            # ── GESTURE 01: Pinch/spread → zoom ────────────────
            if ctrl_lm and not self._pan_locked:
                self._detect_rotation_zoom(ctrl_lm, now)
            else:
                self._pinch_prev_dist  = None
                self._pinch_zoom_accum = 0.0

            # ── GESTURE 02: Index point hold 2s → pan ──────────
            if ctrl_lm:
                self._detect_pan(ctrl_lm, now)

            # ── GESTURE 03: Two index fingers → windowing ───────
            if left_lm and right_lm:
                self._detect_windowing(left_lm, right_lm, now)
            else:
                self._win_prev_dist = None

            # ── GESTURE 04: L-shape hold 2s → lock/unlock ───────
            if ctrl_lm:
                self._detect_lock(ctrl_lm, now)

            # ── GESTURE 05: Three fingers → reset ───────────────
            if ctrl_lm:
                self._detect_reset(ctrl_lm, now)

            # ── GESTURE 06: Both thumbs up 3s → segmentation ────
            if left_lm and right_lm:
                self._detect_segmentation(left_lm, right_lm, now)
            else:
                self._seg_held  = False
                self._seg_fired = False

            # ── GESTURE 07: Thumb + two fingers → overlay ───────
            if left_lm and right_lm:
                self._detect_overlay_intensity(left_lm, right_lm, now)
            else:
                self._overlay_prev_x = None

            # ── GESTURE 08: A-shape (two index tips) → recal ────
            if left_lm and right_lm:
                self._detect_recalibrate(left_lm, right_lm, now)

            # ── Draw overlay ────────────────────────────────────
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            frame     = self._draw_ui(frame, ctrl_lm, left_lm, right_lm, fps, w, h, now)
            self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False

    # ──────────────────────────────────────────────────────────
    #  Gesture Detectors
    # ──────────────────────────────────────────────────────────

    def _detect_swipe(self, lm, now):
        """Gesture 11: Swipe up/down with any hand posture."""
        # Use wrist + middle finger base motion over 6 frame buffer
        if not hasattr(self, '_swipe_buf'):
            self._swipe_buf = deque(maxlen=6)
        self._swipe_buf.append((lm[0][1], now))
        if len(self._swipe_buf) < 4:
            return
        dy = self._swipe_buf[-1][0] - self._swipe_buf[0][0]
        dt = self._swipe_buf[-1][1] - self._swipe_buf[0][1]
        if dt < 0.001:
            return
        velocity = abs(dy) / dt
        if velocity < 0.15:   # must be fast
            return
        if now - self._last_swipe < self.SWIPE_COOLDOWN:
            return
        if dy < -self.SWIPE_THRESHOLD:
            self._last_swipe = now
            self._swipe_buf.clear()
            self.slice_navigate.emit(-1)
            self._status("Swipe UP → Prev Slice")
        elif dy > self.SWIPE_THRESHOLD:
            self._last_swipe = now
            self._swipe_buf.clear()
            self.slice_navigate.emit(1)
            self._status("Swipe DOWN → Next Slice")

    def _detect_rotation_zoom(self, lm, now):
        """
        Gesture 01 (REVISED): Pinch = zoom in, Spread = zoom out.
        Uses thumb-index distance normalised by palm size.
        Much more intuitive and doesn't conflict with swipe.
        """
        if not hasattr(self, '_pinch_prev_dist'):
            self._pinch_prev_dist = None
            self._pinch_zoom_accum = 0.0

        # Thumb tip to index tip distance
        tx, ty = lm[4]
        ix, iy = lm[8]
        dist   = math.hypot(tx - ix, ty - iy)

        # Normalise by palm size (wrist to middle MCP)
        palm   = math.hypot(lm[9][0]-lm[0][0], lm[9][1]-lm[0][1])
        if palm < 0.01:
            return
        dist_norm = dist / palm

        if self._pinch_prev_dist is None:
            self._pinch_prev_dist = dist_norm
            return

        delta = dist_norm - self._pinch_prev_dist
        self._pinch_prev_dist = dist_norm

        # Accumulate small deltas to avoid jitter
        self._pinch_zoom_accum += delta

        if abs(self._pinch_zoom_accum) > 0.08:
            direction = 1 if self._pinch_zoom_accum > 0 else -1
            self._pinch_zoom_accum = 0.0
            self.zoom_changed.emit(direction * 0.12)
            self._status(f"Pinch {'open' if direction > 0 else 'close'} → Zoom {'In' if direction > 0 else 'Out'}")

    def _detect_pan(self, lm, now):
        """Gesture 02: Index point → hold 2s → move = pan."""
        fingers = self._fingers_up(lm)
        is_pointing = (fingers[1] == 1 and fingers[2] == 0 and
                       fingers[3] == 0 and fingers[4] == 0)

        if is_pointing:
            if not self._pan_hold_done:
                if self._pan_point_start == 0.0:
                    self._pan_point_start = now
                held = now - self._pan_point_start
                if held >= 2.0:
                    self._pan_hold_done = True
                    self._pan_prev_pos  = lm[8]  # index tip
                    self._status("Pan ACTIVE — move hand")
                else:
                    # Show charging bar
                    self._status(f"Hold to pan... {held:.1f}s / 2.0s")
            else:
                # Panning active
                pos = lm[8]
                if self._pan_prev_pos is not None:
                    dx = pos[0] - self._pan_prev_pos[0]
                    dy = pos[1] - self._pan_prev_pos[1]
                    if abs(dx) > self.PAN_DEADZONE or abs(dy) > self.PAN_DEADZONE:
                        self.pan_moved.emit(
                            dx * self.PAN_SCALE,
                            dy * self.PAN_SCALE
                        )
                self._pan_prev_pos = pos
                self._pan_locked   = True
        else:
            # Reset pan state when not pointing
            self._pan_point_start = 0.0
            self._pan_hold_done   = False
            self._pan_prev_pos    = None
            self._pan_locked      = False

    def _detect_windowing(self, left_lm, right_lm, now):
        """
        Gesture 03: Two index fingers apart/together → Window Level (WL).

        Apart  (+distance) → WL increases (brighter)
        Together (-distance) → WL decreases (darker)

        WL controls brightness/contrast centre point.
        High sensitivity: delta * 1500 gives ~30-150 HU per gesture unit.
        """
        l_fingers = self._fingers_up(left_lm)
        r_fingers = self._fingers_up(right_lm)
        l_point = (l_fingers[1] == 1 and l_fingers[2] == 0 and
                   l_fingers[3] == 0 and l_fingers[4] == 0)
        r_point = (r_fingers[1] == 1 and r_fingers[2] == 0 and
                   r_fingers[3] == 0 and r_fingers[4] == 0)
        if not (l_point and r_point):
            self._win_prev_dist = None
            return

        lx, ly = left_lm[8]
        rx, ry = right_lm[8]
        dist   = math.hypot(rx - lx, ry - ly)

        if self._win_prev_dist is None:
            self._win_prev_dist = dist
            return

        delta = dist - self._win_prev_dist
        self._win_prev_dist = dist

        if abs(delta) > 0.003:   # lower threshold = more responsive
            # Emit WL change (first param), WW unchanged (second param = 0)
            dwl = delta * 1500   # high sensitivity: ~45-150 HU per cm of finger movement
            self.window_changed.emit(dwl, 0)
            self._status(f"WL {'up' if delta > 0 else 'down'} ({dwl:+.0f})")

    def _detect_lock(self, lm, now):
        """Gesture 04: L-shape single hand hold 2s → lock/unlock."""
        if self._is_L_shape(lm):
            if not self._lock_held:
                self._lock_held  = True
                self._lock_start = now
                self._lock_fired = False
            held = now - self._lock_start
            if held >= self.LOCK_HOLD and not self._lock_fired:
                self._lock_fired = True
                self.lock_toggled.emit()
                self._status("L-shape 2s → Lock/Unlock View")
        else:
            self._lock_held  = False
            self._lock_fired = False

    def _detect_reset(self, lm, now):
        """Gesture 05: Three fingers (index+middle+ring) → reset."""
        if not hasattr(self, '_reset_cooldown'):
            self._reset_cooldown = 0.0
        fingers = self._fingers_up(lm)
        three = (fingers[0] == 0 and fingers[1] == 1 and
                 fingers[2] == 1 and fingers[3] == 1 and fingers[4] == 0)
        if three and now - self._reset_cooldown > 2.0:
            self._reset_cooldown = now
            self.reset_triggered.emit()
            self._status("Three Fingers → Reset View")

    def _detect_segmentation(self, left_lm, right_lm, now):
        """Gesture 06: Both thumbs up hold 3s → run segmentation."""
        l_thumb = self._is_thumb_up(left_lm)
        r_thumb = self._is_thumb_up(right_lm)
        if l_thumb and r_thumb:
            if not self._seg_held:
                self._seg_held  = True
                self._seg_start = now
                self._seg_fired = False
            held = now - self._seg_start
            if held >= self.SEG_HOLD and not self._seg_fired:
                self._seg_fired = True
                self.segmentation_triggered.emit()
                self._status("Both Thumbs 3s → SEGMENTATION STARTED!")
            else:
                self._status(f"Hold both thumbs... {held:.1f}s / 3.0s")
        else:
            self._seg_held  = False
            self._seg_fired = False

    def _detect_overlay_intensity(self, left_lm, right_lm, now):
        """Gesture 07: Thumb up one hand + two fingers other → overlay."""
        l_thumb = self._is_thumb_up(left_lm)
        r_thumb = self._is_thumb_up(right_lm)
        l_two   = self._is_two_fingers(left_lm)
        r_two   = self._is_two_fingers(right_lm)

        if (l_thumb and r_two) or (r_thumb and l_two):
            two_lm = right_lm if l_thumb else left_lm
            # Track horizontal movement of the two-finger hand
            mid_x = (two_lm[8][0] + two_lm[12][0]) / 2
            if self._overlay_prev_x is not None:
                dx = mid_x - self._overlay_prev_x
                if abs(dx) > 0.02 and now - self._last_overlay > self.OVERLAY_COOLDOWN:
                    self._last_overlay = now
                    direction = 1 if dx > 0 else -1
                    self.overlay_intensity.emit(direction)
                    self._status(f"Overlay {'brighter' if direction > 0 else 'dimmer'}")
            self._overlay_prev_x = mid_x
        else:
            self._overlay_prev_x = None

    def _detect_recalibrate(self, left_lm, right_lm, now):
        """Gesture 08: Two index fingertips touch (A-shape) → recalibrate."""
        l_fingers = self._fingers_up(left_lm)
        r_fingers = self._fingers_up(right_lm)
        l_point = (l_fingers[1] == 1 and l_fingers[2] == 0)
        r_point = (r_fingers[1] == 1 and r_fingers[2] == 0)
        if not (l_point and r_point):
            self._recal_fired = False
            return
        dist = math.hypot(
            left_lm[8][0] - right_lm[8][0],
            left_lm[8][1] - right_lm[8][1]
        )
        if dist < 0.05 and not self._recal_fired and now - self._recal_cooldown > 3.0:
            self._recal_fired    = True
            self._recal_cooldown = now
            self.recalibrate.emit()
            self._status("A-shape → Recalibrate!")
        elif dist > 0.1:
            self._recal_fired = False

    # ──────────────────────────────────────────────────────────
    #  Hand Classifiers
    # ──────────────────────────────────────────────────────────

    def _fingers_up(self, lm):
        """Returns [thumb, index, middle, ring, pinky] 1=up 0=down."""
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        thumb = 1 if lm[tips[0]][0] > lm[pip[0]][0] else 0
        rest  = [1 if lm[tips[i]][1] < lm[pip[i]][1] else 0 for i in range(1, 5)]
        return [thumb] + rest

    def _is_thumb_up(self, lm) -> bool:
        """
        Thumb up: thumb tip clearly above wrist, all fingers curled.
        Uses wrist-distance method — works palm-facing-camera.
        """
        # Thumb tip must be well above wrist
        thumb_tip_y  = lm[4][1]
        wrist_y      = lm[0][1]
        thumb_above  = (wrist_y - thumb_tip_y) > 0.12   # at least 12% of frame height

        # All 4 fingers must be DOWN (tips below their PIP joints)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        fingers_down = all(lm[tips[i]][1] > lm[pips[i]][1] for i in range(4))

        return thumb_above and fingers_down

    def _is_two_fingers(self, lm) -> bool:
        """Index and middle extended, others down."""
        f = self._fingers_up(lm)
        return f[0] == 0 and f[1] == 1 and f[2] == 1 and f[3] == 0 and f[4] == 0

    def _is_L_shape(self, lm) -> bool:
        """Thumb + index extended, middle/ring/pinky down."""
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        idx_ext   = lm[tips[1]][1] < lm[pip[1]][1]
        mid_down  = lm[tips[2]][1] > lm[pip[2]][1]
        ring_down = lm[tips[3]][1] > lm[pip[3]][1]
        pink_down = lm[tips[4]][1] > lm[pip[4]][1]
        thu_ext   = abs(lm[tips[0]][0] - lm[0][0]) > 0.07
        return idx_ext and mid_down and ring_down and pink_down and thu_ext

    # ──────────────────────────────────────────────────────────
    #  Drawing
    # ──────────────────────────────────────────────────────────

    def _draw_hand(self, frame, hand_lms, w, h, color):
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17),
        ]
        pts = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 3, color, -1)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(frame, pts[tip], 6, COL_YELLOW, -1)

    def _status(self, msg):
        self._status_msg  = msg
        self._status_time = time.time()

    def _draw_ui(self, frame, ctrl_lm, left_lm, right_lm, fps, w, h, now):
        # FPS
        cv2.putText(frame, f"FPS:{fps:.0f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 2)

        # Pan hold charge bar
        if ctrl_lm and not self._pan_hold_done and self._pan_point_start > 0:
            held   = min(now - self._pan_point_start, 2.0)
            charge = held / 2.0
            bx, by = 8, 35
            cv2.rectangle(frame, (bx, by), (bx+120, by+7), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*120), by+7), COL_CYAN, -1)
            cv2.putText(frame, "PAN HOLD", (bx, by-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1)

        # Pan active indicator
        if self._pan_locked:
            cv2.putText(frame, "PAN ACTIVE", (w-100, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_CYAN, 2)

        # Lock charge bar
        if self._lock_held and not self._lock_fired:
            held   = min(now - self._lock_start, self.LOCK_HOLD)
            charge = held / self.LOCK_HOLD
            bx, by = 8, 52
            cv2.rectangle(frame, (bx, by), (bx+120, by+7), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*120), by+7), COL_ORANGE, -1)
            cv2.putText(frame, "LOCK HOLD", (bx, by-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1)

        # Segmentation charge bar
        if self._seg_held and not self._seg_fired:
            held   = min(now - self._seg_start, self.SEG_HOLD)
            charge = held / self.SEG_HOLD
            bx, by = w//2 - 80, 8
            cv2.rectangle(frame, (bx, by), (bx+160, by+10), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*160), by+10), (0, 200, 80), -1)
            cv2.putText(frame, f"SEGMENTATION {held:.1f}s/3s", (bx, by+22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,255,180), 1)

        # Status message (fades after 2s)
        if self._status_msg and now - self._status_time < 2.0:
            alpha = max(0, 1.0 - (now - self._status_time) / 2.0)
            color = tuple(int(c * alpha) for c in COL_WHITE)
            cv2.putText(frame, self._status_msg, (8, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

        # Bottom bar
        cv2.rectangle(frame, (0, h-32), (w, h), (10,10,10), -1)
        hints = [
            "Rotate=Zoom  Point+hold=Pan  3fingers=Reset",
            "BothThumbs3s=Segment  L-hold=Lock  2index=Window",
        ]
        hint = hints[int(now*0.5) % 2]   # alternates every 2s
        cv2.putText(frame, hint, (6, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100,100,100), 1)

        return frame




# ──────────────────────────────────────────────
#  Preprocessing Engine
#  All filters run on CPU via numpy/scipy/opencv
#  Applied per-slice at display time
#  Original volume is NEVER modified
# ──────────────────────────────────────────────
class PreprocessingEngine:
    FILTERS = [
        "None (Original)",
        "Gaussian Denoise",
        "Median Filter",
        "Sharpen",
        "CLAHE (Contrast)",
        "Unsharp Mask",
        "Edge Enhance",
        "Normalize",
    ]

    def __init__(self):
        self.active_filter = "None (Original)"
        self.strength      = 1.0   # 0.1 – 2.0

    def apply(self, img_uint8: np.ndarray) -> np.ndarray:
        """
        Apply active filter to a uint8 grayscale slice.
        Returns uint8 grayscale — never modifies input.
        """
        if self.active_filter == "None (Original)":
            return img_uint8

        img = img_uint8.astype(np.float32)
        s   = max(0.1, self.strength)

        if self.active_filter == "Gaussian Denoise":
            result = gaussian_filter(img, sigma=s * 1.2)

        elif self.active_filter == "Median Filter":
            size   = max(3, int(s * 3) | 1)   # must be odd
            result = median_filter(img, size=size)

        elif self.active_filter == "Sharpen":
            blur   = gaussian_filter(img, sigma=1.0)
            result = np.clip(img + s * (img - blur) * 1.5, 0, 255)

        elif self.active_filter == "CLAHE (Contrast)":
            clahe  = cv2.createCLAHE(
                clipLimit    = 2.0 + s * 2.0,
                tileGridSize = (8, 8)
            )
            result = clahe.apply(img_uint8).astype(np.float32)

        elif self.active_filter == "Unsharp Mask":
            blur   = gaussian_filter(img, sigma=s * 2.0)
            result = np.clip(img + s * 0.8 * (img - blur), 0, 255)

        elif self.active_filter == "Edge Enhance":
            sx     = cv2.Sobel(img_uint8, cv2.CV_32F, 1, 0, ksize=3)
            sy     = cv2.Sobel(img_uint8, cv2.CV_32F, 0, 1, ksize=3)
            edges  = np.sqrt(sx**2 + sy**2)
            edges  = edges / (edges.max() + 1e-6) * 255
            result = np.clip(img + s * 0.4 * edges, 0, 255)

        elif self.active_filter == "Normalize":
            lo     = np.percentile(img, 1)
            hi     = np.percentile(img, 99)
            result = np.clip((img - lo) / (hi - lo + 1e-6) * 255, 0, 255)

        else:
            result = img

        return result.astype(np.uint8)


# ──────────────────────────────────────────────
#  DICOM Loader
# ──────────────────────────────────────────────
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
                sitk_img = sitk.GetImageFromArray(arr)
                self.loaded.emit(arr, sitk_img, self._meta(ds))

            elif os.path.isdir(self.path):
                reader    = sitk.ImageSeriesReader()
                dicom_dir = self.path
                series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
                if not series_ids:
                    for sub in sorted(os.listdir(dicom_dir)):
                        sub_path = os.path.join(dicom_dir, sub)
                        if os.path.isdir(sub_path):
                            series_ids = reader.GetGDCMSeriesIDs(sub_path)
                            if series_ids:
                                dicom_dir = sub_path
                                break
                if not series_ids:
                    self.error.emit("No DICOM series found.")
                    return
                files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
                reader.SetFileNames(files)
                sitk_img = reader.Execute()
                arr      = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                ds       = pydicom.dcmread(files[0])
                self.loaded.emit(arr, sitk_img, self._meta(ds))
            else:
                self.error.emit("Invalid path.")
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _meta(ds):
        def s(tag, d="N/A"): return str(getattr(ds, tag, d))
        return {
            "Patient"    : s("PatientName"),
            "Modality"   : s("Modality"),
            "Study Date" : s("StudyDate"),
            "Rows"       : s("Rows"),
            "Columns"    : s("Columns"),
            "Institution": s("InstitutionName"),
        }


# ──────────────────────────────────────────────
#  TotalSegmentator Worker
# ──────────────────────────────────────────────
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
            task = _pick_task(self.modality)
            self.progress.emit(f"Modality: {self.modality} → Task: {task}\nConverting DICOM to NIfTI...")

            tmp_dir  = tempfile.mkdtemp(prefix="totalseg_")
            nii_in   = os.path.join(tmp_dir, "input.nii.gz")
            out_dir  = os.path.join(tmp_dir, "segs")
            os.makedirs(out_dir, exist_ok=True)

            # ── DICOM → NIfTI ──────────────────
            reader    = sitk.ImageSeriesReader()
            dicom_dir = self.dicom_path
            if os.path.isdir(self.dicom_path):
                series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
                if not series_ids:
                    for sub in sorted(os.listdir(dicom_dir)):
                        sub_path = os.path.join(dicom_dir, sub)
                        if os.path.isdir(sub_path):
                            series_ids = reader.GetGDCMSeriesIDs(sub_path)
                            if series_ids:
                                dicom_dir = sub_path
                                break
                if not series_ids:
                    self.error.emit("No DICOM series found.\nSelect the folder containing .dcm files directly.")
                    return
                files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            else:
                files = [self.dicom_path]

            reader.SetFileNames(files)
            sitk.WriteImage(reader.Execute(), nii_in)
            self.progress.emit(f"NIfTI ready. Running TotalSegmentator ({task})...\n⏳ ~20-30 sec on CPU")

            # ── Run TotalSegmentator ───────────
            cmd = [
                sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
                "-i", nii_in,
                "-o", out_dir,
                "-ta", task,
                "-d", "cpu",
                "--fast",
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                line = line.strip()
                if line and not line.startswith("W0") and not line.startswith("E0"):
                    self.progress.emit(line[:100])
            proc.wait()

            if proc.returncode != 0:
                self.error.emit(f"TotalSegmentator failed (code {proc.returncode}).")
                return

            self.progress.emit("Merging organ masks...")

            # ── Merge per-organ masks → label volume ──
            seg_files = [f for f in os.listdir(out_dir)
                         if f.endswith(".nii.gz") or f.endswith(".nii")]

            if not seg_files:
                self.error.emit("No output files found from TotalSegmentator.")
                return

            label_vol = None
            merged    = 0
            for fname in sorted(seg_files):
                stem     = fname.replace(".nii.gz", "").replace(".nii", "").lower()
                label_id = _NAME_TO_ID.get(stem)
                if label_id is None:
                    continue
                organ_img = sitk.ReadImage(os.path.join(out_dir, fname))
                organ_arr = sitk.GetArrayFromImage(organ_img).astype(np.uint8)
                if label_vol is None:
                    label_vol = np.zeros_like(organ_arr, dtype=np.int32)
                label_vol[organ_arr > 0] = label_id
                merged += 1

            if label_vol is None or label_vol.max() == 0:
                self.error.emit(
                    f"Mask is empty — {len(seg_files)} files found but none matched organ map.\n"
                    f"First few files: {seg_files[:5]}"
                )
                return

            self.progress.emit(f"✅ Segmentation complete! {merged} organs merged.")
            self.finished.emit(label_vol)

        except Exception as e:
            import traceback
            self.error.emit(f"Error: {str(e)}\n{traceback.format_exc()}")
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
#  Two-Hand Gesture Engine
# ──────────────────────────────────────────────
class TwoHandGestureEngine(QThread):
    gesture_detected = pyqtSignal(str, str)
    frame_ready      = pyqtSignal(np.ndarray)
    ai_mode_changed  = pyqtSignal(bool)
    two_hand_gesture = pyqtSignal(str)    # "BOTH_PEACE", "BOTH_L", "BOTH_FIST", "BOTH_PALM"
    pan_moved        = pyqtSignal(float, float)  # dx, dy normalised

    TWO_HAND_HOLD     = 0.6  # seconds both hands must hold gesture to fire
    PAN_SMOOTH        = 6    # frames of position history for pan smoothing

    def __init__(self):
        super().__init__()
        self.running              = True
        self.prev_pos_right       = []
        self.prev_pos_left        = []
        self.smooth_window        = 4   # reduced for faster swipe response
        self.last_time            = {}
        self._left_fist_held      = False
        self._left_fist_start     = 0
        # Two-hand gesture state
        self._two_hand_gesture    = None   # current matching two-hand gesture
        self._two_hand_start      = 0.0
        self._two_hand_fired      = False
        self._last_two_hand       = {}     # cooldown per two-hand gesture
        # Pan
        self._pan_history         = []
        self._pan_active          = False
        self._pan_prev_pos        = None

    def run(self):
        model_path = self._get_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        options    = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.2,
            min_hand_presence_confidence=0.2,
            min_tracking_confidence=0.2,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time      = time.time()
        ai_mode_active = False

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            left_landmarks  = None
            right_landmarks = None

            if result.hand_landmarks and result.handedness:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    lm_list    = [(lm.x, lm.y) for lm in hand_lms]
                    handedness = result.handedness[i][0].category_name
                    # After cv2.flip: Left=Left, Right=Right (confirmed by debug)
                    if handedness == "Left":
                        left_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (200, 100, 0))
                    else:
                        right_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (0, 200, 100))

            left_gesture = None
            if left_landmarks:
                left_gesture = self._classify_static(left_landmarks)
                if left_gesture == "FIST":
                    if not self._left_fist_held:
                        self._left_fist_held  = True
                        self._left_fist_start = time.time()
                    if time.time() - self._left_fist_start > 1.0:
                        ai_mode_active = not ai_mode_active
                        self.ai_mode_changed.emit(ai_mode_active)
                        self._left_fist_held = False
                else:
                    self._left_fist_held = False

            if right_landmarks:
                gesture = self._classify_full(right_landmarks, "right")

                # ── POINT_UP + ACTIVE MOVEMENT → pan ──────
                # Only consume the gesture if hand is actually moving
                # If hand is still with POINT_UP, let SWIPE_UP fire normally
                fingers_up = self._fingers_up(right_landmarks)
                is_point   = (fingers_up == [0, 1, 0, 0, 0])
                if is_point and self._pan_prev_pos is not None:
                    wrist_pos = right_landmarks[0]
                    dx = wrist_pos[0] - self._pan_prev_pos[0]
                    dy = wrist_pos[1] - self._pan_prev_pos[1]
                    if abs(dx) > 0.012 or abs(dy) > 0.012:
                        # Actively moving — pan mode
                        self.pan_moved.emit(dx, dy)
                        self._pan_active = True
                        gesture = None   # consume: panning, not swiping
                    else:
                        # Stationary POINT_UP — let swipe/gesture fire
                        self._pan_active = False
                    self._pan_prev_pos = wrist_pos
                elif is_point:
                    self._pan_prev_pos = right_landmarks[0]
                    self._pan_active   = False
                else:
                    self._pan_prev_pos = None
                    self._pan_active   = False

                if gesture:
                    self.gesture_detected.emit(gesture, "right")

            # ── Two-hand gesture detection ─────────────
            # Runs whenever BOTH hands are visible
            if left_landmarks and right_landmarks:
                left_g  = self._classify_static(left_landmarks)
                right_g = self._classify_static(right_landmarks)
                two     = self._match_two_hand(left_g, right_g)

                if two:
                    now = time.time()
                    if self._two_hand_gesture != two:
                        # New gesture started
                        self._two_hand_gesture = two
                        self._two_hand_start   = now
                        self._two_hand_fired   = False
                    else:
                        # Same gesture held — check hold duration
                        held = now - self._two_hand_start
                        if held >= self.TWO_HAND_HOLD and not self._two_hand_fired:
                            cooldown = self._last_two_hand.get(two, 0)
                            if now - cooldown >= 2.0:
                                self._two_hand_fired        = True
                                self._last_two_hand[two]    = now
                                self.two_hand_gesture.emit(two)
                else:
                    self._two_hand_gesture = None
                    self._two_hand_fired   = False

            now       = time.time()
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            frame = self._draw_overlay(
                frame, left_landmarks, right_landmarks,
                left_gesture, ai_mode_active, fps, w, h
            )
            self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False

    def _draw_landmarks(self, frame, hand_lms, w, h, color):
        connections = [
            (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17),
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(frame, pts[tip], 7, COL_YELLOW, -1)

    def _draw_overlay(self, frame, left_lm, right_lm, left_gesture, ai_mode, fps, w, h):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GREEN, 2)
        if ai_mode:
            cv2.rectangle(frame, (0, 0), (w, 4), COL_ORANGE, -1)
            cv2.putText(frame, "AI MODE ACTIVE", (w//2 - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ORANGE, 2)
        if left_lm:
            lx, ly = int(left_lm[0][0]*w), int(left_lm[0][1]*h)
            cv2.putText(frame, "LEFT (AI Lock)" if left_gesture=="FIST" else "LEFT",
                        (lx, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ORANGE, 1)
        if right_lm:
            rx, ry = int(right_lm[0][0]*w), int(right_lm[0][1]*h)
            cv2.putText(frame, "RIGHT (Control)", (rx, ry-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 1)
        # Two-hand gesture charge bar
        if self._two_hand_gesture and not self._two_hand_fired:
            held   = min(time.time() - self._two_hand_start, self.TWO_HAND_HOLD)
            charge = held / self.TWO_HAND_HOLD
            bar_w  = int(charge * 160)
            bx, by = w//2 - 80, h - 55
            colours = {
                "BOTH_PEACE": (0,   80, 255),
                "BOTH_L"    : (0,  200,  80),
                "BOTH_FIST" : (0,   60, 220),
                "BOTH_PALM" : (200, 160,  0),
            }
            bar_col = colours.get(self._two_hand_gesture, (200,200,200))
            labels  = {
                "BOTH_PEACE": "✌✌ LOCK",
                "BOTH_L"    : "L+L UNLOCK",
                "BOTH_FIST" : "✊✊ RESET",
                "BOTH_PALM" : "🖐🖐 OVERLAY",
            }
            bar_lbl = labels.get(self._two_hand_gesture, self._two_hand_gesture)
            cv2.rectangle(frame, (bx, by), (bx+160, by+10), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+bar_w, by+10), bar_col, -1)
            cv2.putText(frame, bar_lbl, (bx, by-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)

        # Pan active indicator
        if self._pan_active:
            cv2.putText(frame, "PAN MODE", (w-90, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_YELLOW, 2)

        cv2.rectangle(frame, (0, h-35), (w, h), (10,10,10), -1)
        if not left_lm and not right_lm:
            cv2.putText(frame, "Show hands to camera",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
        elif not ai_mode:
            cv2.putText(frame, "L.FIST=AI | ✌✌=Lock | LL=Unlock | ✊✊=Reset",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ORANGE, 1)
        else:
            cv2.putText(frame, "AI ON | ✌✌=Lock | LL=Unlock | POINT+move=Pan",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1)
        return frame

    def _match_two_hand(self, left_g, right_g) -> str | None:
        """
        Match both-hand gesture combinations.
        Returns gesture key or None.

        BOTH_PEACE : both hands PEACE ✌️✌️
        BOTH_L     : both hands L-shape (thumb+index up, others down) 👍☝️
        BOTH_FIST  : both hands FIST 👊👊
        BOTH_PALM  : both hands OPEN_PALM 🖐🖐
        """
        if left_g is None or right_g is None:
            return None
        combo = {left_g, right_g}
        if left_g == "PEACE"     and right_g == "PEACE"     : return "BOTH_PEACE"
        if left_g == "FIST"      and right_g == "FIST"      : return "BOTH_FIST"
        if left_g == "OPEN_PALM" and right_g == "OPEN_PALM" : return "BOTH_PALM"
        # L-shape: thumb + index extended, middle/ring/pinky down
        if left_g == "L_SHAPE"   and right_g == "L_SHAPE"   : return "BOTH_L"
        return None

    def _is_L_shape(self, lm) -> bool:
        """
        L-shape: thumb extended sideways + index pointing up,
        middle/ring/pinky all down.
        Orientation-independent using wrist-distance method.
        """
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        # Index must be extended
        idx_ext  = lm[tips[1]][1] < lm[pip[1]][1]
        # Middle, ring, pinky must be DOWN
        mid_down  = lm[tips[2]][1] > lm[pip[2]][1]
        ring_down = lm[tips[3]][1] > lm[pip[3]][1]
        pink_down = lm[tips[4]][1] > lm[pip[4]][1]
        # Thumb extended (use x-axis distance from palm)
        thu_ext  = abs(lm[tips[0]][0] - lm[0][0]) > 0.08
        return idx_ext and mid_down and ring_down and pink_down and thu_ext

    def _classify_static(self, lm):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)
        if num_up == 0: return "FIST"
        if num_up == 5: return "OPEN_PALM"
        if fingers_up == [0,1,0,0,0]: return "POINT_UP"
        if fingers_up == [0,1,1,0,0]: return "PEACE"
        if self._is_pinch(lm):        return "PINCH"
        if self._is_L_shape(lm):      return "L_SHAPE"
        return None

    def _classify_full(self, lm, hand_key):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)
        pos_list   = self.prev_pos_right if hand_key == "right" else self.prev_pos_left
        pos_list.append(lm[0])
        if len(pos_list) > self.smooth_window:
            pos_list.pop(0)

        # ── Check swipe FIRST using velocity over position history ──
        # This way POINT_UP during a swipe motion still fires SWIPE
        swipe = None
        if len(pos_list) >= self.smooth_window:
            dy = pos_list[-1][1] - pos_list[0][1]
            dx = pos_list[-1][0] - pos_list[0][0]
            # Only fire swipe if vertical motion dominates
            if abs(dy) > abs(dx) * 1.5:
                if dy < -0.05:   swipe = "SWIPE_UP"
                elif dy > 0.05:  swipe = "SWIPE_DOWN"

        # ── Static pose classification ──────────────────────────────
        gesture = None
        if swipe:
            gesture = swipe
        elif num_up == 5:                  gesture = "OPEN_PALM"
        elif num_up == 0:                  gesture = "FIST"
        elif fingers_up == [0,1,1,0,0]:   gesture = "PEACE"
        elif self._is_pinch(lm):          gesture = "PINCH"
        elif fingers_up == [0,1,0,0,0]:   gesture = "POINT_UP"  # last: conflicts with swipe

        # ── Per-gesture cooldown ────────────────────────────────────
        if gesture:
            key      = f"{hand_key}_{gesture}"
            now      = time.time()
            cooldown = GESTURE_COOLDOWN.get(gesture, 0.8)
            if now - self.last_time.get(key, 0) < cooldown:
                gesture = None
            else:
                self.last_time[key] = now
        return gesture

    def _fingers_up(self, lm):
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        f = [1 if lm[tips[0]][0] > lm[pip[0]][0] else 0]
        for i in range(1, 5):
            f.append(1 if lm[tips[i]][1] < lm[pip[i]][1] else 0)
        return f

    def _is_pinch(self, lm):
        tx, ty = lm[4]
        ix, iy = lm[8]
        return np.hypot(tx-ix, ty-iy) < 0.025

    @staticmethod
    def _get_model():
        import urllib.request
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmark model...")
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                model_path
            )
        return model_path


# ──────────────────────────────────────────────
#  Overlay Renderer
# ──────────────────────────────────────────────
def apply_segmentation_overlay(scan_slice, mask_slice, opacity=0.45):
    bgr     = cv2.cvtColor(scan_slice, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()
    for label, (name, colour) in ORGAN_COLOURS.items():
        mask = (mask_slice == label)
        if mask.any():
            overlay[mask] = colour
    return cv2.addWeighted(overlay, opacity, bgr, 1-opacity, 0)


# ──────────────────────────────────────────────
#  Legend Widget
# ──────────────────────────────────────────────
class LegendWidget(QWidget):
    def __init__(self, active_labels=None):
        super().__init__()
        self.active_labels = active_labels or []
        self._build()

    def _build(self):
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
            row = QHBoxLayout()
            swatch = QLabel("  ")
            swatch.setFixedSize(16, 12)
            swatch.setStyleSheet(f"background:rgb({r},{g},{b}); border-radius:2px;")
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("color:#ccc; font-size:10px;")
            row.addWidget(swatch)
            row.addWidget(name_lbl)
            row.addStretch()
            layout.addLayout(row)
        layout.addStretch()


# ──────────────────────────────────────────────
#  Main Window
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume       = None
        self.sitk_image   = None
        self.dicom_path   = None
        self.modality     = "CT"
        self.current      = 0
        self.ww           = 400.0
        self.wl           = 40.0
        self.zoom         = 1.0
        self.ai_mode      = False
        self.seg_mask     = None
        self.show_overlay = False
        self.opacity      = 0.45
        self.pan_x        = 0      # pan offset in pixels (image coords)
        self.pan_y        = 0
        self.view_locked  = False  # True = zoom/pan/slice locked
        self.preprocessor = PreprocessingEngine()
        self._setup_ui()
        self._start_gesture_engine()

    def _setup_ui(self):
        self.setWindowTitle("Gesture-Controlled Medical Imaging Workstation — Phase 4")
        self.setMinimumSize(1500, 800)
        self._apply_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── LEFT ───────────────────────────────
        left = QVBoxLayout()
        left.setSpacing(5)
        for text, slot in [("📂  Open DICOM File", self.open_file),
                            ("📁  Open DICOM Folder", self.open_folder)]:
            b = QPushButton(text)
            b.setFixedHeight(34)
            b.setStyleSheet(self._btn_style())
            b.clicked.connect(slot)
            left.addWidget(b)

        meta_group = QGroupBox("Patient Info")
        meta_group.setStyleSheet(self._group_style())
        mg = QGridLayout(meta_group)
        self.meta_labels = {}
        for i, key in enumerate(["Patient","Modality","Study Date","Rows","Columns","Institution"]):
            lk = QLabel(f"{key}:"); lk.setStyleSheet("color:#666; font-size:10px;")
            lv = QLabel("—");       lv.setStyleSheet("color:#ddd; font-size:10px;"); lv.setWordWrap(True)
            mg.addWidget(lk, i, 0); mg.addWidget(lv, i, 1)
            self.meta_labels[key] = lv
        left.addWidget(meta_group)

        win_group = QGroupBox("Windowing")
        win_group.setStyleSheet(self._group_style())
        wg = QVBoxLayout(win_group)
        self.lbl_wl = QLabel("WL: 40"); self.lbl_ww = QLabel("WW: 400")
        self.slider_wl = QSlider(Qt.Horizontal); self.slider_wl.setRange(-1000,3000); self.slider_wl.setValue(40); self.slider_wl.valueChanged.connect(self._on_wl)
        self.slider_ww = QSlider(Qt.Horizontal); self.slider_ww.setRange(1,4000);     self.slider_ww.setValue(400); self.slider_ww.valueChanged.connect(self._on_ww)
        for w in (self.lbl_wl, self.slider_wl, self.lbl_ww, self.slider_ww):
            w.setStyleSheet("color:#bbb; font-size:10px;" if isinstance(w,QLabel) else ""); wg.addWidget(w)
        presets = QHBoxLayout()
        for name,(l,w) in [("Brain",(40,80)),("Lung",(-600,1500)),("Bone",(400,1800))]:
            pb = QPushButton(name); pb.setFixedHeight(24); pb.setStyleSheet(self._btn_style(small=True))
            pb.clicked.connect(lambda _,lv=l,wv=w: self._preset(lv,wv)); presets.addWidget(pb)
        wg.addLayout(presets)
        left.addWidget(win_group)

        # ── AI Preprocessing Panel ────────────
        pre_group = QGroupBox("AI Preprocessing")
        pre_group.setStyleSheet(self._group_style())
        pg = QVBoxLayout(pre_group)

        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet("color:#bbb; font-size:10px;")
        pg.addWidget(filter_lbl)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(PreprocessingEngine.FILTERS)
        self.filter_combo.setStyleSheet(
            "QComboBox{background:#222;color:#ddd;border:1px solid #333;"
            "border-radius:3px;padding:2px;font-size:10px;}"
            "QComboBox::drop-down{border:none;}"
            "QComboBox QAbstractItemView{background:#222;color:#ddd;"
            "selection-background-color:#3af;}"
        )
        self.filter_combo.currentTextChanged.connect(self._on_filter_change)
        pg.addWidget(self.filter_combo)

        self.lbl_strength = QLabel("Strength: 1.0")
        self.lbl_strength.setStyleSheet("color:#bbb; font-size:10px;")
        pg.addWidget(self.lbl_strength)

        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(1, 20)
        self.slider_strength.setValue(10)
        self.slider_strength.valueChanged.connect(self._on_strength_change)
        pg.addWidget(self.slider_strength)

        btn_reset_filter = QPushButton("Reset Filter")
        btn_reset_filter.setFixedHeight(24)
        btn_reset_filter.setStyleSheet(self._btn_style(small=True))
        btn_reset_filter.clicked.connect(self._reset_filter)
        pg.addWidget(btn_reset_filter)

        left.addWidget(pre_group)


        ai_group = QGroupBox("AI Segmentation  (TotalSegmentator)")
        ai_group.setStyleSheet(self._group_style())
        ag = QVBoxLayout(ai_group)

        self.lbl_modality_hint = QLabel("Load a DICOM to detect modality")
        self.lbl_modality_hint.setStyleSheet("color:#555; font-size:9px; font-style:italic;")
        ag.addWidget(self.lbl_modality_hint)

        self.btn_segment = QPushButton("🤖  Run Segmentation")
        self.btn_segment.setFixedHeight(34)
        self.btn_segment.setStyleSheet(self._btn_style(highlight=True))
        self.btn_segment.clicked.connect(self._run_segmentation)
        ag.addWidget(self.btn_segment)

        self.btn_toggle_overlay = QPushButton("👁  Toggle Overlay")
        self.btn_toggle_overlay.setFixedHeight(28)
        self.btn_toggle_overlay.setStyleSheet(self._btn_style())
        self.btn_toggle_overlay.clicked.connect(self._toggle_overlay)
        self.btn_toggle_overlay.setEnabled(False)
        ag.addWidget(self.btn_toggle_overlay)

        lbl_op = QLabel("Overlay Opacity:"); lbl_op.setStyleSheet("color:#bbb; font-size:10px;"); ag.addWidget(lbl_op)
        self.slider_opacity = QSlider(Qt.Horizontal); self.slider_opacity.setRange(10,90); self.slider_opacity.setValue(45); self.slider_opacity.valueChanged.connect(self._on_opacity); ag.addWidget(self.slider_opacity)

        self.progress_bar = QProgressBar(); self.progress_bar.setRange(0,0); self.progress_bar.setVisible(False); self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet("QProgressBar{border:none;background:#222;border-radius:3px;}QProgressBar::chunk{background:#3af;border-radius:3px;}")
        ag.addWidget(self.progress_bar)

        self.lbl_seg_status = QLabel("No segmentation loaded"); self.lbl_seg_status.setStyleSheet("color:#666; font-size:10px;"); self.lbl_seg_status.setWordWrap(True)
        ag.addWidget(self.lbl_seg_status)
        left.addWidget(ai_group)

        gest_group = QGroupBox("Gesture Control"); gest_group.setStyleSheet(self._group_style())
        gg = QVBoxLayout(gest_group)
        self.lbl_gesture = QLabel("Waiting..."); self.lbl_gesture.setStyleSheet("color:#3af; font-size:12px; font-weight:bold; qproperty-alignment:AlignCenter;"); gg.addWidget(self.lbl_gesture)
        self.lbl_ai_indicator = QLabel("AI Mode: OFF"); self.lbl_ai_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;"); gg.addWidget(self.lbl_ai_indicator)

        self.lbl_lock_indicator = QLabel("🔓 View: UNLOCKED")
        self.lbl_lock_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;")
        gg.addWidget(self.lbl_lock_indicator)

        for line in [
            "LEFT  FIST hold 1s  → AI Mode",
            "RIGHT Swipe Up/Down → Prev/Next Slice",
            "RIGHT Pinch         → Zoom In",
            "RIGHT Fist          → Zoom Out",
            "RIGHT POINT_UP+move → Pan",
            "── Two-Hand Gestures ─────────────",
            "✌️✌️  Both PEACE     → Lock View",
            "👍☝️  Both L-shape   → Unlock View",
            "👊👊  Both FIST     → Reset View",
            "🖐🖐  Both PALM     → Toggle Overlay",
            "── Hold 0.6s to trigger ──────────",
        ]:
            l = QLabel(line); l.setStyleSheet("color:#444; font-size:9px;"); gg.addWidget(l)
        left.addWidget(gest_group)

        self.lbl_slice = QLabel("Slice: — / —"); self.lbl_slice.setStyleSheet("color:#aaa; font-size:11px; qproperty-alignment:AlignCenter;"); left.addWidget(self.lbl_slice)
        left.addStretch()
        btn_reset = QPushButton("↺  Reset View"); btn_reset.setFixedHeight(30); btn_reset.setStyleSheet(self._btn_style()); btn_reset.clicked.connect(self._reset_view); left.addWidget(btn_reset)
        root.addLayout(left, 1)

        # ── CENTRE ─────────────────────────────
        centre = QVBoxLayout()
        self.dicom_label = QLabel(); self.dicom_label.setAlignment(Qt.AlignCenter); self.dicom_label.setMinimumSize(620,540)
        self.dicom_label.setText("Open a DICOM file or folder"); self.dicom_label.setFont(QFont("Courier New",12))
        self.dicom_label.setStyleSheet("background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;")
        self.slice_slider = QSlider(Qt.Vertical); self.slice_slider.setRange(0,0); self.slice_slider.valueChanged.connect(self._on_slice_change)
        img_row = QHBoxLayout(); img_row.addWidget(self.dicom_label); img_row.addWidget(self.slice_slider); centre.addLayout(img_row)
        nav = QHBoxLayout()
        for text, slot in [("◀  Prev", self._prev_slice),("Next  ▶", self._next_slice)]:
            b = QPushButton(text); b.setFixedHeight(30); b.setStyleSheet(self._btn_style()); b.clicked.connect(slot); nav.addWidget(b)
        centre.addLayout(nav)
        root.addLayout(centre, 3)

        # ── RIGHT ──────────────────────────────
        right = QVBoxLayout(); right.setSpacing(6)
        cam_title = QLabel("✋  Gesture Camera (Two-Hand Mode)"); cam_title.setStyleSheet("color:#3af; font-size:12px; font-weight:bold;"); cam_title.setAlignment(Qt.AlignCenter); right.addWidget(cam_title)
        self.cam_label = QLabel(); self.cam_label.setAlignment(Qt.AlignCenter); self.cam_label.setFixedSize(360,290)
        self.cam_label.setStyleSheet("background:#0a0a0a; border:1px solid #2a2a2a; border-radius:6px;"); right.addWidget(self.cam_label)
        self.lbl_cam_status = QLabel("● Camera: connecting..."); self.lbl_cam_status.setStyleSheet("color:#888; font-size:10px;"); right.addWidget(self.lbl_cam_status)
        legend_title = QLabel("Organ Legend"); legend_title.setStyleSheet("color:#3af; font-size:11px; font-weight:bold;"); right.addWidget(legend_title)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setStyleSheet("border:none; background:#161616;")
        self.legend_widget = LegendWidget(); scroll.setWidget(self.legend_widget); right.addWidget(scroll)
        root.addLayout(right, 1)

        self.status = QStatusBar(); self.setStatusBar(self.status); self.status.setStyleSheet("color:#666; background:#111;")
        self.status.showMessage("Ready — open a DICOM file to begin.")

    def _start_gesture_engine(self):
        self.gesture_engine = GestureEngine()
        self.gesture_engine.frame_ready.connect(self._on_cam_frame)
        self.gesture_engine.slice_navigate.connect(self._on_slice_navigate)
        self.gesture_engine.zoom_changed.connect(self._on_zoom_gesture)
        self.gesture_engine.pan_moved.connect(self._on_pan_gesture)
        self.gesture_engine.window_changed.connect(self._on_window_gesture)
        self.gesture_engine.lock_toggled.connect(self._toggle_lock)
        self.gesture_engine.reset_triggered.connect(self._reset_view)
        self.gesture_engine.segmentation_triggered.connect(self._run_segmentation)
        self.gesture_engine.overlay_intensity.connect(self._on_overlay_intensity)
        self.gesture_engine.recalibrate.connect(self._on_recalibrate)
        self.gesture_engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")

    # ── New GestureEngine signal handlers ──────────────────────
    def _on_slice_navigate(self, direction):
        """direction: -1 = prev, +1 = next"""
        if self.view_locked:
            return
        if direction < 0:
            self._prev_slice()
        else:
            self._next_slice()
        self.lbl_gesture.setText(f"Swipe\n{'Prev' if direction < 0 else 'Next'} Slice")

    def _on_zoom_gesture(self, delta):
        """delta: positive = zoom in, negative = zoom out. Blocked when locked."""
        if self.view_locked:
            self.status.showMessage("View LOCKED — zoom disabled")
            return
        self.zoom = max(1.0, min(4.0, self.zoom + delta))
        self._show_slice()
        self.lbl_gesture.setText(f"Zoom {chr(43) if delta > 0 else chr(45)}{abs(delta):.2f}\n{self.zoom:.1f}x")

    def _on_window_gesture(self, dwl, dww):
        """
        Adjust window level (WL) and/or window width (WW).
        dwl: delta WL (brightness centre)
        dww: delta WW (contrast range) — currently 0, reserved
        """
        if self.view_locked:
            return
        if dwl != 0:
            self.wl = max(-1000, min(3000, self.wl + dwl))
            self.slider_wl.blockSignals(True)
            self.slider_wl.setValue(int(self.wl))
            self.slider_wl.blockSignals(False)
            self.lbl_wl.setText(f"WL: {int(self.wl)}")
        if dww != 0:
            self.ww = max(1.0, min(4000, self.ww + dww))
            self.slider_ww.blockSignals(True)
            self.slider_ww.setValue(int(self.ww))
            self.slider_ww.blockSignals(False)
            self.lbl_ww.setText(f"WW: {int(self.ww)}")
        self._show_slice()
        self.lbl_gesture.setText(f"Windowing\nWL:{self.wl:.0f} WW:{self.ww:.0f}")

    def _on_overlay_intensity(self, direction):
        """direction: +1 brighter, -1 dimmer"""
        self.opacity = max(0.1, min(0.9, self.opacity + direction * 0.05))
        self.slider_opacity.setValue(int(self.opacity * 100))
        self._show_slice()
        self.lbl_gesture.setText(f"Overlay\n{self.opacity*100:.0f}%")

    def _on_recalibrate(self):
        """Reset pan and zoom but keep slice."""
        self.zoom  = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._show_slice()
        self.lbl_gesture.setText("A-shape\nRecalibrated")
        self.status.showMessage("Recalibrated — zoom and pan reset")

    def _on_gesture(self, gesture, hand):
        action = GESTURE_ACTIONS.get(gesture, gesture)
        self.lbl_gesture.setText(f"{gesture}\n{action}")
        self.status.showMessage(f"{hand.upper()} hand: {gesture}  →  {action}")

        # When view is locked, block all navigation except unlock
        if self.view_locked:
            self.status.showMessage("View LOCKED — Both L-shape to unlock")
            return

        if   gesture == "SWIPE_UP"  : self._prev_slice()
        elif gesture == "SWIPE_DOWN": self._next_slice()
        elif gesture == "OPEN_PALM" : self._reset_view()
        elif gesture == "PINCH"     : self._zoom_in()
        elif gesture == "FIST"      : self._zoom_out()
        # POINT_UP handled by pan engine — no fallback needed
        # PEACE single tap — no action (reserved for two-hand lock)

    # ── Two-Hand Gesture Handler ───────────────────────────────
    def _on_two_hand_gesture(self, gesture):
        """Handle two-hand gestures fired from gesture engine."""
        if gesture == "BOTH_PEACE":
            # Lock view (only if not already locked)
            if not self.view_locked:
                self._toggle_lock()
                self.lbl_gesture.setText("BOTH PEACE\n🔒 View Locked")
        elif gesture == "BOTH_L":
            # Unlock view (only if locked)
            if self.view_locked:
                self._toggle_lock()
                self.lbl_gesture.setText("BOTH L-SHAPE\n🔓 View Unlocked")
        elif gesture == "BOTH_FIST":
            self._reset_view()
            self.lbl_gesture.setText("BOTH FIST\n↺ View Reset")
        elif gesture == "BOTH_PALM":
            self._toggle_overlay()
            self.lbl_gesture.setText("BOTH PALM\n👁 Overlay Toggled")

    # ── View Lock ──────────────────────────────────────────────
    def _toggle_lock(self):
        self.view_locked = not self.view_locked
        if self.view_locked:
            self.lbl_lock_indicator.setText("🔒 View: LOCKED")
            self.lbl_lock_indicator.setStyleSheet(
                "color:#f33; font-size:11px; font-weight:bold; qproperty-alignment:AlignCenter;")
            self.dicom_label.setStyleSheet(
                "background:#0d0d0d; color:#333; border:3px solid #f33; border-radius:6px;")
            self.status.showMessage("🔒 View LOCKED — zoom, pan and slice frozen")
        else:
            self.lbl_lock_indicator.setText("🔓 View: UNLOCKED")
            self.lbl_lock_indicator.setStyleSheet(
                "color:#3f3; font-size:11px; qproperty-alignment:AlignCenter;")
            # Restore border based on AI mode
            if self.ai_mode:
                self.dicom_label.setStyleSheet(
                    "background:#0d0d0d; color:#333; border:2px solid #f90; border-radius:6px;")
            else:
                self.dicom_label.setStyleSheet(
                    "background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;")
            self.status.showMessage("🔓 View UNLOCKED")

    # ── Pan ────────────────────────────────────────────────────
    def _pan(self, dx, dy):
        """Move pan offset. dx/dy in image pixels."""
        if self.view_locked or self.volume is None:
            return
        h, w = self.volume[self.current].shape
        max_pan_x = int(w * (1 - 1/max(self.zoom, 1.0)))
        max_pan_y = int(h * (1 - 1/max(self.zoom, 1.0)))
        self.pan_x = max(-max_pan_x, min(max_pan_x, self.pan_x + dx))
        self.pan_y = max(-max_pan_y, min(max_pan_y, self.pan_y + dy))
        self._show_slice()

    def _on_pan_gesture(self, dx_norm, dy_norm):
        """
        Called from gesture engine with normalised deltas.
        Scale to image pixel space for actual pan movement.
        """
        if self.volume is None or self.view_locked:
            return
        h, w = self.volume[self.current].shape
        # Scale: normalised delta * image dimension * sensitivity
        # Negative sign on dy because image y is inverted vs screen y
        px = int(dx_norm * w * 3.0)
        py = int(dy_norm * h * 3.0)
        if abs(px) > 0 or abs(py) > 0:
            self._pan(px, py)
            self.lbl_gesture.setText(f"Pan\n{px:+d},{py:+d}px")

    def _on_ai_mode_changed(self, active):
        self.ai_mode = active
        if active:
            self.lbl_ai_indicator.setText("🤖 AI Mode: ACTIVE")
            self.lbl_ai_indicator.setStyleSheet("color:#f90; font-size:11px; font-weight:bold; qproperty-alignment:AlignCenter;")
            self.dicom_label.setStyleSheet("background:#0d0d0d; color:#333; border:2px solid #f90; border-radius:6px;")
            if self.seg_mask is not None:
                self.show_overlay = True; self._show_slice()
            self.status.showMessage("AI Mode ACTIVE — overlay enabled")
        else:
            self.lbl_ai_indicator.setText("AI Mode: OFF")
            self.lbl_ai_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;")
            self.dicom_label.setStyleSheet("background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;")
            self.show_overlay = False; self._show_slice()

    def _on_cam_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w*3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(self.cam_label.width(), self.cam_label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.cam_label.setPixmap(pix)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM (*.dcm);;All (*)")
        if path: self.dicom_path = path; self._load(path)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if path: self.dicom_path = path; self._load(path)

    def _load(self, path):
        self.status.showMessage("Loading DICOM...")
        self.seg_mask = None; self.show_overlay = False
        self.loader = DicomLoader(path)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.error.connect(lambda e: self.status.showMessage(f"Error: {e}"))
        self.loader.start()

    def _on_loaded(self, volume, sitk_image, meta):
        self.volume = volume; self.sitk_image = sitk_image
        self.modality = meta.get("Modality","CT").upper()
        self.current  = volume.shape[0]//2; self.zoom = 1.0
        self.slice_slider.setRange(0, volume.shape[0]-1); self.slice_slider.setValue(self.current)
        self._update_meta(meta); self._reset_windowing(); self._show_slice()
        task = _pick_task(self.modality)
        self.lbl_modality_hint.setText(f"Detected: {self.modality}  →  task='{task}'")
        self.lbl_modality_hint.setStyleSheet("color:#3af; font-size:9px;")
        self.status.showMessage(f"Loaded {volume.shape[0]} slices | {self.modality} | Task: {task}")

    def _run_segmentation(self):
        if self.dicom_path is None:
            self.status.showMessage("Please open a DICOM file or folder first."); return
        task = _pick_task(self.modality)
        self.btn_segment.setEnabled(False); self.progress_bar.setVisible(True)
        self.lbl_seg_status.setText(f"Starting TotalSegmentator...\nTask: {task} | CPU fast mode\n~20-30 seconds")
        self.seg_worker = SegmentationWorker(self.dicom_path, self.modality)
        self.seg_worker.progress.connect(self._on_seg_progress)
        self.seg_worker.finished.connect(self._on_seg_done)
        self.seg_worker.error.connect(self._on_seg_error)
        self.seg_worker.start()

    def _on_seg_progress(self, msg):
        self.lbl_seg_status.setText(msg); self.status.showMessage(msg)

    def _on_seg_done(self, mask):
        self.seg_mask = mask; self.progress_bar.setVisible(False)
        self.btn_segment.setEnabled(True); self.btn_toggle_overlay.setEnabled(True); self.show_overlay = True
        active = [l for l in ORGAN_COLOURS if np.any(mask==l)]
        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            self.legend_widget = LegendWidget(active_labels=active); scroll_area.setWidget(self.legend_widget)
        self.lbl_seg_status.setText(f"✅ Done! {len(active)} structures found.")
        self.status.showMessage(f"Segmentation complete — {len(active)} organs detected")
        self._show_slice()

    def _on_seg_error(self, msg):
        self.progress_bar.setVisible(False); self.btn_segment.setEnabled(True)
        self.lbl_seg_status.setText(f"❌ {msg}"); self.status.showMessage("Segmentation error — see panel")


    # ── Preprocessing Handlers ─────────────────────────────────
    def _on_filter_change(self, filter_name):
        self.preprocessor.active_filter = filter_name
        self._show_slice()
        self.status.showMessage("Filter: " + filter_name)
        short = filter_name.split("(")[0].strip()
        self.lbl_gesture.setText("Filter\n" + short)

    def _on_strength_change(self, val):
        self.preprocessor.strength = val / 10.0
        self.lbl_strength.setText("Strength: " + str(round(self.preprocessor.strength, 1)))
        self._show_slice()

    def _reset_filter(self):
        self.filter_combo.setCurrentIndex(0)
        self.slider_strength.setValue(10)
        self.status.showMessage("Filter reset to original")

    def _toggle_overlay(self):
        if self.seg_mask is not None:
            self.show_overlay = not self.show_overlay; self._show_slice()

    def _on_opacity(self, val):
        self.opacity = val/100.0; self._show_slice()

    def _show_slice(self):
        if self.volume is None:
            return

        # 1. Window
        raw = self.volume[self.current]
        img = self._apply_window(raw)

        # 2. Preprocessing filter
        img = self.preprocessor.apply(img)

        # 3. Zoom + pan crop (image)
        if self.zoom != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            h, w  = img.shape
            new_h = int(h / self.zoom)
            new_w = int(w / self.zoom)
            cy = h//2 + self.pan_y
            cx = w//2 + self.pan_x
            y1 = max(0, cy - new_h//2); y2 = min(h, y1 + new_h)
            x1 = max(0, cx - new_w//2); x2 = min(w, x1 + new_w)
            if y2 > h: y1 = max(0, h - new_h); y2 = h
            if x2 > w: x1 = max(0, w - new_w); x2 = w
            img = cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Overlay
        if self.show_overlay and self.seg_mask is not None:
            mz = self.seg_mask.shape[0]
            vz = self.volume.shape[0]
            mask_idx   = min(int(self.current * mz / vz), mz - 1)
            mask_slice = self.seg_mask[mask_idx]

            # Resize mask to raw slice dimensions
            raw_h, raw_w = self.volume[self.current].shape
            if mask_slice.shape != (raw_h, raw_w):
                mask_slice = cv2.resize(
                    mask_slice.astype(np.float32),
                    (raw_w, raw_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            # Apply same zoom + pan to mask
            if self.zoom != 1.0 or self.pan_x != 0 or self.pan_y != 0:
                h0, w0 = mask_slice.shape
                new_h  = int(h0 / self.zoom)
                new_w  = int(w0 / self.zoom)
                cy = h0//2 + self.pan_y
                cx = w0//2 + self.pan_x
                y1 = max(0, cy - new_h//2); y2 = min(h0, y1 + new_h)
                x1 = max(0, cx - new_w//2); x2 = min(w0, x1 + new_w)
                if y2 > h0: y1 = max(0, h0 - new_h); y2 = h0
                if x2 > w0: x1 = max(0, w0 - new_w); x2 = w0
                mask_slice = cv2.resize(
                    mask_slice[y1:y2, x1:x2].astype(np.float32),
                    (w0, h0),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            # Final size match
            if mask_slice.shape != img.shape:
                mask_slice = cv2.resize(
                    mask_slice.astype(np.float32),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            display = apply_segmentation_overlay(img, mask_slice, self.opacity)
            h, w    = display.shape[:2]
            qimg    = QImage(display.tobytes(), w, h, w * 3, QImage.Format_BGR888)
        else:
            h, w = img.shape
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)

        pix = QPixmap.fromImage(qimg).scaled(
            self.dicom_label.width() - 4,
            self.dicom_label.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.dicom_label.setPixmap(pix)
        self.lbl_slice.setText(f"Slice:  {self.current+1} / {self.volume.shape[0]}")

    def _apply_window(self, raw):
        lo=self.wl-self.ww/2; hi=self.wl+self.ww/2
        return ((np.clip(raw,lo,hi)-lo)/(hi-lo)*255).astype(np.uint8)

    def _on_slice_change(self, val): self.current=val; self._show_slice()
    def _prev_slice(self):
        if self.volume is not None and self.current>0: self.current-=1; self.slice_slider.setValue(self.current)
    def _next_slice(self):
        if self.volume is not None and self.current<self.volume.shape[0]-1: self.current+=1; self.slice_slider.setValue(self.current)
    def wheelEvent(self, event):
        if event.angleDelta().y()>0: self._prev_slice()
        else: self._next_slice()
    def _zoom_in(self):  self.zoom=min(self.zoom+0.1,3.0); self._show_slice()
    def _zoom_out(self): self.zoom=max(self.zoom-0.1,1.0); self._show_slice()
    def _on_wl(self, val): self.wl=float(val); self.lbl_wl.setText(f"WL: {val}"); self._show_slice()
    def _on_ww(self, val): self.ww=float(val); self.lbl_ww.setText(f"WW: {val}"); self._show_slice()
    def _preset(self, wl, ww): self.slider_wl.setValue(wl); self.slider_ww.setValue(ww)
    def _reset_windowing(self):
        if self.volume is not None:
            self.wl=float(np.median(self.volume)); self.ww=float(self.volume.max()-self.volume.min())
            self.slider_wl.setValue(int(self.wl)); self.slider_ww.setValue(int(self.ww))
    def _reset_view(self):
        if self.volume is not None:
            self.current  = self.volume.shape[0]//2
            self.zoom     = 1.0
            self.pan_x    = 0
            self.pan_y    = 0
            self.view_locked = False
            self.lbl_lock_indicator.setText("🔓 View: UNLOCKED")
            self.lbl_lock_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;")
            self.slice_slider.setValue(self.current)
            self._reset_windowing()
    def _update_meta(self, meta):
        for k,v in meta.items():
            if k in self.meta_labels: self.meta_labels[k].setText(v)
    def resizeEvent(self, event): super().resizeEvent(event); self._show_slice()
    def closeEvent(self, event):
        if self.gesture_engine: self.gesture_engine.stop(); self.gesture_engine.wait()
        event.accept()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow,QWidget{background:#1a1a1a;color:#ddd;font-family:'Segoe UI',sans-serif;font-size:11px;}
            QSlider::groove:horizontal{height:4px;background:#2a2a2a;border-radius:2px;}
            QSlider::handle:horizontal{background:#3af;width:12px;height:12px;margin:-4px 0;border-radius:6px;}
            QSlider::sub-page:horizontal{background:#3af;border-radius:2px;}
            QSlider::groove:vertical{width:4px;background:#2a2a2a;border-radius:2px;}
            QSlider::handle:vertical{background:#3af;width:12px;height:12px;margin:0 -4px;border-radius:6px;}
            QSlider::sub-page:vertical{background:#3af;border-radius:2px;}
            QStatusBar{background:#111;color:#555;}
            QScrollArea{border:none;}
        """)

    def _btn_style(self, small=False, highlight=False):
        p="3px 6px" if small else "5px 10px"
        bg="#1a3a1a" if highlight else "#222"
        bc="#3a3" if highlight else "#333"
        hb="#2a4a2a" if highlight else "#2a2a2a"
        return f"QPushButton{{background:{bg};color:#ccc;border:1px solid {bc};border-radius:4px;padding:{p};}}QPushButton:hover{{background:{hb};color:#fff;border-color:#3af;}}QPushButton:pressed{{background:#111;}}QPushButton:disabled{{background:#181818;color:#444;border-color:#222;}}"

    def _group_style(self):
        return "QGroupBox{border:1px solid #252525;border-radius:6px;margin-top:8px;padding:5px;color:#555;font-size:10px;}QGroupBox::title{subcontrol-origin:margin;left:8px;color:#3af;}"


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())   