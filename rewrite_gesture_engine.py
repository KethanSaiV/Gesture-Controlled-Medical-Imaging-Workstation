"""
Complete Gesture Engine Rewrite
Based on the 12-gesture medical imaging chart

GESTURE MAP:
  01. Palm rotate CW/CCW          → Zoom in / out
  02. Index point → hold 2s → move → Pan image
  03. Two index fingers apart/together → Windowing W/L
  04. L-shape (ONE hand) hold 2s  → Lock / unlock
  05. Three fingers                → Reset image
  06. Both thumbs up hold 3s       → Run segmentation
  07. Thumb + other hand two fingers → Overlay intensity
  08. Join two index fingers (A)   → Recalibrate
  09. O shape (curl fingers)       → Preprocess/filter (future)
  10. Both hands flat together     → Register modalities (future)
  11. Swipe up/down                → Navigate slices
  12. Fist + other hand swipe      → Change modality (future)
"""

GESTURE_ENGINE_CODE = '''
import cv2
import numpy as np
import time
import math
from collections import deque

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
    PAN_DEADZONE        = 0.008  # min movement to pan
    PAN_SCALE           = 0.20   # pan speed
    WINDOW_SCALE        = 0.015  # windowing sensitivity
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

            # ── GESTURE 01: Palm rotate → zoom ─────────────────
            if ctrl_lm and not self._pan_locked:
                self._detect_rotation_zoom(ctrl_lm, now)
            else:
                self._prev_wrist_angle = None
                self._rotation_accum   = 0.0

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
        """Gesture 01: Palm rotation → zoom in/out."""
        # Compute angle of wrist→middle_mcp vector
        wrist  = np.array(lm[0])
        mid    = np.array(lm[9])   # middle finger MCP
        vec    = mid - wrist
        angle  = math.degrees(math.atan2(vec[1], vec[0]))

        if self._prev_wrist_angle is None:
            self._prev_wrist_angle = angle
            return

        delta = angle - self._prev_wrist_angle
        # Unwrap angle jump (avoid -180/180 flip)
        if delta > 90:  delta -= 180
        if delta < -90: delta += 180

        self._prev_wrist_angle = angle
        self._rotation_accum  += delta

        if self._rotation_accum > self.ZOOM_SENSITIVITY:
            self._rotation_accum = 0.0
            self.zoom_changed.emit(0.08)
            self._status("Rotate CW → Zoom In")
        elif self._rotation_accum < -self.ZOOM_SENSITIVITY:
            self._rotation_accum = 0.0
            self.zoom_changed.emit(-0.08)
            self._status("Rotate CCW → Zoom Out")

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
        """Gesture 03: Two index fingers apart/together → windowing."""
        l_fingers = self._fingers_up(left_lm)
        r_fingers = self._fingers_up(right_lm)
        # Both pointing only index
        l_point = (l_fingers[1] == 1 and l_fingers[2] == 0 and
                   l_fingers[3] == 0 and l_fingers[4] == 0)
        r_point = (r_fingers[1] == 1 and r_fingers[2] == 0 and
                   r_fingers[3] == 0 and r_fingers[4] == 0)
        if not (l_point and r_point):
            self._win_prev_dist = None
            return

        # Distance between two index tips
        lx, ly = left_lm[8]
        rx, ry = right_lm[8]
        dist   = math.hypot(rx - lx, ry - ly)

        if self._win_prev_dist is None:
            self._win_prev_dist = dist
            return

        delta = dist - self._win_prev_dist
        self._win_prev_dist = dist

        if abs(delta) > 0.005:
            # Apart → increase WW (wider window), together → decrease WW
            dww = delta * 800
            self.window_changed.emit(0, dww)
            self._status(f"Windowing WW {'wider' if delta > 0 else 'narrower'}")

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
        """Thumb extended upward, all other fingers curled."""
        f = self._fingers_up(lm)
        # Thumb up: thumb extended, others down
        # Also check thumb tip is ABOVE wrist (pointing up)
        thumb_up = lm[4][1] < lm[0][1]
        return f[0] == 1 and f[1] == 0 and f[2] == 0 and f[3] == 0 and f[4] == 0 and thumb_up

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
'''

# ─────────────────────────────────────────────────────────────
# Now patch phase4_segmentation.py:
# 1. Replace TwoHandGestureEngine class with GestureEngine
# 2. Wire new signals to MainWindow
# ─────────────────────────────────────────────────────────────

with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Insert the new engine class after imports, before old engine
INSERT_AFTER = "def _pick_task(modality: str) -> str:\n    return \"total_mr\" if modality.upper() in (\"MR\", \"MRI\") else \"total\"\n"

if INSERT_AFTER not in src:
    print("❌ Insertion point not found — check _pick_task function")
else:
    src = src.replace(INSERT_AFTER, INSERT_AFTER + "\n" + GESTURE_ENGINE_CODE + "\n")
    print("✅ GestureEngine class inserted")

# ── Replace _start_gesture_engine to use new engine ──────────
old_start = '''    def _start_gesture_engine(self):
        self.gesture_engine = TwoHandGestureEngine()
        self.gesture_engine.gesture_detected.connect(self._on_gesture)
        self.gesture_engine.frame_ready.connect(self._on_cam_frame)
        self.gesture_engine.ai_mode_changed.connect(self._on_ai_mode_changed)
        self.gesture_engine.two_hand_gesture.connect(self._on_two_hand_gesture)
        self.gesture_engine.pan_moved.connect(self._on_pan_gesture)
        self.gesture_engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")'''

new_start = '''    def _start_gesture_engine(self):
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
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")'''

if old_start in src:
    src = src.replace(old_start, new_start)
    print("✅ _start_gesture_engine rewired to GestureEngine")
else:
    print("⚠️  _start_gesture_engine not found — check manually")

# ── Add new signal handlers to MainWindow ─────────────────────
old_handlers = '''    def _on_gesture(self, gesture, hand):'''

new_handlers = '''    # ── New GestureEngine signal handlers ──────────────────────
    def _on_slice_navigate(self, direction):
        """direction: -1 = prev, +1 = next"""
        if self.view_locked:
            return
        if direction < 0:
            self._prev_slice()
        else:
            self._next_slice()
        self.lbl_gesture.setText(f"Swipe\\n{'Prev' if direction < 0 else 'Next'} Slice")

    def _on_zoom_gesture(self, delta):
        """delta: positive = zoom in, negative = zoom out"""
        if self.view_locked:
            return
        self.zoom = max(1.0, min(4.0, self.zoom + delta))
        self._show_slice()
        self.lbl_gesture.setText(f"Rotate\\nZoom {'In' if delta > 0 else 'Out'} {self.zoom:.1f}x")

    def _on_window_gesture(self, dwl, dww):
        """Adjust window level and width."""
        if self.view_locked:
            return
        self.wl += dwl
        self.ww  = max(1.0, self.ww + dww)
        self.slider_wl.setValue(int(self.wl))
        self.slider_ww.setValue(int(self.ww))
        self.lbl_gesture.setText(f"2-Index\\nWW:{self.ww:.0f}")

    def _on_overlay_intensity(self, direction):
        """direction: +1 brighter, -1 dimmer"""
        self.opacity = max(0.1, min(0.9, self.opacity + direction * 0.05))
        self.slider_opacity.setValue(int(self.opacity * 100))
        self._show_slice()
        self.lbl_gesture.setText(f"Overlay\\n{self.opacity*100:.0f}%")

    def _on_recalibrate(self):
        """Reset pan and zoom but keep slice."""
        self.zoom  = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._show_slice()
        self.lbl_gesture.setText("A-shape\\nRecalibrated")
        self.status.showMessage("Recalibrated — zoom and pan reset")

    def _on_gesture(self, gesture, hand):'''

if old_handlers in src:
    src = src.replace(old_handlers, new_handlers)
    print("✅ New signal handlers added to MainWindow")
else:
    print("⚠️  _on_gesture not found — handlers not added")

# ── Write final file ──────────────────────────────────────────
with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print("\n✅ Syntax OK — run: python phase4_segmentation.py")
except SyntaxError as e:
    print(f"\n❌ Syntax error at line {e.lineno}: {e.msg}")
    print(f"   Text: {repr(e.text)}")