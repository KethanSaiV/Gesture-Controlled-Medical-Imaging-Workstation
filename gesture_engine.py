"""
gesture_engine.py
Gesture-Controlled Medical Imaging Workstation

GestureEngine — replaces TwoHandGestureEngine.

Improvements:
  - CLAHE on V channel before MediaPipe (dim light fix)
  - Velocity-based swipe (not position delta)
  - Temporal voting buffer (6 frames, 70% consensus)
  - Landmark visibility guard (None-safe)
  - Pinch zoom (not rotation — no conflict with swipe)
  - Visual charge bars for hold gestures
  - Per-gesture cooldowns
  - Emits gesture tokens: SWIPE_UP, SWIPE_DOWN, PINCH,
    OPEN_PALM, FIST, POINT, PEACE

Two-hand gestures (hold 0.6s):
  Both PEACE    → lock_toggled
  Both L-shape  → unlock_toggled
  Both FIST     → reset_triggered
  Both PALM     → overlay_toggled
"""

from __future__ import annotations
import cv2
import numpy as np
import time
import math
from collections import deque, Counter

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PyQt5.QtCore import QThread, pyqtSignal


COL_GREEN  = (0, 255, 120)
COL_YELLOW = (0, 220, 255)
COL_ORANGE = (0, 165, 255)
COL_CYAN   = (255, 220,   0)
COL_WHITE  = (255, 255, 255)
COL_RED    = (0,   60, 220)


class GestureEngine(QThread):
    # ── Discrete token signals ─────────────────
    gesture_token    = pyqtSignal(str)         # single token for dispatcher
    pan_moved        = pyqtSignal(float, float) # dx_norm, dy_norm (continuous)
    window_changed   = pyqtSignal(float, float) # dwl, dww

    # ── Special two-hand signals ───────────────
    lock_toggled     = pyqtSignal()
    reset_triggered  = pyqtSignal()
    overlay_toggled  = pyqtSignal()
    seg_triggered    = pyqtSignal()            # both thumbs 3s

    # ── Camera feed ────────────────────────────
    frame_ready      = pyqtSignal(np.ndarray)
    cycle_view_mode  = pyqtSignal()            # both palms 1s → cycle CT/MRI/Overlay

    # ── Tuning ─────────────────────────────────
    VOTE_WINDOW      = 6      # frames
    VOTE_THRESHOLD   = 0.70   # fraction that must agree
    SWIPE_VEL_MIN    = 0.18   # normalised units/sec — must be fast
    SWIPE_COOLDOWN   = 0.35   # seconds between slices
    TWO_HAND_HOLD    = 0.6    # seconds to fire two-hand gesture
    SEG_HOLD         = 3.0    # seconds for both-thumbs segmentation
    PAN_HOLD         = 2.0    # seconds POINT must be held before pan activates
    PAN_DEADZONE     = 0.004
    LOCK_HOLD        = 2.0    # seconds L-shape held to lock

    COOLDOWNS = {
        "SWIPE_UP"  : 0.35,
        "SWIPE_DOWN": 0.35,
        "PINCH"     : 0.7,
        "FIST"      : 0.7,
        "OPEN_PALM" : 1.0,
        "PEACE"     : 0.8,
        "POINT"     : 0.05,
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

        # Vote buffers (right hand only for tokens)
        self._votes      = deque(maxlen=self.VOTE_WINDOW)
        self._last_fired = {}

        # Swipe detection
        self._swipe_buf  = deque(maxlen=8)
        self._last_swipe = 0.0

        # Pinch zoom
        self._pinch_prev  = None
        self._pinch_accum = 0.0

        # Pan
        self._pan_start    = 0.0
        self._pan_ready    = False
        self._pan_prev_pos = None
        self._pan_active   = False

        # Windowing (two index fingers)
        self._win_prev_dist = None

        # Lock (L-shape hold)
        self._lock_start  = 0.0
        self._lock_held   = False
        self._lock_fired  = False

        # Segmentation (both thumbs 3s)
        self._seg_start = 0.0
        self._seg_held  = False
        self._seg_fired = False

        # Two-hand gestures
        self._two_gesture = None
        self._two_start   = 0.0
        self._two_fired   = False
        self._two_last    = {}

        # Status overlay
        self._status_msg  = ""
        self._status_time = 0.0
        # Both-palm view cycle
        self._both_palm_start = 0.0
        self._both_palm_held  = False
        self._both_palm_fired = False

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
        detector  = vision.HandLandmarker.create_from_options(options)
        cap       = cv2.VideoCapture(0)
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

            # ── CLAHE on V channel (dim light fix) ──
            hsv          = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            clahe        = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
            enhanced     = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            rgb    = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            left_lm  = None
            right_lm = None

            if result.hand_landmarks and result.handedness:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    lm   = [(lm.x, lm.y) for lm in hand_lms]
                    side = result.handedness[i][0].category_name
                    col  = (0, 200, 100) if side == "Right" else (200, 100, 0)
                    self._draw_hand(frame, hand_lms, w, h, col)
                    if side == "Left":
                        left_lm = lm
                    else:
                        right_lm = lm

            ctrl_lm = right_lm or left_lm

            # ── Swipe (velocity-based) ──────────
            if ctrl_lm:
                self._detect_swipe(ctrl_lm, now)

            # ── Pinch zoom ──────────────────────
            if ctrl_lm and not self._pan_active:
                self._detect_pinch_zoom(ctrl_lm, now)
            else:
                self._pinch_prev  = None
                self._pinch_accum = 0.0

            # ── Pan (index hold 2s) ─────────────
            if ctrl_lm:
                self._detect_pan(ctrl_lm, now)

            # ── Windowing (two index fingers) ───
            if left_lm and right_lm:
                self._detect_windowing(left_lm, right_lm, now)
            else:
                self._win_prev_dist = None

            # ── Lock (L-shape hold) ─────────────
            if ctrl_lm:
                self._detect_lock(ctrl_lm, now)

            # ── Segmentation (both thumbs 3s) ───
            if left_lm and right_lm:
                self._detect_segmentation(left_lm, right_lm, now)
            else:
                self._seg_held  = False
                self._seg_fired = False

            # ── Two-hand gestures ───────────────
            if left_lm and right_lm:
                self._detect_two_hand(left_lm, right_lm, now)
            else:
                self._two_gesture = None
                self._two_fired   = False

            # ── Static token voting (right hand) ─
            if right_lm and not self._pan_active:
                raw  = self._classify_static(right_lm)
                self._votes.append(raw)
                token = self._resolve_votes()
                if token and token not in ("SWIPE_UP", "SWIPE_DOWN"):
                    self._emit_token(token)

            elif not right_lm:
                self._votes.clear()

            # ── Draw UI ─────────────────────────
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now
            frame     = self._draw_ui(frame, ctrl_lm, left_lm, right_lm,
                                      fps, w, h, now)
            # Both palms → cycle view
            if left_lm and right_lm:
                self._detect_both_palm(left_lm, right_lm, now)
            else:
                self._both_palm_held = False; self._both_palm_fired = False

            self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False

    # ──────────────────────────────────────────
    #  Detectors
    # ──────────────────────────────────────────

    def _detect_swipe(self, lm, now):
        self._swipe_buf.append((lm[0][1], now))
        if len(self._swipe_buf) < 4:
            return
        dy = self._swipe_buf[-1][0] - self._swipe_buf[0][0]
        dt = self._swipe_buf[-1][1] - self._swipe_buf[0][1]
        if dt < 0.001:
            return
        velocity = abs(dy) / dt
        if velocity < self.SWIPE_VEL_MIN:
            return
        if now - self._last_swipe < self.SWIPE_COOLDOWN:
            return
        if abs(dy) > 0.04:
            self._last_swipe = now
            self._swipe_buf.clear()
            token = "SWIPE_UP" if dy < 0 else "SWIPE_DOWN"
            self._emit_token_direct(token)
            self._status(f"Swipe {'UP' if dy < 0 else 'DOWN'}")

    def _detect_pinch_zoom(self, lm, now):
        tx, ty = lm[4]; ix, iy = lm[8]
        dist   = math.hypot(tx - ix, ty - iy)
        palm   = math.hypot(lm[9][0] - lm[0][0], lm[9][1] - lm[0][1])
        if palm < 0.01:
            return
        dist_n = dist / palm
        if self._pinch_prev is None:
            self._pinch_prev = dist_n
            return
        delta             = dist_n - self._pinch_prev
        self._pinch_prev  = dist_n
        self._pinch_accum += delta
        if abs(self._pinch_accum) > 0.08:
            token             = "PINCH" if self._pinch_accum > 0 else "FIST"
            self._pinch_accum = 0.0
            self._emit_token_direct(token)
            self._status(f"Pinch {'open=ZoomIn' if token == 'PINCH' else 'close=ZoomOut'}")

    def _detect_pan(self, lm, now):
        fingers   = self._fingers_up(lm)
        is_point  = (fingers[1] == 1 and fingers[2] == 0 and
                     fingers[3] == 0 and fingers[4] == 0)
        if is_point:
            if self._pan_start == 0.0:
                self._pan_start = now
            held = now - self._pan_start
            if held >= self.PAN_HOLD:
                self._pan_ready = True
                pos = lm[8]
                if self._pan_prev_pos is not None and self._pan_ready:
                    dx = pos[0] - self._pan_prev_pos[0]
                    dy = pos[1] - self._pan_prev_pos[1]
                    if abs(dx) > self.PAN_DEADZONE or abs(dy) > self.PAN_DEADZONE:
                        self.pan_moved.emit(dx, dy)
                        self._pan_active = True
                        self._status("Pan ACTIVE")
                self._pan_prev_pos = pos
            else:
                self._status(f"Hold to pan {held:.1f}s/2s")
        else:
            self._pan_start    = 0.0
            self._pan_ready    = False
            self._pan_prev_pos = None
            self._pan_active   = False

    def _detect_windowing(self, left_lm, right_lm, now):
        lf = self._fingers_up(left_lm)
        rf = self._fingers_up(right_lm)
        if not (lf[1] == 1 and lf[2] == 0 and rf[1] == 1 and rf[2] == 0):
            self._win_prev_dist = None
            return
        dist = math.hypot(right_lm[8][0] - left_lm[8][0],
                          right_lm[8][1] - left_lm[8][1])
        if self._win_prev_dist is None:
            self._win_prev_dist = dist
            return
        delta               = dist - self._win_prev_dist
        self._win_prev_dist = dist
        if abs(delta) > 0.003:
            self.window_changed.emit(delta * 1500, 0)
            self._status(f"WL {'up' if delta > 0 else 'down'}")

    def _detect_lock(self, lm, now):
        if self._is_L_shape(lm):
            if not self._lock_held:
                self._lock_held  = True
                self._lock_start = now
                self._lock_fired = False
            held = now - self._lock_start
            if held >= self.LOCK_HOLD and not self._lock_fired:
                self._lock_fired = True
                self.lock_toggled.emit()
                self._status("L-shape → Lock/Unlock")
        else:
            self._lock_held  = False
            self._lock_fired = False

    def _detect_segmentation(self, left_lm, right_lm, now):
        if self._is_thumb_up(left_lm) and self._is_thumb_up(right_lm):
            if not self._seg_held:
                self._seg_held  = True
                self._seg_start = now
                self._seg_fired = False
            held = now - self._seg_start
            if held >= self.SEG_HOLD and not self._seg_fired:
                self._seg_fired = True
                self.seg_triggered.emit()
                self._status("Both Thumbs → SEGMENTATION!")
            else:
                self._status(f"Seg hold {held:.1f}s/3s")
        else:
            self._seg_held  = False
            self._seg_fired = False

    def _detect_two_hand(self, left_lm, right_lm, now):
        lg = self._classify_static(left_lm)
        rg = self._classify_static(right_lm)
        two = None
        if lg == "PEACE"     and rg == "PEACE"    : two = "BOTH_PEACE"
        elif lg == "FIST"    and rg == "FIST"     : two = "BOTH_FIST"
        elif lg == "OPEN_PALM" and rg == "OPEN_PALM": two = "BOTH_PALM"
        elif self._is_L_shape(left_lm) and self._is_L_shape(right_lm): two = "BOTH_L"

        if two:
            if self._two_gesture != two:
                self._two_gesture = two
                self._two_start   = now
                self._two_fired   = False
            else:
                held = now - self._two_start
                if held >= self.TWO_HAND_HOLD and not self._two_fired:
                    cooldown = self._two_last.get(two, 0)
                    if now - cooldown >= 2.0:
                        self._two_fired      = True
                        self._two_last[two]  = now
                        self._fire_two_hand(two)
        else:
            self._two_gesture = None
            self._two_fired   = False

    def _fire_two_hand(self, two: str):
        if two == "BOTH_PEACE":
            self.lock_toggled.emit()
            self._status("Both PEACE → Lock")
        elif two == "BOTH_FIST":
            self.reset_triggered.emit()
            self._status("Both FIST → Reset")
        elif two == "BOTH_PALM":
            self.overlay_toggled.emit()
            self._status("Both PALM → Overlay")
        elif two == "BOTH_L":
            self.lock_toggled.emit()   # L+L = unlock (same signal, state decides)
            self._status("Both L → Unlock")

    # ──────────────────────────────────────────
    #  Classifiers
    # ──────────────────────────────────────────

    def _classify_static(self, lm) -> str | None:
        f      = self._fingers_up(lm)
        n      = sum(f)
        if n == 0:                      return "FIST"
        if n == 5:                      return "OPEN_PALM"
        if f == [0, 1, 0, 0, 0]:       return "POINT"
        if f == [0, 1, 1, 0, 0]:       return "PEACE"
        if self._is_pinch_static(lm):  return "PINCH"
        if self._is_L_shape(lm):       return "L_SHAPE"
        return None

    def _resolve_votes(self) -> str | None:
        if len(self._votes) < self.VOTE_WINDOW:
            return None
        counts = Counter(g for g in self._votes if g is not None)
        if not counts:
            return None
        top, count = counts.most_common(1)[0]
        if count / self.VOTE_WINDOW >= self.VOTE_THRESHOLD:
            return top
        return None

    def _emit_token(self, token: str):
        """Emit token through vote buffer with cooldown."""
        cooldown = self.COOLDOWNS.get(token, 0.8)
        now      = time.time()
        if now - self._last_fired.get(token, 0) >= cooldown:
            self._last_fired[token] = now
            self.gesture_token.emit(token)

    def _emit_token_direct(self, token: str):
        """Emit token bypassing vote buffer (for swipe/pinch)."""
        self.gesture_token.emit(token)


    def _detect_both_palm(self, left_lm, right_lm, now):
        """Both open palms held 1s → cycle view mode (CT / MRI / Overlay)."""
        lf = self._fingers_up(left_lm)
        rf = self._fingers_up(right_lm)
        if sum(lf) == 5 and sum(rf) == 5:
            if not self._both_palm_held:
                self._both_palm_held  = True
                self._both_palm_start = now
                self._both_palm_fired = False
            held = now - self._both_palm_start
            if held >= 1.0 and not self._both_palm_fired:
                self._both_palm_fired = True
                self.cycle_view_mode.emit()
                self._status("Both Palms 1s → Cycle View")
        else:
            self._both_palm_held  = False
            self._both_palm_fired = False

    def _fingers_up(self, lm):
        tips  = [4, 8, 12, 16, 20]
        pip   = [3, 6, 10, 14, 18]
        thumb = 1 if lm[tips[0]][0] > lm[pip[0]][0] else 0
        rest  = [1 if lm[tips[i]][1] < lm[pip[i]][1] else 0 for i in range(1, 5)]
        return [thumb] + rest

    def _is_pinch_static(self, lm) -> bool:
        return math.hypot(lm[4][0]-lm[8][0], lm[4][1]-lm[8][1]) < 0.025

    def _is_L_shape(self, lm) -> bool:
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        return (lm[tips[1]][1] < lm[pip[1]][1] and
                lm[tips[2]][1] > lm[pip[2]][1] and
                lm[tips[3]][1] > lm[pip[3]][1] and
                lm[tips[4]][1] > lm[pip[4]][1] and
                abs(lm[tips[0]][0] - lm[0][0]) > 0.07)

    def _is_thumb_up(self, lm) -> bool:
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        thumb_above   = (lm[0][1] - lm[4][1]) > 0.12
        fingers_down  = all(lm[tips[i]][1] > lm[pips[i]][1] for i in range(4))
        return thumb_above and fingers_down

    # ──────────────────────────────────────────
    #  Drawing
    # ──────────────────────────────────────────

    def _draw_hand(self, frame, hand_lms, w, h, color):
        CONN = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),(0,17)]
        pts  = [(int(lm.x*w), int(lm.y*h)) for lm in hand_lms]
        for a, b in CONN:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 3, color, -1)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(frame, pts[tip], 6, COL_YELLOW, -1)

    def _status(self, msg: str):
        self._status_msg  = msg
        self._status_time = time.time()

    def _draw_ui(self, frame, ctrl_lm, left_lm, right_lm, fps, w, h, now):
        # FPS
        cv2.putText(frame, f"FPS:{fps:.0f}", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 2)

        # Pan hold charge bar
        if ctrl_lm and not self._pan_active and self._pan_start > 0:
            charge = min((now - self._pan_start) / self.PAN_HOLD, 1.0)
            bx, by = 8, 35
            cv2.rectangle(frame, (bx, by), (bx+120, by+7), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*120), by+7), COL_CYAN, -1)
            cv2.putText(frame, "PAN HOLD", (bx, by-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1)

        if self._pan_active:
            cv2.putText(frame, "PAN", (w-55, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_CYAN, 2)

        # Lock charge bar
        if self._lock_held and not self._lock_fired:
            charge = min((now - self._lock_start) / self.LOCK_HOLD, 1.0)
            bx, by = 8, 52
            cv2.rectangle(frame, (bx, by), (bx+120, by+7), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*120), by+7), COL_ORANGE, -1)
            cv2.putText(frame, "LOCK HOLD", (bx, by-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180,180,180), 1)

        # Seg charge bar
        if self._seg_held and not self._seg_fired:
            charge = min((now - self._seg_start) / self.SEG_HOLD, 1.0)
            bx, by = w//2 - 80, 8
            cv2.rectangle(frame, (bx, by), (bx+160, by+10), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*160), by+10), (0,200,80), -1)
            cv2.putText(frame, f"SEG {now-self._seg_start:.1f}s/3s",
                        (bx, by+22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,255,180), 1)

        # Two-hand charge bar
        if self._two_gesture and not self._two_fired:
            charge = min((now - self._two_start) / self.TWO_HAND_HOLD, 1.0)
            label  = self._two_gesture.replace("BOTH_", "")
            bx, by = w//2 - 60, h - 55
            cv2.rectangle(frame, (bx, by), (bx+120, by+8), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(charge*120), by+8), COL_ORANGE, -1)
            cv2.putText(frame, label, (bx, by-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)

        # Status message (fades over 2s)
        if self._status_msg and now - self._status_time < 2.0:
            alpha = max(0.0, 1.0 - (now - self._status_time) / 2.0)
            col   = tuple(int(c * alpha) for c in COL_WHITE)
            cv2.putText(frame, self._status_msg, (8, h - 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1)

        # Bottom hint bar (alternates every 2s)
        cv2.rectangle(frame, (0, h-30), (w, h), (10,10,10), -1)
        hints = [
            "Pinch=Zoom  Point+2s=Pan  3fin=Reset  LHold=Lock",
            "ThumThumbs3s=Seg  2index=WL  BothPeace=Lock",
        ]
        cv2.putText(frame, hints[int(now*0.5) % 2], (6, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (90,90,90), 1)

        # Both-palm charge bar
        if self._both_palm_held and not self._both_palm_fired:
            held   = min(now - self._both_palm_start, 1.0)
            bx, by = w//2 - 60, 30
            cv2.rectangle(frame, (bx, by), (bx+120, by+8), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+int(held*120), by+8), (200,160,0), -1)
            cv2.putText(frame, "CYCLE VIEW", (bx, by-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (220,200,100), 1)
        return frame
