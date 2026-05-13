"""
Patch: Add panning + view lock/unlock to phase4_segmentation.py

New features:
  Pan    — hold POINT_UP and move hand left/right/up/down
  Lock   — hold PEACE on RIGHT hand for 1.5s → locks zoom+pan+slice
  Unlock — hold PEACE again for 1.5s → unlocks
  Visual — red border + LOCKED label when locked
           camera overlay shows lock state
"""

with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

# ─────────────────────────────────────────────────────────────
# 1. Add pan_x, pan_y, view_locked to MainWindow.__init__
# ─────────────────────────────────────────────────────────────
old = '''        self.ai_mode      = False
        self.seg_mask     = None
        self.show_overlay = False
        self.opacity      = 0.45
        self._setup_ui()'''

new = '''        self.ai_mode      = False
        self.seg_mask     = None
        self.show_overlay = False
        self.opacity      = 0.45
        self.pan_x        = 0      # pan offset in pixels (image coords)
        self.pan_y        = 0
        self.view_locked  = False  # True = zoom/pan/slice locked
        self._setup_ui()'''

assert old in src, 'BLOCK 1 not found'
src = src.replace(old, new)
print('✅ 1. pan_x/pan_y/view_locked added to __init__')


# ─────────────────────────────────────────────────────────────
# 2. Add lock indicator label to LEFT panel (after lbl_ai_indicator)
# ─────────────────────────────────────────────────────────────
old = '''        self.lbl_ai_indicator = QLabel("AI Mode: OFF"); self.lbl_ai_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;"); gg.addWidget(self.lbl_ai_indicator)
        for line in ["LEFT FIST (hold 1s) → AI Mode","RIGHT Swipe Up → Prev Slice","RIGHT Swipe Down → Next Slice","RIGHT Pinch → Zoom In","RIGHT Fist → Zoom Out","RIGHT Open Palm → Reset","RIGHT Point Up → Scroll Up","RIGHT Peace → Scroll Down"]:'''

new = '''        self.lbl_ai_indicator = QLabel("AI Mode: OFF"); self.lbl_ai_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;"); gg.addWidget(self.lbl_ai_indicator)

        self.lbl_lock_indicator = QLabel("🔓 View: UNLOCKED")
        self.lbl_lock_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;")
        gg.addWidget(self.lbl_lock_indicator)

        for line in ["LEFT FIST (hold 1s)  → AI Mode","RIGHT Swipe Up       → Prev Slice","RIGHT Swipe Down     → Next Slice","RIGHT Pinch          → Zoom In","RIGHT Fist           → Zoom Out","RIGHT Open Palm      → Reset","RIGHT Point Up       → Pan Mode","RIGHT Peace (hold)   → Lock / Unlock View","─────────────────────────────────","When LOCKED: all navigation disabled","Pan = hold POINT_UP + move hand"]:'''

assert old in src, 'BLOCK 2 not found'
src = src.replace(old, new)
print('✅ 2. Lock indicator label added to gesture panel')


# ─────────────────────────────────────────────────────────────
# 3. Update _on_gesture to respect lock and handle pan
# ─────────────────────────────────────────────────────────────
old = '''    def _on_gesture(self, gesture, hand):
        action = GESTURE_ACTIONS.get(gesture, gesture)
        self.lbl_gesture.setText(f"{gesture}\\n{action}")
        self.status.showMessage(f"{hand.upper()} hand: {gesture}  →  {action}")
        if   gesture == "SWIPE_UP"  : self._prev_slice()
        elif gesture == "SWIPE_DOWN": self._next_slice()
        elif gesture == "OPEN_PALM" : self._reset_view()
        elif gesture == "POINT_UP"  : self._prev_slice()
        elif gesture == "PEACE"     : self._next_slice()
        elif gesture == "PINCH"     : self._zoom_in()
        elif gesture == "FIST"      : self._zoom_out()'''

new = '''    def _on_gesture(self, gesture, hand):
        action = GESTURE_ACTIONS.get(gesture, gesture)
        self.lbl_gesture.setText(f"{gesture}\\n{action}")
        self.status.showMessage(f"{hand.upper()} hand: {gesture}  →  {action}")

        # When view is locked, block all navigation except unlock
        if self.view_locked:
            self.status.showMessage("🔒 View LOCKED — hold PEACE 1.5s to unlock")
            return

        if   gesture == "SWIPE_UP"  : self._prev_slice()
        elif gesture == "SWIPE_DOWN": self._next_slice()
        elif gesture == "OPEN_PALM" : self._reset_view()
        elif gesture == "PINCH"     : self._zoom_in()
        elif gesture == "FIST"      : self._zoom_out()
        # POINT_UP and PEACE are now handled by hold-timer in gesture engine
        # (pan and lock respectively) — single-tap still scrolls as fallback
        elif gesture == "POINT_UP"  : self._prev_slice()
        elif gesture == "PEACE"     : self._next_slice()'''

assert old in src, 'BLOCK 3 not found'
src = src.replace(old, new)
print('✅ 3. _on_gesture updated with lock guard')


# ─────────────────────────────────────────────────────────────
# 4. Add _toggle_lock, _pan, update _reset_view and _show_slice
# ─────────────────────────────────────────────────────────────
old = '''    def _on_ai_mode_changed(self, active):'''

new = '''    # ── View Lock ──────────────────────────────────────────────
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
        """Called from gesture engine with normalised deltas (-1 to 1)."""
        if self.volume is None or self.view_locked:
            return
        h, w = self.volume[self.current].shape
        self._pan(int(dx_norm * w * 0.15), int(dy_norm * h * 0.15))

    def _on_ai_mode_changed(self, active):'''

assert old in src, 'BLOCK 4 not found'
src = src.replace(old, new)
print('✅ 4. _toggle_lock and _pan methods added')


# ─────────────────────────────────────────────────────────────
# 5. Update _reset_view to also reset pan
# ─────────────────────────────────────────────────────────────
old = '''    def _reset_view(self):
        if self.volume is not None:
            self.current=self.volume.shape[0]//2; self.zoom=1.0
            self.slice_slider.setValue(self.current); self._reset_windowing()'''

new = '''    def _reset_view(self):
        if self.volume is not None:
            self.current  = self.volume.shape[0]//2
            self.zoom     = 1.0
            self.pan_x    = 0
            self.pan_y    = 0
            self.view_locked = False
            self.lbl_lock_indicator.setText("🔓 View: UNLOCKED")
            self.lbl_lock_indicator.setStyleSheet("color:#666; font-size:11px; qproperty-alignment:AlignCenter;")
            self.slice_slider.setValue(self.current)
            self._reset_windowing()'''

assert old in src, 'BLOCK 5 not found'
src = src.replace(old, new)
print('✅ 5. _reset_view updated to clear pan + lock')


# ─────────────────────────────────────────────────────────────
# 6. Update _show_slice to use pan_x/pan_y
# ─────────────────────────────────────────────────────────────
old = '''        if self.zoom != 1.0:
            h,w=img.shape; new_h=int(h/self.zoom); new_w=int(w/self.zoom)
            cy,cx=h//2,w//2
            y1=max(0,cy-new_h//2); y2=min(h,cy+new_h//2); x1=max(0,cx-new_w//2); x2=min(w,cx+new_w//2)
            img=img[y1:y2,x1:x2]; img=cv2.resize(img,(w,h),interpolation=cv2.INTER_LINEAR)'''

new = '''        if self.zoom != 1.0 or (self.pan_x != 0 or self.pan_y != 0):
            h, w   = img.shape
            new_h  = int(h / self.zoom)
            new_w  = int(w / self.zoom)
            # Pan offsets shift the crop centre
            cy = h//2 + self.pan_y
            cx = w//2 + self.pan_x
            y1 = max(0, cy - new_h//2); y2 = min(h, y1 + new_h)
            x1 = max(0, cx - new_w//2); x2 = min(w, x1 + new_w)
            # Clamp so we never crop outside image
            if y2 > h: y1 = max(0, h - new_h); y2 = h
            if x2 > w: x1 = max(0, w - new_w); x2 = w
            img = cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)'''

assert old in src, 'BLOCK 6 not found'
src = src.replace(old, new)
print('✅ 6. _show_slice updated with pan-aware crop')


# ─────────────────────────────────────────────────────────────
# 7. Update mask zoom block to also use pan
# ─────────────────────────────────────────────────────────────
old = '''            # Apply zoom to MASK the same way as the image
            # (crop same region, resize to same display size)
            if self.zoom != 1.0:
                h0, w0 = mask_slice.shape
                new_h = int(h0 / self.zoom); new_w = int(w0 / self.zoom)
                cy, cx = h0//2, w0//2
                y1=max(0,cy-new_h//2); y2=min(h0,cy+new_h//2)
                x1=max(0,cx-new_w//2); x2=min(w0,cx+new_w//2)
                mask_slice = cv2.resize(mask_slice[y1:y2, x1:x2].astype(np.float32),
                                        (w0, h0),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)'''

new = '''            # Apply SAME zoom + pan to mask as to the image
            if self.zoom != 1.0 or (self.pan_x != 0 or self.pan_y != 0):
                h0, w0 = mask_slice.shape
                new_h  = int(h0 / self.zoom); new_w = int(w0 / self.zoom)
                cy = h0//2 + self.pan_y; cx = w0//2 + self.pan_x
                y1 = max(0, cy - new_h//2); y2 = min(h0, y1 + new_h)
                x1 = max(0, cx - new_w//2); x2 = min(w0, x1 + new_w)
                if y2 > h0: y1 = max(0, h0 - new_h); y2 = h0
                if x2 > w0: x1 = max(0, w0 - new_w); x2 = w0
                mask_slice = cv2.resize(mask_slice[y1:y2, x1:x2].astype(np.float32),
                                        (w0, h0),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)'''

assert old in src, 'BLOCK 7 not found'
src = src.replace(old, new)
print('✅ 7. Mask crop updated to match pan-aware image crop')


# ─────────────────────────────────────────────────────────────
# 8. Add lock signal to TwoHandGestureEngine and wire it up
# ─────────────────────────────────────────────────────────────
old = '''class TwoHandGestureEngine(QThread):
    gesture_detected = pyqtSignal(str, str)
    frame_ready      = pyqtSignal(np.ndarray)
    ai_mode_changed  = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.running           = True
        self.prev_pos_right    = []
        self.prev_pos_left     = []
        self.smooth_window     = 5
        self.last_time         = {}
        self._left_fist_held   = False
        self._left_fist_start  = 0'''

new = '''class TwoHandGestureEngine(QThread):
    gesture_detected = pyqtSignal(str, str)
    frame_ready      = pyqtSignal(np.ndarray)
    ai_mode_changed  = pyqtSignal(bool)
    lock_toggled     = pyqtSignal()       # fired when PEACE held 1.5s on right hand
    pan_moved        = pyqtSignal(float, float)  # dx, dy normalised

    LOCK_HOLD_SECONDS = 1.5
    PAN_SMOOTH        = 6    # frames of position history for pan smoothing

    def __init__(self):
        super().__init__()
        self.running              = True
        self.prev_pos_right       = []
        self.prev_pos_left        = []
        self.smooth_window        = 5
        self.last_time            = {}
        self._left_fist_held      = False
        self._left_fist_start     = 0
        # Lock
        self._peace_held          = False
        self._peace_start         = 0.0
        self._lock_toggled_this   = False
        # Pan
        self._pan_history         = []
        self._pan_active          = False
        self._pan_prev_pos        = None'''

assert old in src, 'BLOCK 8 not found'
src = src.replace(old, new)
print('✅ 8. lock_toggled + pan_moved signals added to engine')


# ─────────────────────────────────────────────────────────────
# 9. Add pan + lock detection to right-hand section of run()
# ─────────────────────────────────────────────────────────────
old = '''            if right_landmarks:
                gesture = self._classify_full(right_landmarks, "right")
                if gesture:
                    self.gesture_detected.emit(gesture, "right")'''

new = '''            if right_landmarks:
                gesture = self._classify_full(right_landmarks, "right")

                # ── PEACE hold → lock/unlock ──────────
                if gesture == "PEACE":
                    if not self._peace_held:
                        self._peace_held        = True
                        self._peace_start       = time.time()
                        self._lock_toggled_this = False
                    held = time.time() - self._peace_start
                    if held >= self.LOCK_HOLD_SECONDS and not self._lock_toggled_this:
                        self._lock_toggled_this = True
                        self.lock_toggled.emit()
                        gesture = None  # consume gesture, don't also fire PEACE
                else:
                    self._peace_held = False

                # ── POINT_UP + movement → pan ─────────
                fingers_up = self._fingers_up(right_landmarks)
                is_point   = (fingers_up == [0, 1, 0, 0, 0])
                if is_point:
                    wrist_pos = right_landmarks[0]
                    if self._pan_prev_pos is not None:
                        dx = wrist_pos[0] - self._pan_prev_pos[0]
                        dy = wrist_pos[1] - self._pan_prev_pos[1]
                        # Only emit if movement is significant (dead zone)
                        if abs(dx) > 0.008 or abs(dy) > 0.008:
                            self.pan_moved.emit(dx, dy)
                    self._pan_prev_pos  = wrist_pos
                    self._pan_active    = True
                    gesture = None  # pan mode: don't also fire POINT_UP
                else:
                    self._pan_prev_pos = None
                    self._pan_active   = False

                if gesture:
                    self.gesture_detected.emit(gesture, "right")'''

assert old in src, 'BLOCK 9 not found'
src = src.replace(old, new)
print('✅ 9. Pan + lock hold detection added to run()')


# ─────────────────────────────────────────────────────────────
# 10. Connect new signals in _start_gesture_engine
# ─────────────────────────────────────────────────────────────
old = '''    def _start_gesture_engine(self):
        self.gesture_engine = TwoHandGestureEngine()
        self.gesture_engine.gesture_detected.connect(self._on_gesture)
        self.gesture_engine.frame_ready.connect(self._on_cam_frame)
        self.gesture_engine.ai_mode_changed.connect(self._on_ai_mode_changed)
        self.gesture_engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")'''

new = '''    def _start_gesture_engine(self):
        self.gesture_engine = TwoHandGestureEngine()
        self.gesture_engine.gesture_detected.connect(self._on_gesture)
        self.gesture_engine.frame_ready.connect(self._on_cam_frame)
        self.gesture_engine.ai_mode_changed.connect(self._on_ai_mode_changed)
        self.gesture_engine.lock_toggled.connect(self._toggle_lock)
        self.gesture_engine.pan_moved.connect(self._on_pan_gesture)
        self.gesture_engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")'''

assert old in src, 'BLOCK 10 not found'
src = src.replace(old, new)
print('✅ 10. lock_toggled and pan_moved signals connected')


# ─────────────────────────────────────────────────────────────
# 11. Add lock/pan overlay to camera feed _draw_overlay
# ─────────────────────────────────────────────────────────────
old = '''        cv2.rectangle(frame, (0, h-35), (w, h), (10,10,10), -1)
        if not left_lm and not right_lm:
            cv2.putText(frame, "Show hands to camera",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
        elif not ai_mode:
            cv2.putText(frame, "Hold LEFT FIST 1 sec to activate AI",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ORANGE, 1)
        else:
            cv2.putText(frame, "AI active | Use RIGHT hand to navigate",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GREEN, 1)
        return frame'''

new = '''        # Peace hold bar (lock gesture charging)
        if self._peace_held:
            held   = min(time.time() - self._peace_start, self.LOCK_HOLD_SECONDS)
            charge = held / self.LOCK_HOLD_SECONDS
            bar_w  = int(charge * 140)
            bx, by = w//2 - 70, h - 55
            cv2.rectangle(frame, (bx, by), (bx+140, by+8), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+bar_w, by+8), (0, 80, 255), -1)
            cv2.putText(frame, "LOCK HOLD", (bx, by-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1)

        # Pan active indicator
        if self._pan_active:
            cv2.putText(frame, "PAN MODE", (w-90, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_YELLOW, 2)

        cv2.rectangle(frame, (0, h-35), (w, h), (10,10,10), -1)
        if not left_lm and not right_lm:
            cv2.putText(frame, "Show hands to camera",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80,80,80), 1)
        elif not ai_mode:
            cv2.putText(frame, "Hold LEFT FIST to toggle AI | PEACE hold = Lock",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ORANGE, 1)
        else:
            cv2.putText(frame, "AI active | POINT_UP=Pan | PEACE hold=Lock",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1)
        return frame'''

assert old in src, 'BLOCK 11 not found'
src = src.replace(old, new)
print('✅ 11. Camera overlay updated with lock/pan indicators')


# ─────────────────────────────────────────────────────────────
# 12. Update GESTURE_ACTIONS dict
# ─────────────────────────────────────────────────────────────
old = '''GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom In",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Zoom Out",
    "POINT_UP"   : "Scroll Up",
    "PEACE"      : "Scroll Down",
}'''

new = '''GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom In",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Zoom Out",
    "POINT_UP"   : "Pan Mode",
    "PEACE"      : "Lock / Unlock (hold 1.5s)",
}'''

assert old in src, 'BLOCK 12 not found'
src = src.replace(old, new)
print('✅ 12. GESTURE_ACTIONS updated')


# ─────────────────────────────────────────────────────────────
# Write and verify
# ─────────────────────────────────────────────────────────────
with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print('\n✅ Syntax OK — run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'\n❌ Syntax error at line {e.lineno}: {e.msg}')
    print(f'   Text: {e.text}')