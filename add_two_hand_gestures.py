"""
Patch: Replace single-hand lock/unlock with intuitive two-hand gestures.

TWO-HAND GESTURES (both hands simultaneously):
  Both PEACE  ✌️✌️  → Lock View
  Both L-shape 👍☝️  → Unlock View
  Both FIST   👊👊  → Reset View
  Both PALM   🖐🖐  → Toggle Overlay

L-shape = thumb extended + index extended, other fingers down
(looks like the letter L)

RIGHT HAND (navigation, unchanged):
  Swipe Up/Down → Prev/Next Slice
  Pinch         → Zoom In
  Fist          → Zoom Out
  POINT_UP+move → Pan

LEFT HAND:
  FIST hold 1s  → AI Mode toggle
"""

with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []

# ─────────────────────────────────────────────────────────────
# 1. Update GESTURE_ACTIONS
# ─────────────────────────────────────────────────────────────
old = '''GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom In",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Zoom Out",
    "POINT_UP"   : "Pan Mode",
    "PEACE"      : "Lock / Unlock (hold 1.5s)",
}'''

new = '''GESTURE_ACTIONS = {
    "SWIPE_UP"        : "Previous Slice",
    "SWIPE_DOWN"      : "Next Slice",
    "PINCH"           : "Zoom In",
    "FIST"            : "Zoom Out",
    "POINT_UP"        : "Pan Mode",
    # Two-hand gestures
    "BOTH_PEACE"      : "Lock View",
    "BOTH_L"          : "Unlock View",
    "BOTH_FIST"       : "Reset View",
    "BOTH_PALM"       : "Toggle Overlay",
}'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 1. GESTURE_ACTIONS updated')
else:
    # Try old format
    old2 = '''GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom In",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Zoom Out",
    "POINT_UP"   : "Scroll Up",
    "PEACE"      : "Scroll Down",
}'''
    if old2 in src:
        src = src.replace(old2, new)
        fixes.append('✅ 1. GESTURE_ACTIONS updated (alt form)')
    else:
        fixes.append('⚠️  1. GESTURE_ACTIONS not found — update manually')


# ─────────────────────────────────────────────────────────────
# 2. Add two-hand signals to TwoHandGestureEngine
# ─────────────────────────────────────────────────────────────
old = '''    lock_toggled     = pyqtSignal()       # fired when PEACE held 1.5s on right hand
    pan_moved        = pyqtSignal(float, float)  # dx, dy normalised'''

new = '''    two_hand_gesture = pyqtSignal(str)    # "BOTH_PEACE", "BOTH_L", "BOTH_FIST", "BOTH_PALM"
    pan_moved        = pyqtSignal(float, float)  # dx, dy normalised'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 2. two_hand_gesture signal added')
else:
    fixes.append('⚠️  2. Signal block not found — may need manual edit')


# ─────────────────────────────────────────────────────────────
# 3. Replace lock hold constants with two-hand detection constants
# ─────────────────────────────────────────────────────────────
old = '''    LOCK_HOLD_SECONDS = 1.5
    PAN_SMOOTH        = 6    # frames of position history for pan smoothing'''

new = '''    TWO_HAND_HOLD     = 0.6  # seconds both hands must hold gesture to fire
    PAN_SMOOTH        = 6    # frames of position history for pan smoothing'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 3. TWO_HAND_HOLD constant added')
else:
    fixes.append('⚠️  3. Constants not found — skipping')


# ─────────────────────────────────────────────────────────────
# 4. Replace lock-related state vars with two-hand state vars
# ─────────────────────────────────────────────────────────────
old = '''        # Lock
        self._peace_held          = False
        self._peace_start         = 0.0
        self._lock_toggled_this   = False
        # Pan
        self._pan_history         = []
        self._pan_active          = False
        self._pan_prev_pos        = None'''

new = '''        # Two-hand gesture state
        self._two_hand_gesture    = None   # current matching two-hand gesture
        self._two_hand_start      = 0.0
        self._two_hand_fired      = False
        self._last_two_hand       = {}     # cooldown per two-hand gesture
        # Pan
        self._pan_history         = []
        self._pan_active          = False
        self._pan_prev_pos        = None'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 4. Two-hand state vars added')
else:
    fixes.append('⚠️  4. State vars not found — skipping')


# ─────────────────────────────────────────────────────────────
# 5. Replace right-hand PEACE-hold + pan block with new two-hand detection
# ─────────────────────────────────────────────────────────────
old = '''            if right_landmarks:
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

new = '''            if right_landmarks:
                gesture = self._classify_full(right_landmarks, "right")

                # ── POINT_UP + movement → pan ─────────
                fingers_up = self._fingers_up(right_landmarks)
                is_point   = (fingers_up == [0, 1, 0, 0, 0])
                if is_point:
                    wrist_pos = right_landmarks[0]
                    if self._pan_prev_pos is not None:
                        dx = wrist_pos[0] - self._pan_prev_pos[0]
                        dy = wrist_pos[1] - self._pan_prev_pos[1]
                        if abs(dx) > 0.008 or abs(dy) > 0.008:
                            self.pan_moved.emit(dx, dy)
                    self._pan_prev_pos = wrist_pos
                    self._pan_active   = True
                    gesture = None
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
                    self._two_hand_fired   = False'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 5. Two-hand detection block added to run()')
else:
    fixes.append('⚠️  5. Right-hand block not found — skipping')


# ─────────────────────────────────────────────────────────────
# 6. Add _match_two_hand and _is_L_shape helper methods
#    (add before _classify_static)
# ─────────────────────────────────────────────────────────────
old = '''    def _classify_static(self, lm):'''

new = '''    def _match_two_hand(self, left_g, right_g) -> str | None:
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

    def _classify_static(self, lm):'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 6. _match_two_hand and _is_L_shape helpers added')
else:
    fixes.append('⚠️  6. _classify_static not found — skipping')


# ─────────────────────────────────────────────────────────────
# 7. Add L_SHAPE to _classify_static and _classify_full
# ─────────────────────────────────────────────────────────────
old = '''    def _classify_static(self, lm):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)
        if num_up == 0: return "FIST"
        if num_up == 5: return "OPEN_PALM"
        if fingers_up == [0,1,0,0,0]: return "POINT_UP"
        if fingers_up == [0,1,1,0,0]: return "PEACE"
        if self._is_pinch(lm): return "PINCH"
        return None'''

new = '''    def _classify_static(self, lm):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)
        if num_up == 0: return "FIST"
        if num_up == 5: return "OPEN_PALM"
        if fingers_up == [0,1,0,0,0]: return "POINT_UP"
        if fingers_up == [0,1,1,0,0]: return "PEACE"
        if self._is_pinch(lm):        return "PINCH"
        if self._is_L_shape(lm):      return "L_SHAPE"
        return None'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 7. L_SHAPE added to _classify_static')
else:
    fixes.append('⚠️  7. _classify_static body not found')


# ─────────────────────────────────────────────────────────────
# 8. Connect two_hand_gesture signal in _start_gesture_engine
# ─────────────────────────────────────────────────────────────
old = '''        self.gesture_engine.lock_toggled.connect(self._toggle_lock)
        self.gesture_engine.pan_moved.connect(self._on_pan_gesture)'''

new = '''        self.gesture_engine.two_hand_gesture.connect(self._on_two_hand_gesture)
        self.gesture_engine.pan_moved.connect(self._on_pan_gesture)'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 8. two_hand_gesture signal connected')
else:
    fixes.append('⚠️  8. lock_toggled connect line not found — add manually')


# ─────────────────────────────────────────────────────────────
# 9. Add _on_two_hand_gesture handler to MainWindow
# ─────────────────────────────────────────────────────────────
old = '''    # ── View Lock ──────────────────────────────────────────────
    def _toggle_lock(self):'''

new = '''    # ── Two-Hand Gesture Handler ───────────────────────────────
    def _on_two_hand_gesture(self, gesture):
        """Handle two-hand gestures fired from gesture engine."""
        if gesture == "BOTH_PEACE":
            # Lock view (only if not already locked)
            if not self.view_locked:
                self._toggle_lock()
                self.lbl_gesture.setText("✌️✌️ BOTH PEACE\n🔒 View Locked")
        elif gesture == "BOTH_L":
            # Unlock view (only if locked)
            if self.view_locked:
                self._toggle_lock()
                self.lbl_gesture.setText("👍☝️ BOTH L\n🔓 View Unlocked")
        elif gesture == "BOTH_FIST":
            self._reset_view()
            self.lbl_gesture.setText("👊👊 BOTH FIST\n↺ View Reset")
        elif gesture == "BOTH_PALM":
            self._toggle_overlay()
            self.lbl_gesture.setText("🖐🖐 BOTH PALM\n👁 Overlay Toggled")

    # ── View Lock ──────────────────────────────────────────────
    def _toggle_lock(self):'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 9. _on_two_hand_gesture handler added')
else:
    fixes.append('⚠️  9. _toggle_lock not found — add handler manually')


# ─────────────────────────────────────────────────────────────
# 10. Update camera overlay to show two-hand charge bar
# ─────────────────────────────────────────────────────────────
old = '''        # Peace hold bar (lock gesture charging)
        if self._peace_held:
            held   = min(time.time() - self._peace_start, self.LOCK_HOLD_SECONDS)
            charge = held / self.LOCK_HOLD_SECONDS
            bar_w  = int(charge * 140)
            bx, by = w//2 - 70, h - 55
            cv2.rectangle(frame, (bx, by), (bx+140, by+8), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx+bar_w, by+8), (0, 80, 255), -1)
            cv2.putText(frame, "LOCK HOLD", (bx, by-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1)'''

new = '''        # Two-hand gesture charge bar
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
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220,220,220), 1)'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 10. Camera overlay updated with two-hand charge bar')
else:
    fixes.append('⚠️  10. Peace hold bar not found — skipping overlay update')


# ─────────────────────────────────────────────────────────────
# 11. Update gesture guide labels in UI
# ─────────────────────────────────────────────────────────────
old = '''        for line in ["LEFT FIST (hold 1s)  → AI Mode","RIGHT Swipe Up       → Prev Slice","RIGHT Swipe Down     → Next Slice","RIGHT Pinch          → Zoom In","RIGHT Fist           → Zoom Out","RIGHT Open Palm      → Reset","RIGHT Point Up       → Pan Mode","RIGHT Peace (hold)   → Lock / Unlock View","─────────────────────────────────","When LOCKED: all navigation disabled","Pan = hold POINT_UP + move hand"]:'''

new = '''        for line in [
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
        ]:'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 11. Gesture guide labels updated')
else:
    fixes.append('⚠️  11. Gesture guide not found — skipping')


# ─────────────────────────────────────────────────────────────
# 12. Update bottom status bar text in camera overlay
# ─────────────────────────────────────────────────────────────
old = '''        elif not ai_mode:
            cv2.putText(frame, "Hold LEFT FIST to toggle AI | PEACE hold = Lock",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ORANGE, 1)
        else:
            cv2.putText(frame, "AI active | POINT_UP=Pan | PEACE hold=Lock",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1)'''

new = '''        elif not ai_mode:
            cv2.putText(frame, "L.FIST=AI | ✌✌=Lock | LL=Unlock | ✊✊=Reset",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_ORANGE, 1)
        else:
            cv2.putText(frame, "AI ON | ✌✌=Lock | LL=Unlock | POINT+move=Pan",
                        (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_GREEN, 1)'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 12. Camera status bar text updated')
else:
    fixes.append('⚠️  12. Status bar text not found — skipping')


# ─────────────────────────────────────────────────────────────
# Write and verify
# ─────────────────────────────────────────────────────────────
with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(src)

print('\n'.join(fixes))
print()

import ast
try:
    ast.parse(src)
    print('✅ Syntax OK — run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'❌ Syntax error at line {e.lineno}: {e.msg}')
    print(f'   Text: {e.text}')