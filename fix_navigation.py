"""
Fix: Slice navigation + gesture reliability

Problems fixed:
1. POINT_UP consuming gesture before swipe fires
2. Cooldown 1.5s too long — reduced to 0.8s for navigation
3. Swipe detection threshold too tight
4. OPEN_PALM not in GESTURE_ACTIONS
5. Pan active blocks ALL right hand gestures including swipe
6. _classify_full never checks swipe properly when POINT_UP matches first
"""

with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []

# ─────────────────────────────────────────────────────────────
# FIX 1: GESTURE_COOLDOWN too long + add OPEN_PALM back
# ─────────────────────────────────────────────────────────────
old = '''GESTURE_COOLDOWN = 1.5

GESTURE_ACTIONS = {
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

new = '''# Per-gesture cooldowns (seconds)
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
}'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 1. Per-gesture cooldowns + OPEN_PALM restored')
else:
    fixes.append('⚠️  1. GESTURE_COOLDOWN block not found')


# ─────────────────────────────────────────────────────────────
# FIX 2: _classify_full — check swipe FIRST before static poses
#         and don't let POINT_UP block swipe
# ─────────────────────────────────────────────────────────────
old = '''    def _classify_full(self, lm, hand_key):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)
        pos_list   = self.prev_pos_right if hand_key == "right" else self.prev_pos_left
        pos_list.append(lm[0])
        if len(pos_list) > self.smooth_window:
            pos_list.pop(0)

        gesture = None
        if num_up == 5:                    gesture = "OPEN_PALM"
        elif num_up == 0:                  gesture = "FIST"
        elif fingers_up == [0,1,0,0,0]:   gesture = "POINT_UP"
        elif fingers_up == [0,1,1,0,0]:   gesture = "PEACE"
        elif self._is_pinch(lm):          gesture = "PINCH"
        elif len(pos_list) >= self.smooth_window:
            dy = pos_list[-1][1] - pos_list[0][1]
            if dy < -0.06:   gesture = "SWIPE_UP"
            elif dy > 0.06:  gesture = "SWIPE_DOWN"

        if gesture:
            key  = f"{hand_key}_{gesture}"
            now  = time.time()
            if now - self.last_time.get(key, 0) < GESTURE_COOLDOWN:
                gesture = None
            else:
                self.last_time[key] = now
        return gesture'''

new = '''    def _classify_full(self, lm, hand_key):
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
        return gesture'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 2. Swipe detection priority fixed in _classify_full')
else:
    fixes.append('⚠️  2. _classify_full not found')


# ─────────────────────────────────────────────────────────────
# FIX 3: Pan should NOT consume gesture when hand is still
#         Only pan when hand is actively moving + POINT_UP
# ─────────────────────────────────────────────────────────────
old = '''                # ── POINT_UP + movement → pan ─────────
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
                    self.gesture_detected.emit(gesture, "right")'''

new = '''                # ── POINT_UP + ACTIVE MOVEMENT → pan ──────
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
                    self.gesture_detected.emit(gesture, "right")'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 3. Pan only activates on active movement (stationary POINT_UP = swipe)')
else:
    fixes.append('⚠️  3. Pan block not found')


# ─────────────────────────────────────────────────────────────
# FIX 4: _on_gesture — restore OPEN_PALM → reset, fix lock message
# ─────────────────────────────────────────────────────────────
old = '''        if self.view_locked:
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

new = '''        if self.view_locked:
            self.status.showMessage("View LOCKED — Both L-shape to unlock")
            return

        if   gesture == "SWIPE_UP"  : self._prev_slice()
        elif gesture == "SWIPE_DOWN": self._next_slice()
        elif gesture == "OPEN_PALM" : self._reset_view()
        elif gesture == "PINCH"     : self._zoom_in()
        elif gesture == "FIST"      : self._zoom_out()
        # POINT_UP handled by pan engine — no fallback needed
        # PEACE single tap — no action (reserved for two-hand lock)'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 4. _on_gesture cleaned up')
else:
    fixes.append('⚠️  4. _on_gesture block not found')


# ─────────────────────────────────────────────────────────────
# FIX 5: smooth_window too large — reduce for faster response
# ─────────────────────────────────────────────────────────────
old = '''        self.smooth_window        = 5'''
new = '''        self.smooth_window        = 4   # reduced for faster swipe response'''

if old in src:
    src = src.replace(old, new)
    fixes.append('✅ 5. smooth_window reduced 5→4')
else:
    fixes.append('⚠️  5. smooth_window not found')


# ─────────────────────────────────────────────────────────────
# Write + verify
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
    print(f'   Text: {repr(e.text)}')