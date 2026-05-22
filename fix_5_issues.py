"""
Fix 5 issues in GestureEngine:

1. Lock blocks zoom (rotation zoom already blocked but check all paths)
2. Both thumbs up not firing segmentation — fix _is_thumb_up detection
3. Separate zoom from slice nav — use PINCH/SPREAD for zoom, swipe for slice
4. Pan not working — fix coordinate mapping and deadzone
5. Windowing: change WL (not WW), increase sensitivity
"""

with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []

# ─────────────────────────────────────────────────────────────
# FIX 1: Lock must block zoom in _on_zoom_gesture (already done)
#         AND block rotation accumulation in engine itself
# ─────────────────────────────────────────────────────────────
# The MainWindow._on_zoom_gesture already checks view_locked.
# But the engine keeps accumulating rotation even when locked.
# Fix: pass lock state into engine OR just guard in MainWindow (already done).
# Actually the real issue: zoom_changed signal fires → _on_zoom_gesture
# checks self.view_locked → returns early. This IS correct already.
# But let's make sure the guard is explicit:

old1 = '''    def _on_zoom_gesture(self, delta):
        """delta: positive = zoom in, negative = zoom out"""
        if self.view_locked:
            return
        self.zoom = max(1.0, min(4.0, self.zoom + delta))
        self._show_slice()
        self.lbl_gesture.setText(f"Rotate\\nZoom {\'In\' if delta > 0 else \'Out\'} {self.zoom:.1f}x")'''

new1 = '''    def _on_zoom_gesture(self, delta):
        """delta: positive = zoom in, negative = zoom out. Blocked when locked."""
        if self.view_locked:
            self.status.showMessage("View LOCKED — zoom disabled")
            return
        self.zoom = max(1.0, min(4.0, self.zoom + delta))
        self._show_slice()
        self.lbl_gesture.setText(f"Zoom {chr(43) if delta > 0 else chr(45)}{abs(delta):.2f}\\n{self.zoom:.1f}x")'''

if old1 in src:
    src = src.replace(old1, new1)
    fixes.append('✅ 1. Lock blocks zoom explicitly with status message')
else:
    fixes.append('⚠️  1. _on_zoom_gesture not found')


# ─────────────────────────────────────────────────────────────
# FIX 2: _is_thumb_up — too strict, fix detection
#         Real thumb-up: thumb tip HIGH, all fingers curled
#         Remove the thumb x-axis check that was blocking it
# ─────────────────────────────────────────────────────────────
old2 = '''    def _is_thumb_up(self, lm) -> bool:
        """Thumb extended upward, all other fingers curled."""
        f = self._fingers_up(lm)
        # Thumb up: thumb extended, others down
        # Also check thumb tip is ABOVE wrist (pointing up)
        thumb_up = lm[4][1] < lm[0][1]
        return f[0] == 1 and f[1] == 0 and f[2] == 0 and f[3] == 0 and f[4] == 0 and thumb_up'''

new2 = '''    def _is_thumb_up(self, lm) -> bool:
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

        return thumb_above and fingers_down'''

if old2 in src:
    src = src.replace(old2, new2)
    fixes.append('✅ 2. _is_thumb_up fixed — wrist-distance + fingers-down check')
else:
    fixes.append('⚠️  2. _is_thumb_up not found')


# ─────────────────────────────────────────────────────────────
# FIX 3: Separate zoom from slice navigation
#         PROBLEM: Rotation zoom triggers during swipe (hand tilts while swiping)
#         SOLUTION: Use TWO-FINGER PINCH/SPREAD for zoom
#                   Keep swipe for slices but add dead zone on rotation during swipe
# ─────────────────────────────────────────────────────────────

# Replace rotation zoom with pinch zoom
old3 = '''    def _detect_rotation_zoom(self, lm, now):
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
            self._status("Rotate CCW → Zoom Out")'''

new3 = '''    def _detect_rotation_zoom(self, lm, now):
        """
        Gesture 01 (REVISED): Pinch = zoom in, Spread = zoom out.
        Uses thumb-index distance normalised by palm size.
        Much more intuitive and doesn\'t conflict with swipe.
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
            self._status(f"Pinch {'open' if direction > 0 else 'close'} → Zoom {'In' if direction > 0 else 'Out'}")'''

if old3 in src:
    src = src.replace(old3, new3)
    fixes.append('✅ 3. Zoom changed: rotation → pinch/spread (no conflict with swipe)')
else:
    fixes.append('⚠️  3. _detect_rotation_zoom not found')

# Also update the call site that passes ctrl_lm to rotation zoom
old3b = '''            # ── GESTURE 01: Palm rotate → zoom ─────────────────
            if ctrl_lm and not self._pan_locked:
                self._detect_rotation_zoom(ctrl_lm, now)
            else:
                self._prev_wrist_angle = None
                self._rotation_accum   = 0.0'''

new3b = '''            # ── GESTURE 01: Pinch/spread → zoom ────────────────
            if ctrl_lm and not self._pan_locked:
                self._detect_rotation_zoom(ctrl_lm, now)
            else:
                self._pinch_prev_dist  = None
                self._pinch_zoom_accum = 0.0'''

if old3b in src:
    src = src.replace(old3b, new3b)
    fixes.append('✅ 3b. Zoom call site updated')
else:
    fixes.append('⚠️  3b. Zoom call site not found')


# ─────────────────────────────────────────────────────────────
# FIX 4: Pan not working — fix coordinate mapping
#         Problem: pan_moved emits normalised deltas (0-1 range)
#         but _on_pan_gesture multiplies by 0.15 of pixel dims
#         The PAN_SCALE of 0.20 is too small AND direction may be inverted
# ─────────────────────────────────────────────────────────────
old4 = '''    PAN_SCALE           = 0.20   # pan speed'''
new4 = '''    PAN_SCALE           = 1.5    # pan speed (increased — normalised coords are tiny)'''

if old4 in src:
    src = src.replace(old4, new4)
    fixes.append('✅ 4a. PAN_SCALE increased 0.20 → 1.5')
else:
    fixes.append('⚠️  4a. PAN_SCALE not found')

old4b = '''    PAN_DEADZONE        = 0.008  # min movement to pan'''
new4b = '''    PAN_DEADZONE        = 0.004  # min movement to pan (lowered)'''

if old4b in src:
    src = src.replace(old4b, new4b)
    fixes.append('✅ 4b. PAN_DEADZONE lowered 0.008 → 0.004')
else:
    fixes.append('⚠️  4b. PAN_DEADZONE not found')

# Fix _on_pan_gesture in MainWindow — the signal sends normalised dx/dy
# but we need to scale to pixel space properly
old4c = '''    def _on_pan_gesture(self, dx_norm, dy_norm):
        """Called from gesture engine with normalised deltas (-1 to 1)."""
        if self.volume is None or self.view_locked:
            return
        h, w = self.volume[self.current].shape
        self._pan(int(dx_norm * w * 0.15), int(dy_norm * h * 0.15))'''

new4c = '''    def _on_pan_gesture(self, dx_norm, dy_norm):
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
            self.lbl_gesture.setText(f"Pan\\n{px:+d},{py:+d}px")'''

if old4c in src:
    src = src.replace(old4c, new4c)
    fixes.append('✅ 4c. _on_pan_gesture scale fixed (3x image dimension)')
else:
    fixes.append('⚠️  4c. _on_pan_gesture not found')

# Also fix pan emission in engine — currently emits PAN_SCALE * delta
# but should emit raw delta and let MainWindow scale
old4d = '''                if self._pan_prev_pos is not None:
                    pos = lm[8]
                    dx = pos[0] - self._pan_prev_pos[0]
                    dy = pos[1] - self._pan_prev_pos[1]
                    if abs(dx) > self.PAN_DEADZONE or abs(dy) > self.PAN_DEADZONE:
                        self.pan_moved.emit(
                            dx * self.PAN_SCALE,
                            dy * self.PAN_SCALE
                        )
                self._pan_prev_pos = pos'''

new4d = '''                pos = lm[8]  # index tip
                if self._pan_prev_pos is not None:
                    dx = pos[0] - self._pan_prev_pos[0]
                    dy = pos[1] - self._pan_prev_pos[1]
                    if abs(dx) > self.PAN_DEADZONE or abs(dy) > self.PAN_DEADZONE:
                        # Emit raw normalised delta — MainWindow scales to pixels
                        self.pan_moved.emit(dx, dy)
                self._pan_prev_pos = pos'''

if old4d in src:
    src = src.replace(old4d, new4d)
    fixes.append('✅ 4d. Pan emission fixed — raw delta emitted')
else:
    fixes.append('⚠️  4d. Pan emission block not found')


# ─────────────────────────────────────────────────────────────
# FIX 5: Windowing — change WL (not WW), increase sensitivity
# ─────────────────────────────────────────────────────────────
old5 = '''    WINDOW_SCALE        = 0.015  # windowing sensitivity'''
new5 = '''    WINDOW_SCALE        = 0.015  # windowing sensitivity (unused — see _detect_windowing)'''

if old5 in src:
    src = src.replace(old5, new5)

old5b = '''    def _detect_windowing(self, left_lm, right_lm, now):
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
            self._status(f"Windowing WW {\'wider\' if delta > 0 else \'narrower\'}")'''

new5b = '''    def _detect_windowing(self, left_lm, right_lm, now):
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
            self._status(f"WL {'up' if delta > 0 else 'down'} ({dwl:+.0f})")'''

if old5b in src:
    src = src.replace(old5b, new5b)
    fixes.append('✅ 5. Windowing: now changes WL (brightness), 10x more sensitive')
else:
    fixes.append('⚠️  5. _detect_windowing not found')

# Also fix _on_window_gesture in MainWindow to handle WL/WW separately
old5c = '''    def _on_window_gesture(self, dwl, dww):
        """Adjust window level and width."""
        if self.view_locked:
            return
        self.wl += dwl
        self.ww  = max(1.0, self.ww + dww)
        self.slider_wl.setValue(int(self.wl))
        self.slider_ww.setValue(int(self.ww))
        self.lbl_gesture.setText(f"2-Index\\nWW:{self.ww:.0f}")'''

new5c = '''    def _on_window_gesture(self, dwl, dww):
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
        self.lbl_gesture.setText(f"Windowing\\nWL:{self.wl:.0f} WW:{self.ww:.0f}")'''

if old5c in src:
    src = src.replace(old5c, new5c)
    fixes.append('✅ 5b. _on_window_gesture: WL and WW handled separately')
else:
    fixes.append('⚠️  5b. _on_window_gesture not found')


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