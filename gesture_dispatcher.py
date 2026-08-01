"""
gesture_dispatcher.py
Gesture-Controlled Medical Imaging Workstation

GestureDispatcher — receives gesture tokens from GestureEngine,
looks up the action for the current mode, and executes it.
"""

from __future__ import annotations
from PyQt5.QtCore import QObject, pyqtSignal
from viewer_state import state, Mode


class GestureDispatcher(QObject):
    gesture_executed = pyqtSignal(str, str, str)   # token, mode, action_name
    pan_executed     = pyqtSignal(int, int)         # dx_px, dy_px
    state_changed    = pyqtSignal()

    PAN_SCALE_X = 3.0
    PAN_SCALE_Y = 3.0

    def on_gesture(self, token: str):
        if state.view_locked and token not in ("PEACE",):
            return

        action_name = self._dispatch(token)
        if action_name:
            self.gesture_executed.emit(token, state.active_mode.name, action_name)
            self.state_changed.emit()

    def _dispatch(self, token: str) -> str:
        """Execute action for current mode + token. Returns action name or empty string."""
        mode = state.active_mode

        # ── Universal navigation (all modes) ──────────────
        if token == "SWIPE_UP":
            state.prev_slice(); return "prev_slice"
        if token == "SWIPE_DOWN":
            state.next_slice(); return "next_slice"
        if token == "OPEN_PALM":
            state.reset_all(); return "reset"

        # ── DICOM mode ────────────────────────────────────
        if mode == Mode.DICOM:
            if token == "PINCH":
                state.zoom_in(0.12); return "zoom_in"
            if token == "FIST":
                state.zoom_out(0.12); return "zoom_out"
            if token == "PEACE":
                state.toggle_lock(); return "toggle_lock"

        # ── FILTER mode ───────────────────────────────────
        elif mode == Mode.FILTER:
            if token == "PINCH":
                state.filter_strength = min(2.0, state.filter_strength + 0.2)
                return "strength_up"
            if token == "FIST":
                state.filter_strength = max(0.1, state.filter_strength - 0.2)
                return "strength_down"
            if token == "PEACE":
                state.toggle_lock(); return "toggle_lock"

        # ── SEGMENT mode ──────────────────────────────────
        elif mode == Mode.SEGMENT:
            if token == "PINCH":
                state.zoom_in(0.12); return "zoom_in"
            if token == "FIST":
                state.zoom_out(0.12); return "zoom_out"
            if token == "PEACE":
                state.toggle_lock(); return "toggle_lock"

        # ── REGISTER mode ─────────────────────────────────
        elif mode == Mode.REGISTER:
            if token == "PINCH":
                state.zoom_in(0.12); return "zoom_in"
            if token == "FIST":
                state.zoom_out(0.12); return "zoom_out"
            if token == "PEACE":
                state.toggle_lock(); return "toggle_lock"

        # ── PATH PLAN mode ────────────────────────────────
        elif mode == Mode.PATH_PLAN:
            if token == "PINCH":
                state.zoom_in(0.12); return "zoom_in"
            if token == "FIST":
                state.zoom_out(0.12); return "zoom_out"
            if token == "PEACE":
                state.toggle_lock(); return "toggle_lock"

        # ── MEASURE / REVIEW ──────────────────────────────
        else:
            if token == "PINCH":
                state.zoom_in(0.12); return "zoom_in"
            if token == "FIST":
                state.zoom_out(0.12); return "zoom_out"

        return ""

    def on_pan(self, dx_norm: float, dy_norm: float):
        if state.view_locked or state.raw_volume is None:
            return
        h, w  = state.raw_volume[state.current_slice].shape
        dx_px = int(dx_norm * w * self.PAN_SCALE_X)
        dy_px = int(dy_norm * h * self.PAN_SCALE_Y)
        if abs(dx_px) > 0 or abs(dy_px) > 0:
            state.pan(dx_px, dy_px)
            self.pan_executed.emit(dx_px, dy_px)
            self.state_changed.emit()

    def on_wl_gesture(self, dwl: float, dww: float = 0.0):
        if state.view_locked:
            return
        if dwl != 0:
            state.adjust_wl(dwl)
        if dww != 0:
            state.adjust_ww(dww)
        self.state_changed.emit()
