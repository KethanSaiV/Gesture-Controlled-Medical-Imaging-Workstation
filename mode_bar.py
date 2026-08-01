"""
mode_bar.py
Gesture-Controlled Medical Imaging Workstation

ModeBar — persistent 32px horizontal strip at the top of the window.
Shows all 6 modes as clickable buttons, highlights the active one.
Emits mode_changed(new_mode) when user clicks a different mode.
"""

from __future__ import annotations
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from viewer_state import Mode, MODE_LABELS, MODE_ICONS, state


class ModeBar(QWidget):
    mode_changed = pyqtSignal(object)   # emits Mode enum value

    ACTIVE_STYLE = """
        QPushButton {
            background: #1a3a5a;
            color: #3af;
            border: 1px solid #3af;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 11px;
            font-weight: bold;
        }
    """
    INACTIVE_STYLE = """
        QPushButton {
            background: #1e1e1e;
            color: #666;
            border: 1px solid #2a2a2a;
            border-radius: 4px;
            padding: 3px 10px;
            font-size: 11px;
        }
        QPushButton:hover {
            background: #252525;
            color: #aaa;
            border-color: #3af;
        }
    """
    LOCKED_SUFFIX = " 🔒"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet("background: #111; border-bottom: 1px solid #222;")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 2, 8, 2)
        layout.setSpacing(4)

        # App title
        title = QLabel("✋ MedGesture")
        title.setStyleSheet("color: #3af; font-size: 12px; font-weight: bold;")
        title.setFixedWidth(120)
        layout.addWidget(title)

        layout.addStretch()

        # Mode buttons
        self._buttons: dict[Mode, QPushButton] = {}
        for mode in Mode:
            icon  = MODE_ICONS[mode]
            label = MODE_LABELS[mode]
            btn   = QPushButton(f"{icon}  {label}")
            btn.setFixedHeight(26)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda _, m=mode: self._on_click(m))
            self._buttons[mode] = btn
            layout.addWidget(btn)

        layout.addStretch()

        # Lock indicator
        self.lbl_lock = QLabel("🔓")
        self.lbl_lock.setStyleSheet("color: #444; font-size: 12px;")
        self.lbl_lock.setFixedWidth(24)
        layout.addWidget(self.lbl_lock)

        self._refresh()

    def _on_click(self, mode: Mode):
        if mode == state.active_mode:
            return
        state.active_mode = mode
        self._refresh()
        self.mode_changed.emit(mode)

    def _refresh(self):
        """Update button styles to reflect current active mode and lock."""
        for mode, btn in self._buttons.items():
            if mode == state.active_mode:
                btn.setStyleSheet(self.ACTIVE_STYLE)
            else:
                btn.setStyleSheet(self.INACTIVE_STYLE)

        if state.view_locked:
            self.lbl_lock.setText("🔒")
            self.lbl_lock.setStyleSheet("color: #f33; font-size: 12px;")
        else:
            self.lbl_lock.setText("🔓")
            self.lbl_lock.setStyleSheet("color: #444; font-size: 12px;")

    def update_lock(self):
        """Call this when state.view_locked changes."""
        self._refresh()

    def set_mode(self, mode: Mode):
        """Programmatically set mode (e.g. from gesture dispatcher)."""
        state.active_mode = mode
        self._refresh()
