"""
viewer_state.py
Gesture-Controlled Medical Imaging Workstation

WorkstationState — singleton that carries all data between modes.
Mode — enum of the 6 operating modes.

Usage:
    from viewer_state import state, Mode
    state.active_mode = Mode.SEGMENT
    state.current_slice = 12
"""

from __future__ import annotations
from enum import Enum, auto
from typing import Optional
import numpy as np


class Mode(Enum):
    DICOM     = auto()   # Load, scroll, zoom, pan, windowing
    FILTER    = auto()   # Denoise, sharpen, CLAHE, normalize
    SEGMENT   = auto()   # Organ segmentation overlay
    REGISTER  = auto()   # CT-MRI alignment (SimpleITK)
    PATH_PLAN = auto()   # Needle path planning (organ/rib-avoiding)
    MEASURE   = auto()   # Distance, area, HU measurement
    REVIEW    = auto()   # Read-only presentation mode


MODE_LABELS = {
    Mode.DICOM    : "DICOM",
    Mode.FILTER   : "Filter",
    Mode.SEGMENT  : "Segment",
    Mode.REGISTER : "Register",
    Mode.PATH_PLAN: "Path Plan",
    Mode.MEASURE  : "Measure",
    Mode.REVIEW   : "Review",
}

MODE_ICONS = {
    Mode.DICOM    : "📂",
    Mode.FILTER   : "🎛",
    Mode.SEGMENT  : "🤖",
    Mode.REGISTER : "🔗",
    Mode.PATH_PLAN: "🎯",
    Mode.MEASURE  : "📏",
    Mode.REVIEW   : "👁",
}


class WorkstationState:
    """
    Singleton state object shared across all modules.

    Data slots:
        raw_volume      — original DICOM pixel array (Z, H, W) float32
        filtered_volume — after PreprocessingEngine (Z, H, W) uint8
        seg_mask        — TotalSegmentator label array (Z, H, W) int32
        reg_volume      — registered secondary volume (Z, H, W) float32
        dicom_path      — path to loaded DICOM file or folder
        modality        — "CT", "MR", "PT" etc.
        metadata        — dict of patient/study info

    View state (persists across mode switches):
        active_mode     — current Mode enum value
        current_slice   — int, current Z index
        zoom            — float, 1.0 = no zoom
        pan_x, pan_y    — int, pixel offset from centre
        wl, ww          — float, window level / width
        opacity         — float 0–1, overlay opacity
        show_overlay    — bool, whether seg overlay is visible
        view_locked     — bool, lock navigation

    Filter state:
        active_filter   — str, name of active preprocessing filter
        filter_strength — float 0.1–2.0
    """

    _instance: Optional["WorkstationState"] = None

    def __new__(cls) -> "WorkstationState":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        # ── Volume data ────────────────────────
        self.raw_volume:      Optional[np.ndarray] = None
        self.filtered_volume: Optional[np.ndarray] = None
        self.seg_mask:        Optional[np.ndarray] = None
        self.reg_volume:      Optional[np.ndarray] = None
        self.dicom_path:      Optional[str]        = None
        self.modality:        str                  = "CT"
        self.metadata:        dict                 = {}
        # SimpleITK image matching raw_volume's array layout (spacing/origin/
        # direction). Needed for any physical-coordinate math (registration
        # crops, needle path planning). May have identity geometry if the
        # volume was loaded from a single 2-D DICOM file.
        self.sitk_image = None

        # ── Registration state (Register mode) ─
        self.reg_ct_arr:  Optional[np.ndarray] = None   # cropped fixed CT
        self.reg_mri_arr: Optional[np.ndarray] = None   # cropped registered MRI
        self.reg_ct_sitk  = None                         # sitk image for reg_ct_arr geometry
        self.reg_metrics:  dict = {}
        self.reg_out_dir:  Optional[str] = None
        self.view_mode:    int = 0        # 0=CT, 1=MRI Registered, 2=Overlay

        # ── Path planning state (Path Plan mode) ─
        self.tumor_point_mm  = None   # (x, y, z) physical mm, or None
        self.tumor_voxel     = None   # (slice, row, col) for drawing/markers
        self.path_candidates: list = []   # all collision-free candidates
        self.path_ranked:     list = []   # top-K shortest, collision-free
        self.selected_path_idx: Optional[int] = None

        # ── Mode ───────────────────────────────
        self.active_mode: Mode = Mode.DICOM

        # ── View state ─────────────────────────
        self.current_slice: int   = 0
        self.zoom:          float = 1.0
        self.pan_x:         int   = 0
        self.pan_y:         int   = 0
        self.wl:            float = 40.0
        self.ww:            float = 400.0
        self.opacity:       float = 0.45
        self.show_overlay:  bool  = False
        self.view_locked:   bool  = False

        # ── Filter state ───────────────────────
        self.active_filter:   str   = "None (Original)"
        self.filter_strength: float = 1.0

        # ── Measure state ──────────────────────
        self.measure_points: list = []   # list of (x, y) pixel coords

    # ── Convenience helpers ────────────────────
    @property
    def has_volume(self) -> bool:
        return self.raw_volume is not None

    @property
    def has_segmentation(self) -> bool:
        return self.seg_mask is not None

    @property
    def num_slices(self) -> int:
        if self.raw_volume is None:
            return 0
        return self.raw_volume.shape[0]

    @property
    def current_raw_slice(self) -> Optional[np.ndarray]:
        if self.raw_volume is None:
            return None
        return self.raw_volume[self.current_slice]

    @property
    def current_mask_slice(self) -> Optional[np.ndarray]:
        if self.seg_mask is None or self.raw_volume is None:
            return None
        mz  = self.seg_mask.shape[0]
        vz  = self.raw_volume.shape[0]
        idx = min(int(self.current_slice * mz / vz), mz - 1)
        return self.seg_mask[idx]

    def reset_view(self):
        """Reset zoom, pan, lock. Keep slice, WL/WW."""
        self.zoom        = 1.0
        self.pan_x       = 0
        self.pan_y       = 0
        self.view_locked = False

    def reset_all(self):
        """Full reset — keeps loaded volume."""
        self.reset_view()
        if self.raw_volume is not None:
            self.current_slice = self.raw_volume.shape[0] // 2
        self.show_overlay = False

    def zoom_in(self, delta: float = 0.1):
        if not self.view_locked:
            self.zoom = min(4.0, self.zoom + delta)

    def zoom_out(self, delta: float = 0.1):
        if not self.view_locked:
            self.zoom = max(1.0, self.zoom - delta)

    def pan(self, dx: int, dy: int):
        if self.view_locked or self.raw_volume is None:
            return
        h, w = self.raw_volume[self.current_slice].shape
        max_x = int(w * (1 - 1 / max(self.zoom, 1.0)))
        max_y = int(h * (1 - 1 / max(self.zoom, 1.0)))
        self.pan_x = max(-max_x, min(max_x, self.pan_x + dx))
        self.pan_y = max(-max_y, min(max_y, self.pan_y + dy))

    def next_slice(self) -> bool:
        if self.view_locked or self.raw_volume is None:
            return False
        if self.current_slice < self.num_slices - 1:
            self.current_slice += 1
            return True
        return False

    def prev_slice(self) -> bool:
        if self.view_locked or self.raw_volume is None:
            return False
        if self.current_slice > 0:
            self.current_slice -= 1
            return True
        return False

    def adjust_wl(self, delta: float):
        if not self.view_locked:
            self.wl = max(-1000, min(3000, self.wl + delta))

    def adjust_ww(self, delta: float):
        if not self.view_locked:
            self.ww = max(1.0, min(4000, self.ww + delta))

    def toggle_lock(self):
        self.view_locked = not self.view_locked

    def toggle_overlay(self):
        if self.has_segmentation:
            self.show_overlay = not self.show_overlay


# ── Module-level singleton ─────────────────────
state = WorkstationState()
