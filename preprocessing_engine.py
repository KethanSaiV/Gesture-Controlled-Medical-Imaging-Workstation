"""
preprocessing_engine.py
Gesture-Controlled Medical Imaging Workstation

PreprocessingEngine — applies image filters to uint8 grayscale slices.
Original volume is never modified; filter is applied at display time.
"""

from __future__ import annotations
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


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
        Returns uint8 grayscale — input is never modified.
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

    def next_filter(self):
        idx = self.FILTERS.index(self.active_filter)
        self.active_filter = self.FILTERS[(idx + 1) % len(self.FILTERS)]

    def prev_filter(self):
        idx = self.FILTERS.index(self.active_filter)
        self.active_filter = self.FILTERS[(idx - 1) % len(self.FILTERS)]

    def reset(self):
        self.active_filter = "None (Original)"
        self.strength      = 1.0
