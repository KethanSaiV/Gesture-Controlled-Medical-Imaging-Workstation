"""
Phase 4 - AI Segmentation Integration
Gesture-Controlled Medical Imaging Workstation

Segmentation backend: TotalSegmentator v2
  CT  → task="total"     (117 structures, ~2.4GB model)
  MRI → task="total_mr"  ( 50 structures, ~0.9GB model)
  PET → treated as CT    (PET/CT scanners share anatomy)

All other features (gestures, overlay, legend, UI) unchanged.
"""

import sys
import os
import cv2
import numpy as np
import pydicom
import SimpleITK as sitk
import time
import threading
import tempfile
import shutil

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QLabel, QSlider, QFileDialog,
    QStatusBar, QGroupBox, QGridLayout, QProgressBar,
    QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor


# ──────────────────────────────────────────────
#  Organ Colour Map  (TotalSegmentator label IDs)
#
#  CT task  "total"    → label IDs 1-117
#  MRI task "total_mr" → label IDs 1-50
#
#  We use a shared colour map for both.
#  Labels that don't exist in a given scan simply
#  never appear in the mask → no harm done.
# ──────────────────────────────────────────────
ORGAN_COLOURS = {
    # ── Core abdominal organs ──────────────────
    1:  ("Spleen",              (0,   255,  80)),
    2:  ("Kidney Right",        (0,   120, 255)),
    3:  ("Kidney Left",         (0,   200, 255)),
    4:  ("Gallbladder",         (0,   255, 220)),
    5:  ("Liver",               (0,    60, 255)),
    6:  ("Stomach",             (255, 180,   0)),
    7:  ("Pancreas",            (0,   255, 255)),
    8:  ("Adrenal Right",       (255, 255,   0)),
    9:  ("Adrenal Left",        (200, 255,   0)),
    # ── Vascular ──────────────────────────────
    10: ("Aorta",               (0,     0, 255)),
    11: ("Inferior Vena Cava",  (180,   0, 255)),
    12: ("Portal/Splenic Vein", (255,   0, 180)),
    13: ("Iliac Artery Right",  (0,    80, 200)),
    14: ("Iliac Artery Left",   (0,   100, 180)),
    15: ("Iliac Vein Right",    (140,   0, 200)),
    16: ("Iliac Vein Left",     (120,   0, 180)),
    # ── Thorax ────────────────────────────────
    17: ("Lung Right",          (100, 180, 255)),
    18: ("Lung Left",           (60,  140, 255)),
    19: ("Heart",               (0,    80, 255)),
    20: ("Trachea",             (200, 220, 255)),
    21: ("Esophagus",           (255, 160, 100)),
    # ── Pelvis ────────────────────────────────
    22: ("Urinary Bladder",     (255, 100, 100)),
    23: ("Prostate/Uterus",     (255, 150, 200)),
    24: ("Rectum",              (180, 255, 100)),
    25: ("Colon",               (140, 200,  80)),
    26: ("Duodenum",            (255, 200, 100)),
    27: ("Small Bowel",         (120, 180,  60)),
    # ── Bones ─────────────────────────────────
    28: ("Sacrum",              (255, 240, 180)),
    29: ("Femur Right",         (200, 150, 255)),
    30: ("Femur Left",          (150, 100, 255)),
    31: ("Hip Right",           (220, 170, 255)),
    32: ("Hip Left",            (170, 120, 255)),
    33: ("Spine",               (255, 220, 150)),
    # ── Muscle / body comp ────────────────────
    34: ("Gluteus Max Right",   (180, 255, 180)),
    35: ("Gluteus Max Left",    (140, 220, 140)),
    36: ("Iliopsoas Right",     (255, 180, 140)),
    37: ("Iliopsoas Left",      (220, 140, 100)),
    # ── Brain (MRI) ───────────────────────────
    38: ("Brain",               (200, 220, 255)),
    39: ("Brainstem",           (160, 180, 255)),
    # ── Misc ──────────────────────────────────
    40: ("Thyroid Gland",       (100, 255, 200)),
    41: ("Spinal Cord",         (240, 240, 100)),
    42: ("Sternum",             (255, 230, 200)),
    43: ("Clavicula Right",     (210, 200, 255)),
    44: ("Clavicula Left",      (180, 170, 255)),
    45: ("Scapula Right",       (230, 210, 255)),
    46: ("Scapula Left",        (200, 180, 255)),
    47: ("Humerus Right",       (255, 210, 210)),
    48: ("Humerus Left",        (220, 180, 180)),
    49: ("Rib Cage",            (200, 200, 200)),
    50: ("Skull",               (240, 240, 240)),
}

# ──────────────────────────────────────────────
#  TotalSegmentator task selection
# ──────────────────────────────────────────────
def _pick_task(modality: str) -> str:
    """
    Returns the correct TotalSegmentator task string.

    CT  → "total"     117 structures
    MRI → "total_mr"   50 structures
    PET → "total"      PET/CT shares CT anatomy model
    """
    m = modality.upper()
    if m in ("MR", "MRI"):
        return "total_mr"
    return "total"   # CT, PT (PET), NM, unknown → CT model


GESTURE_COOLDOWN = 1.5

GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom In",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Zoom Out",
    "POINT_UP"   : "Scroll Up",
    "PEACE"      : "Scroll Down",
}

COL_GREEN  = (0, 255, 120)
COL_YELLOW = (0, 220, 255)
COL_RED    = (0, 80,  255)
COL_WHITE  = (255, 255, 255)
COL_BLUE   = (255, 180, 0)
COL_ORANGE = (0, 165, 255)


# ──────────────────────────────────────────────
#  DICOM Loader
# ──────────────────────────────────────────────
class DicomLoader(QThread):
    loaded = pyqtSignal(object, object, dict)
    error  = pyqtSignal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            if os.path.isfile(self.path):
                ds  = pydicom.dcmread(self.path)
                arr = ds.pixel_array.astype(np.float32)
                if arr.ndim == 2:
                    arr = arr[np.newaxis, ...]
                sitk_img = sitk.GetImageFromArray(arr)
                self.loaded.emit(arr, sitk_img, self._meta(ds))

            elif os.path.isdir(self.path):
                reader     = sitk.ImageSeriesReader()
                series_ids = reader.GetGDCMSeriesIDs(self.path)
                if not series_ids:
                    # Try one level deeper (handles vishwa/3/00020001/... structure)
                    for sub in os.listdir(self.path):
                        sub_path = os.path.join(self.path, sub)
                        if os.path.isdir(sub_path):
                            series_ids = reader.GetGDCMSeriesIDs(sub_path)
                            if series_ids:
                                self.path = sub_path
                                break
                if not series_ids:
                    self.error.emit("No DICOM series found in folder.")
                    return
                files = reader.GetGDCMSeriesFileNames(self.path, series_ids[0])
                reader.SetFileNames(files)
                sitk_img = reader.Execute()
                arr      = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
                ds       = pydicom.dcmread(files[0])
                self.loaded.emit(arr, sitk_img, self._meta(ds))
            else:
                self.error.emit("Invalid path.")
        except Exception as e:
            self.error.emit(str(e))

    @staticmethod
    def _meta(ds):
        def s(tag, d="N/A"): return str(getattr(ds, tag, d))
        return {
            "Patient"    : s("PatientName"),
            "Modality"   : s("Modality"),
            "Study Date" : s("StudyDate"),
            "Rows"       : s("Rows"),
            "Columns"    : s("Columns"),
            "Institution": s("InstitutionName"),
        }


# ──────────────────────────────────────────────
#  TotalSegmentator Worker  (replaces MOOSEZ)
# ──────────────────────────────────────────────
class SegmentationWorker(QThread):
    """
    Runs TotalSegmentator in a background thread.

    Flow:
      1. Convert DICOM folder → NIfTI (SimpleITK)
      2. Call totalsegmentator(input_nii, output_dir, task, device, fast)
      3. TotalSegmentator writes one .nii.gz per organ into output_dir
      4. We merge all organ masks into a single integer label volume
      5. Emit the label array (Z, H, W) int32 back to the main thread
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(object)   # label array (Z, H, W) int32
    error    = pyqtSignal(str)

    # Label name → ID  (used to rebuild a merged mask from per-organ files)
    # TotalSegmentator writes filenames like "liver.nii.gz", "spleen.nii.gz", etc.
    # We map those file stems to our ORGAN_COLOURS IDs.
    _NAME_TO_ID = {
        "spleen"              : 1,
        "kidney_right"        : 2,
        "kidney_left"         : 3,
        "gallbladder"         : 4,
        "liver"               : 5,
        "stomach"             : 6,
        "pancreas"            : 7,
        "adrenal_gland_right" : 8,
        "adrenal_gland_left"  : 9,
        "aorta"               : 10,
        "inferior_vena_cava"  : 11,
        "portal_vein_and_splenic_vein": 12,
        "iliac_artery_right"  : 13,
        "iliac_artery_left"   : 14,
        "iliac_vein_right"    : 15,
        "iliac_vein_left"     : 16,
        "lung_upper_lobe_right": 17,
        "lung_lower_lobe_right": 17,
        "lung_upper_lobe_left" : 18,
        "lung_lower_lobe_left" : 18,
        "lung_middle_lobe_right": 17,
        "heart"               : 19,
        "trachea"             : 20,
        "esophagus"           : 21,
        "urinary_bladder"     : 22,
        "prostate"            : 23,
        "uterus"              : 23,
        "rectum"              : 24,
        "colon"               : 25,
        "duodenum"            : 26,
        "small_bowel"         : 27,
        "sacrum"              : 28,
        "femur_right"         : 29,
        "femur_left"          : 30,
        "hip_right"           : 31,
        "hip_left"            : 32,
        "vertebrae_l1"        : 33,
        "vertebrae_l2"        : 33,
        "vertebrae_l3"        : 33,
        "vertebrae_l4"        : 33,
        "vertebrae_l5"        : 33,
        "vertebrae_t1"        : 33,
        "vertebrae_t2"        : 33,
        "vertebrae_t3"        : 33,
        "vertebrae_t4"        : 33,
        "vertebrae_t5"        : 33,
        "vertebrae_t6"        : 33,
        "vertebrae_t7"        : 33,
        "vertebrae_t8"        : 33,
        "vertebrae_t9"        : 33,
        "vertebrae_t10"       : 33,
        "vertebrae_t11"       : 33,
        "vertebrae_t12"       : 33,
        "vertebrae_c1"        : 33,
        "vertebrae_c2"        : 33,
        "vertebrae_c3"        : 33,
        "vertebrae_c4"        : 33,
        "vertebrae_c5"        : 33,
        "vertebrae_c6"        : 33,
        "vertebrae_c7"        : 33,
        "gluteus_maximus_right": 34,
        "gluteus_maximus_left" : 35,
        "iliopsoas_right"     : 36,
        "iliopsoas_left"      : 37,
        "brain"               : 38,
        "brainstem"           : 39,
        "thyroid_gland"       : 40,
        "spinal_cord"         : 41,
        "sternum"             : 42,
        "clavicula_right"     : 43,
        "clavicula_left"      : 44,
        "scapula_right"       : 45,
        "scapula_left"        : 46,
        "humerus_right"       : 47,
        "humerus_left"        : 48,
        "rib_left_1"          : 49,
        "rib_left_2"          : 49,
        "rib_left_3"          : 49,
        "rib_left_4"          : 49,
        "rib_left_5"          : 49,
        "rib_left_6"          : 49,
        "rib_left_7"          : 49,
        "rib_left_8"          : 49,
        "rib_left_9"          : 49,
        "rib_left_10"         : 49,
        "rib_left_11"         : 49,
        "rib_left_12"         : 49,
        "rib_right_1"         : 49,
        "rib_right_2"         : 49,
        "rib_right_3"         : 49,
        "rib_right_4"         : 49,
        "rib_right_5"         : 49,
        "rib_right_6"         : 49,
        "rib_right_7"         : 49,
        "rib_right_8"         : 49,
        "rib_right_9"         : 49,
        "rib_right_10"        : 49,
        "rib_right_11"        : 49,
        "rib_right_12"        : 49,
        "skull"               : 50,
    }

    def __init__(self, dicom_path, modality):
        super().__init__()
        self.dicom_path = dicom_path
        self.modality   = modality.upper()

    def run(self):
        tmp_dir = None
        try:
            import subprocess, sys

            task = _pick_task(self.modality)
            self.progress.emit(
                f"Modality: {self.modality}  →  Task: {task}\n"
                f"Converting DICOM to NIfTI..."
            )

            # ── Step 1: DICOM → NIfTI ───────────────
            tmp_dir = tempfile.mkdtemp(prefix="totalseg_")

            # Windows fix: TotalSegmentator's nnUNet backend passes the NIfTI
            # path to a multiprocessing worker that expects a DIRECTORY, causing
            # [WinError 267].  The fix is to put input.nii.gz inside its own
            # subfolder and pass that FOLDER — TS will find the .nii.gz inside.
            in_dir  = os.path.join(tmp_dir, "input")
            os.makedirs(in_dir, exist_ok=True)
            nii_in  = os.path.join(in_dir, "s01.nii.gz")   # TS expects s01 naming
            out_dir = os.path.join(tmp_dir, "segs")
            os.makedirs(out_dir, exist_ok=True)

            reader = sitk.ImageSeriesReader()
            if os.path.isdir(self.dicom_path):
                dicom_dir  = self.dicom_path
                series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
                if not series_ids:
                    for sub in os.listdir(dicom_dir):
                        sub_path = os.path.join(dicom_dir, sub)
                        if os.path.isdir(sub_path):
                            series_ids = reader.GetGDCMSeriesIDs(sub_path)
                            if series_ids:
                                dicom_dir = sub_path
                                break
                if not series_ids:
                    self.error.emit(
                        "No DICOM series found.\n"
                        "Please select the folder that directly contains the .dcm files."
                    )
                    return
                files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
            else:
                files = [self.dicom_path]

            reader.SetFileNames(files)
            sitk_img = reader.Execute()
            sitk.WriteImage(sitk_img, nii_in)
            self.progress.emit(f"NIfTI ready. Launching TotalSegmentator ({task})...")
            self.progress.emit("⏳ First run downloads model weights (~0.9–2.4 GB). Please wait...")

            # ── Step 2: Run TotalSegmentator as SUBPROCESS ──────────
            #
            # WHY SUBPROCESS: importing totalsegmentator into this process causes
            # PyTorch + nnUNet multiprocessing workers to conflict with the PyQt5
            # event loop on Windows → the app closes mid-run.
            # A subprocess is fully isolated and can't crash the GUI.
            #
            self.progress.emit(f"Launching TotalSegmentator ({task}) in subprocess...")
            self.progress.emit("⏳ Models already downloaded — estimating ~5-10 min on GPU")

            cmd = [
                sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator",
                "-i", nii_in,
                "-o", out_dir,
                "-ta", task,
                "--ml",
                "-d", "cpu",
                "--fast",  # 3mm resolution — much faster on CPU
            ]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                line = line.strip()
                if line and "clearcut" not in line and not line.startswith("E0000") and not line.startswith("W0000") and not line.startswith("==="):
                    self.progress.emit(line[:120])

            proc.wait()

            if proc.returncode != 0:
                self.error.emit(
                    f"TotalSegmentator exited with code {proc.returncode}.\n"
                    "If models are missing, run in terminal:\n"
                    f"  TotalSegmentator -i input.nii.gz -o out/ -ta {task} --ml"
                )
                return

            self.progress.emit("Loading segmentation results...")

            # ── Step 3: Load output ──────────────────
            # ml=True produces a single multilabel .nii.gz where each voxel value
            # is the TotalSegmentator class index.  We remap those to our
            # ORGAN_COLOURS IDs using _NAME_TO_ID via the class_map.
            #
            # TotalSegmentator multilabel file is named after the task, e.g.:
            #   total.nii.gz  /  total_mr.nii.gz
            # but sometimes just the first .nii.gz in out_dir.

            label_vol = None

            # Try multilabel file first (ml=True output)
            ml_candidates = [
                os.path.join(out_dir, f"{task}.nii.gz"),
                os.path.join(out_dir, "s01.nii.gz"),
            ]
            for candidate in ml_candidates:
                if os.path.exists(candidate):
                    img = sitk.ReadImage(candidate)
                    label_vol = sitk.GetArrayFromImage(img).astype(np.int32)
                    self.progress.emit(f"Loaded multilabel file: {os.path.basename(candidate)}")
                    break

            # Fallback: scan all .nii.gz in out_dir
            if label_vol is None:
                seg_files = [f for f in os.listdir(out_dir)
                             if f.endswith(".nii.gz") or f.endswith(".nii")]
                if not seg_files:
                    self.error.emit(
                        "TotalSegmentator produced no output files.\n"
                        f"Output dir: {out_dir}\n"
                        "Check GPU memory and that the input scan is valid."
                    )
                    return

                if len(seg_files) == 1:
                    # Single file → either multilabel or single organ
                    img = sitk.ReadImage(os.path.join(out_dir, seg_files[0]))
                    label_vol = sitk.GetArrayFromImage(img).astype(np.int32)
                else:
                    # Multiple per-organ binary masks → merge into one label volume
                    self.progress.emit(f"Merging {len(seg_files)} organ masks...")
                    for fname in sorted(seg_files):
                        stem     = fname.replace(".nii.gz", "").replace(".nii", "").lower()
                        label_id = self._NAME_TO_ID.get(stem)
                        if label_id is None:
                            continue
                        organ_img = sitk.ReadImage(os.path.join(out_dir, fname))
                        organ_arr = sitk.GetArrayFromImage(organ_img).astype(np.uint8)
                        if label_vol is None:
                            label_vol = np.zeros_like(organ_arr, dtype=np.int32)
                        label_vol[organ_arr > 0] = label_id

            if label_vol is None or label_vol.max() == 0:
                self.error.emit(
                    "Segmentation mask is empty — no organs detected.\n"
                    f"Task: {task} | Modality: {self.modality}\n"
                    "This can happen if the scan field-of-view doesn't cover "
                    "the structures the model expects (e.g. CT model on MRI)."
                )
                return

            self.progress.emit("Segmentation complete! ✅")
            self.finished.emit(label_vol)

        except ImportError:
            self.error.emit(
                "TotalSegmentator not found.\n"
                "Install with:  pip install TotalSegmentator"
            )
        except Exception as e:
            import traceback
            self.error.emit(f"Segmentation error:\n{str(e)}\n\n{traceback.format_exc()}")
        finally:
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ──────────────────────────────────────────────
#  Two-Hand Gesture Engine  (unchanged)
# ──────────────────────────────────────────────
class TwoHandGestureEngine(QThread):
    gesture_detected  = pyqtSignal(str, str)
    frame_ready       = pyqtSignal(np.ndarray)
    ai_mode_changed   = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.running           = True
        self.prev_pos_right    = []
        self.prev_pos_left     = []
        self.smooth_window     = 5
        self.last_time         = {}
        self._left_fist_held   = False
        self._left_fist_start  = 0

    def run(self):
        model_path = self._get_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        options    = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        prev_time      = time.time()
        ai_mode_active = False

        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)

            left_landmarks  = None
            right_landmarks = None

            if result.hand_landmarks and result.handedness:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    lm_list    = [(lm.x, lm.y) for lm in hand_lms]
                    handedness = result.handedness[i][0].category_name

                    self._draw_landmarks(frame, hand_lms, w, h,
                                         color=(0,200,100) if handedness=="Right" else (200,100,0))

                    if handedness == "Left":
                        left_landmarks = lm_list
                    else:
                        right_landmarks = lm_list

            left_gesture = None
            if left_landmarks:
                left_gesture = self._classify_static(left_landmarks)
                if left_gesture == "FIST":
                    if not self._left_fist_held:
                        self._left_fist_held  = True
                        self._left_fist_start = time.time()
                    held_dur = time.time() - self._left_fist_start
                    if held_dur > 1.0:
                        ai_mode_active = not ai_mode_active
                        self.ai_mode_changed.emit(ai_mode_active)
                        self._left_fist_held = False
                else:
                    self._left_fist_held = False

            if right_landmarks:
                gesture = self._classify_full(right_landmarks, "right")
                if gesture:
                    self.gesture_detected.emit(gesture, "right")

            now       = time.time()
            fps       = 1.0 / (now - prev_time + 1e-9)
            prev_time = now

            frame = self._draw_overlay(
                frame, left_landmarks, right_landmarks,
                left_gesture, ai_mode_active, fps, w, h
            )
            self.frame_ready.emit(frame.copy())

        cap.release()

    def stop(self):
        self.running = False

    def _draw_landmarks(self, frame, hand_lms, w, h, color):
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17),
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(frame, pts[tip], 7, COL_YELLOW, -1)

    def _draw_overlay(self, frame, left_lm, right_lm,
                      left_gesture, ai_mode, fps, w, h):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GREEN, 2)

        if ai_mode:
            cv2.rectangle(frame, (0, 0), (w, 4), COL_ORANGE, -1)
            cv2.putText(frame, "AI MODE ACTIVE", (w//2 - 80, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_ORANGE, 2)

        if left_lm:
            lx = int(left_lm[0][0] * w)
            ly = int(left_lm[0][1] * h)
            label = "LEFT (AI Lock)" if left_gesture == "FIST" else "LEFT"
            cv2.putText(frame, label, (lx, ly - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_ORANGE, 1)

        if right_lm:
            rx = int(right_lm[0][0] * w)
            ry = int(right_lm[0][1] * h)
            cv2.putText(frame, "RIGHT (Control)", (rx, ry - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COL_GREEN, 1)

        cv2.rectangle(frame, (0, h - 35), (w, h), (10, 10, 10), -1)
        if not left_lm and not right_lm:
            cv2.putText(frame, "Show hands to camera",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        elif not ai_mode:
            cv2.putText(frame, "Hold LEFT FIST 1 sec to activate AI",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_ORANGE, 1)
        else:
            cv2.putText(frame, "AI active | Use RIGHT hand to navigate",
                        (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_GREEN, 1)

        return frame

    def _classify_static(self, lm):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)
        if num_up == 0:
            return "FIST"
        if num_up == 5:
            return "OPEN_PALM"
        if fingers_up == [0, 1, 0, 0, 0]:
            return "POINT_UP"
        if fingers_up == [0, 1, 1, 0, 0]:
            return "PEACE"
        if self._is_pinch(lm):
            return "PINCH"
        return None

    def _classify_full(self, lm, hand_key):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)

        pos_list = self.prev_pos_right if hand_key == "right" else self.prev_pos_left
        pos_list.append(lm[0])
        if len(pos_list) > self.smooth_window:
            pos_list.pop(0)

        gesture = None

        if num_up == 5:
            gesture = "OPEN_PALM"
        elif num_up == 0:
            gesture = "FIST"
        elif fingers_up == [0, 1, 0, 0, 0]:
            gesture = "POINT_UP"
        elif fingers_up == [0, 1, 1, 0, 0]:
            gesture = "PEACE"
        elif self._is_pinch(lm):
            gesture = "PINCH"
        elif len(pos_list) >= self.smooth_window:
            dy = pos_list[-1][1] - pos_list[0][1]
            if dy < -0.06:
                gesture = "SWIPE_UP"
            elif dy > 0.06:
                gesture = "SWIPE_DOWN"

        if gesture:
            key  = f"{hand_key}_{gesture}"
            now  = time.time()
            last = self.last_time.get(key, 0)
            if now - last < GESTURE_COOLDOWN:
                gesture = None
            else:
                self.last_time[key] = now

        return gesture

    def _fingers_up(self, lm):
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        f = [1 if lm[tips[0]][0] > lm[pip[0]][0] else 0]
        for i in range(1, 5):
            f.append(1 if lm[tips[i]][1] < lm[pip[i]][1] else 0)
        return f

    def _is_pinch(self, lm):
        tx, ty = lm[4]
        ix, iy = lm[8]
        return np.hypot(tx - ix, ty - iy) < 0.025

    @staticmethod
    def _get_model():
        import urllib.request
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmark model...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
        return model_path


# ──────────────────────────────────────────────
#  Overlay Renderer  (unchanged)
# ──────────────────────────────────────────────
def apply_segmentation_overlay(scan_slice, mask_slice, opacity=0.45):
    bgr     = cv2.cvtColor(scan_slice, cv2.COLOR_GRAY2BGR)
    overlay = bgr.copy()

    for label, (name, colour) in ORGAN_COLOURS.items():
        mask = (mask_slice == label)
        if mask.any():
            overlay[mask] = colour

    result = cv2.addWeighted(overlay, opacity, bgr, 1 - opacity, 0)
    return result


# ──────────────────────────────────────────────
#  Legend Widget  (unchanged)
# ──────────────────────────────────────────────
class LegendWidget(QWidget):
    def __init__(self, active_labels=None):
        super().__init__()
        self.active_labels = active_labels or []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)

        title = QLabel("Organ Legend")
        title.setStyleSheet("color:#3af; font-size:11px; font-weight:bold;")
        layout.addWidget(title)

        for label, (name, bgr) in ORGAN_COLOURS.items():
            if self.active_labels and label not in self.active_labels:
                continue
            r, g, b = bgr[2], bgr[1], bgr[0]
            row    = QHBoxLayout()
            swatch = QLabel("  ")
            swatch.setFixedSize(16, 12)
            swatch.setStyleSheet(f"background:rgb({r},{g},{b}); border-radius:2px;")
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("color:#ccc; font-size:10px;")
            row.addWidget(swatch)
            row.addWidget(name_lbl)
            row.addStretch()
            layout.addLayout(row)

        layout.addStretch()


# ──────────────────────────────────────────────
#  Main Window  (unchanged except status text)
# ──────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.volume       = None
        self.sitk_image   = None
        self.dicom_path   = None
        self.modality     = "CT"
        self.current      = 0
        self.ww           = 400.0
        self.wl           = 40.0
        self.zoom         = 1.0
        self.ai_mode      = False
        self.seg_mask     = None
        self.show_overlay = False
        self.opacity      = 0.45

        self._setup_ui()
        self._start_gesture_engine()

    def _setup_ui(self):
        self.setWindowTitle("Gesture-Controlled Medical Imaging Workstation — Phase 4 (TotalSegmentator)")
        self.setMinimumSize(1500, 800)
        self._apply_theme()

        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(8)
        root.setContentsMargins(8, 8, 8, 8)

        # ── LEFT PANEL ─────────────────────────
        left = QVBoxLayout()
        left.setSpacing(5)

        for text, slot in [("📂  Open DICOM File", self.open_file),
                            ("📁  Open DICOM Folder", self.open_folder)]:
            b = QPushButton(text)
            b.setFixedHeight(34)
            b.setStyleSheet(self._btn_style())
            b.clicked.connect(slot)
            left.addWidget(b)

        meta_group = QGroupBox("Patient Info")
        meta_group.setStyleSheet(self._group_style())
        mg = QGridLayout(meta_group)
        self.meta_labels = {}
        for i, key in enumerate(["Patient","Modality","Study Date","Rows","Columns","Institution"]):
            lk = QLabel(f"{key}:")
            lk.setStyleSheet("color:#666; font-size:10px;")
            lv = QLabel("—")
            lv.setStyleSheet("color:#ddd; font-size:10px;")
            lv.setWordWrap(True)
            mg.addWidget(lk, i, 0)
            mg.addWidget(lv, i, 1)
            self.meta_labels[key] = lv
        left.addWidget(meta_group)

        win_group = QGroupBox("Windowing")
        win_group.setStyleSheet(self._group_style())
        wg = QVBoxLayout(win_group)
        self.lbl_wl    = QLabel("WL: 40")
        self.lbl_ww    = QLabel("WW: 400")
        self.slider_wl = QSlider(Qt.Horizontal)
        self.slider_wl.setRange(-1000, 3000)
        self.slider_wl.setValue(40)
        self.slider_wl.valueChanged.connect(self._on_wl)
        self.slider_ww = QSlider(Qt.Horizontal)
        self.slider_ww.setRange(1, 4000)
        self.slider_ww.setValue(400)
        self.slider_ww.valueChanged.connect(self._on_ww)
        for w in (self.lbl_wl, self.slider_wl, self.lbl_ww, self.slider_ww):
            w.setStyleSheet("color:#bbb; font-size:10px;" if isinstance(w, QLabel) else "")
            wg.addWidget(w)
        presets = QHBoxLayout()
        for name, (l, w) in [("Brain",(40,80)),("Lung",(-600,1500)),("Bone",(400,1800))]:
            pb = QPushButton(name)
            pb.setFixedHeight(24)
            pb.setStyleSheet(self._btn_style(small=True))
            pb.clicked.connect(lambda _, lv=l, wv=w: self._preset(lv, wv))
            presets.addWidget(pb)
        wg.addLayout(presets)
        left.addWidget(win_group)

        # AI Segmentation Controls
        ai_group = QGroupBox("AI Segmentation  (TotalSegmentator)")
        ai_group.setStyleSheet(self._group_style())
        ag = QVBoxLayout(ai_group)

        # Modality hint label
        self.lbl_modality_hint = QLabel("Auto-detects CT / MRI / PET")
        self.lbl_modality_hint.setStyleSheet("color:#555; font-size:9px; font-style:italic;")
        ag.addWidget(self.lbl_modality_hint)

        self.btn_segment = QPushButton("🤖  Run Segmentation")
        self.btn_segment.setFixedHeight(34)
        self.btn_segment.setStyleSheet(self._btn_style(highlight=True))
        self.btn_segment.clicked.connect(self._run_segmentation)
        ag.addWidget(self.btn_segment)

        self.btn_toggle_overlay = QPushButton("👁  Toggle Overlay")
        self.btn_toggle_overlay.setFixedHeight(28)
        self.btn_toggle_overlay.setStyleSheet(self._btn_style())
        self.btn_toggle_overlay.clicked.connect(self._toggle_overlay)
        self.btn_toggle_overlay.setEnabled(False)
        ag.addWidget(self.btn_toggle_overlay)

        opacity_lbl = QLabel("Overlay Opacity:")
        opacity_lbl.setStyleSheet("color:#bbb; font-size:10px;")
        ag.addWidget(opacity_lbl)

        self.slider_opacity = QSlider(Qt.Horizontal)
        self.slider_opacity.setRange(10, 90)
        self.slider_opacity.setValue(45)
        self.slider_opacity.valueChanged.connect(self._on_opacity)
        ag.addWidget(self.slider_opacity)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setStyleSheet(
            "QProgressBar{border:none;background:#222;border-radius:3px;}"
            "QProgressBar::chunk{background:#3af;border-radius:3px;}"
        )
        ag.addWidget(self.progress_bar)

        self.lbl_seg_status = QLabel("No segmentation loaded")
        self.lbl_seg_status.setStyleSheet("color:#666; font-size:10px;")
        self.lbl_seg_status.setWordWrap(True)
        ag.addWidget(self.lbl_seg_status)

        left.addWidget(ai_group)

        gest_group = QGroupBox("Gesture Control")
        gest_group.setStyleSheet(self._group_style())
        gg = QVBoxLayout(gest_group)

        self.lbl_gesture = QLabel("Waiting...")
        self.lbl_gesture.setStyleSheet(
            "color:#3af; font-size:12px; font-weight:bold; qproperty-alignment:AlignCenter;"
        )
        gg.addWidget(self.lbl_gesture)

        self.lbl_ai_indicator = QLabel("AI Mode: OFF")
        self.lbl_ai_indicator.setStyleSheet(
            "color:#666; font-size:11px; qproperty-alignment:AlignCenter;"
        )
        gg.addWidget(self.lbl_ai_indicator)

        guide = [
            "LEFT FIST (hold 1s) → AI Mode",
            "RIGHT Swipe Up     → Prev Slice",
            "RIGHT Swipe Down   → Next Slice",
            "RIGHT Pinch        → Zoom In",
            "RIGHT Fist         → Zoom Out",
            "RIGHT Open Palm    → Reset",
            "RIGHT Point Up     → Scroll Up",
            "RIGHT Peace        → Scroll Down",
        ]
        for line in guide:
            l = QLabel(line)
            l.setStyleSheet("color:#444; font-size:9px;")
            gg.addWidget(l)

        left.addWidget(gest_group)

        self.lbl_slice = QLabel("Slice: — / —")
        self.lbl_slice.setStyleSheet(
            "color:#aaa; font-size:11px; qproperty-alignment:AlignCenter;"
        )
        left.addWidget(self.lbl_slice)
        left.addStretch()

        btn_reset = QPushButton("↺  Reset View")
        btn_reset.setFixedHeight(30)
        btn_reset.setStyleSheet(self._btn_style())
        btn_reset.clicked.connect(self._reset_view)
        left.addWidget(btn_reset)

        root.addLayout(left, 1)

        # ── CENTRE ─────────────────────────────
        centre = QVBoxLayout()

        self.dicom_label = QLabel()
        self.dicom_label.setAlignment(Qt.AlignCenter)
        self.dicom_label.setMinimumSize(620, 540)
        self.dicom_label.setText("Open a DICOM file or folder")
        self.dicom_label.setFont(QFont("Courier New", 12))
        self.dicom_label.setStyleSheet(
            "background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;"
        )

        self.slice_slider = QSlider(Qt.Vertical)
        self.slice_slider.setRange(0, 0)
        self.slice_slider.valueChanged.connect(self._on_slice_change)

        img_row = QHBoxLayout()
        img_row.addWidget(self.dicom_label)
        img_row.addWidget(self.slice_slider)
        centre.addLayout(img_row)

        nav = QHBoxLayout()
        for text, slot in [("◀  Prev", self._prev_slice), ("Next  ▶", self._next_slice)]:
            b = QPushButton(text)
            b.setFixedHeight(30)
            b.setStyleSheet(self._btn_style())
            b.clicked.connect(slot)
            nav.addWidget(b)
        centre.addLayout(nav)

        root.addLayout(centre, 3)

        # ── RIGHT PANEL ────────────────────────
        right = QVBoxLayout()
        right.setSpacing(6)

        cam_title = QLabel("✋  Gesture Camera (Two-Hand Mode)")
        cam_title.setStyleSheet("color:#3af; font-size:12px; font-weight:bold;")
        cam_title.setAlignment(Qt.AlignCenter)
        right.addWidget(cam_title)

        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        self.cam_label.setFixedSize(360, 290)
        self.cam_label.setStyleSheet(
            "background:#0a0a0a; border:1px solid #2a2a2a; border-radius:6px;"
        )
        right.addWidget(self.cam_label)

        self.lbl_cam_status = QLabel("● Camera: connecting...")
        self.lbl_cam_status.setStyleSheet("color:#888; font-size:10px;")
        right.addWidget(self.lbl_cam_status)

        legend_title = QLabel("Organ Legend")
        legend_title.setStyleSheet("color:#3af; font-size:11px; font-weight:bold;")
        right.addWidget(legend_title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border:none; background:#161616;")
        self.legend_widget = LegendWidget()
        scroll.setWidget(self.legend_widget)
        right.addWidget(scroll)

        root.addLayout(right, 1)

        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status.setStyleSheet("color:#666; background:#111;")
        self.status.showMessage("Ready — open a DICOM file to begin.")

    def _start_gesture_engine(self):
        self.gesture_engine = TwoHandGestureEngine()
        self.gesture_engine.gesture_detected.connect(self._on_gesture)
        self.gesture_engine.frame_ready.connect(self._on_cam_frame)
        self.gesture_engine.ai_mode_changed.connect(self._on_ai_mode_changed)
        self.gesture_engine.start()
        self.lbl_cam_status.setText("● Camera: active")
        self.lbl_cam_status.setStyleSheet("color:#3f3; font-size:10px;")

    def _on_gesture(self, gesture, hand):
        action = GESTURE_ACTIONS.get(gesture, gesture)
        self.lbl_gesture.setText(f"{gesture}\n{action}")
        self.status.showMessage(f"{hand.upper()} hand: {gesture}  →  {action}")

        if gesture == "SWIPE_UP"   : self._prev_slice()
        elif gesture == "SWIPE_DOWN": self._next_slice()
        elif gesture == "OPEN_PALM" : self._reset_view()
        elif gesture == "POINT_UP"  : self._prev_slice()
        elif gesture == "PEACE"     : self._next_slice()
        elif gesture == "PINCH"     : self._zoom_in()
        elif gesture == "FIST"      : self._zoom_out()

    def _on_ai_mode_changed(self, active):
        self.ai_mode = active
        if active:
            self.lbl_ai_indicator.setText("🤖 AI Mode: ACTIVE")
            self.lbl_ai_indicator.setStyleSheet(
                "color:#f90; font-size:11px; font-weight:bold; qproperty-alignment:AlignCenter;"
            )
            self.dicom_label.setStyleSheet(
                "background:#0d0d0d; color:#333; border:2px solid #f90; border-radius:6px;"
            )
            if self.seg_mask is not None:
                self.show_overlay = True
                self._show_slice()
            self.status.showMessage("AI Mode ACTIVE — overlay enabled")
        else:
            self.lbl_ai_indicator.setText("AI Mode: OFF")
            self.lbl_ai_indicator.setStyleSheet(
                "color:#666; font-size:11px; qproperty-alignment:AlignCenter;"
            )
            self.dicom_label.setStyleSheet(
                "background:#0d0d0d; color:#333; border:1px solid #222; border-radius:6px;"
            )
            self.show_overlay = False
            self._show_slice()

    def _on_cam_frame(self, frame):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.cam_label.width(), self.cam_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.cam_label.setPixmap(pix)

    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open DICOM File", "", "DICOM (*.dcm);;All (*)")
        if path:
            self.dicom_path = path
            self._load(path)

    def open_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Open DICOM Folder")
        if path:
            self.dicom_path = path
            self._load(path)

    def _load(self, path):
        self.status.showMessage("Loading DICOM...")
        self.seg_mask     = None
        self.show_overlay = False
        self.loader = DicomLoader(path)
        self.loader.loaded.connect(self._on_loaded)
        self.loader.error.connect(lambda e: self.status.showMessage(f"Error: {e}"))
        self.loader.start()

    def _on_loaded(self, volume, sitk_image, meta):
        self.volume     = volume
        self.sitk_image = sitk_image
        self.modality   = meta.get("Modality", "CT").upper()
        self.current    = volume.shape[0] // 2
        self.zoom       = 1.0
        self.slice_slider.setRange(0, volume.shape[0] - 1)
        self.slice_slider.setValue(self.current)
        self._update_meta(meta)
        self._reset_windowing()
        self._show_slice()

        task = _pick_task(self.modality)
        self.lbl_modality_hint.setText(
            f"Detected: {self.modality}  →  will use task='{task}'"
        )
        self.lbl_modality_hint.setStyleSheet("color:#3af; font-size:9px;")

        self.status.showMessage(
            f"Loaded {volume.shape[0]} slices  |  {volume.shape[2]}×{volume.shape[1]} px"
            f"  |  Modality: {self.modality}  |  Task: {task}"
        )

    def _run_segmentation(self):
        if self.dicom_path is None:
            self.status.showMessage("Please open a DICOM file or folder first.")
            return

        task = _pick_task(self.modality)
        self.btn_segment.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.lbl_seg_status.setText(
            f"Starting TotalSegmentator...\n"
            f"Task: {task}  |  Device: GPU (fast mode)\n"
            f"First run will download model weights."
        )
        self.status.showMessage(f"Segmentation running (task={task}) — please wait...")

        self.seg_worker = SegmentationWorker(self.dicom_path, self.modality)
        self.seg_worker.progress.connect(self._on_seg_progress)
        self.seg_worker.finished.connect(self._on_seg_done)
        self.seg_worker.error.connect(self._on_seg_error)
        self.seg_worker.start()

    def _on_seg_progress(self, msg):
        self.lbl_seg_status.setText(msg)
        self.status.showMessage(msg)

    def _on_seg_done(self, mask):
        self.seg_mask = mask
        self.progress_bar.setVisible(False)
        self.btn_segment.setEnabled(True)
        self.btn_toggle_overlay.setEnabled(True)
        self.show_overlay = True

        active = [l for l in ORGAN_COLOURS if np.any(mask == l)]

        scroll_area = self.findChild(QScrollArea)
        if scroll_area:
            self.legend_widget = LegendWidget(active_labels=active)
            scroll_area.setWidget(self.legend_widget)

        self.lbl_seg_status.setText(
            f"✅ Segmentation complete!\n{len(active)} structures found."
        )
        self.status.showMessage(f"Done — {len(active)} organs/structures segmented")
        self._show_slice()

    def _on_seg_error(self, msg):
        self.progress_bar.setVisible(False)
        self.btn_segment.setEnabled(True)
        self.lbl_seg_status.setText(f"❌ Error:\n{msg}")
        self.status.showMessage(f"Segmentation error — see panel")

    def _toggle_overlay(self):
        if self.seg_mask is not None:
            self.show_overlay = not self.show_overlay
            self._show_slice()

    def _on_opacity(self, val):
        self.opacity = val / 100.0
        self._show_slice()

    def _show_slice(self):
        if self.volume is None:
            return

        raw = self.volume[self.current]
        img = self._apply_window(raw)

        if self.zoom != 1.0:
            h, w   = img.shape
            new_h  = int(h / self.zoom)
            new_w  = int(w / self.zoom)
            cy, cx = h // 2, w // 2
            y1 = max(0, cy - new_h // 2)
            y2 = min(h, cy + new_h // 2)
            x1 = max(0, cx - new_w // 2)
            x2 = min(w, cx + new_w // 2)
            img = img[y1:y2, x1:x2]
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

        if self.show_overlay and self.seg_mask is not None:
            mz       = self.seg_mask.shape[0]
            vz       = self.volume.shape[0]
            mask_idx = int(self.current * mz / vz)
            mask_idx = min(mask_idx, mz - 1)
            mask_slice = self.seg_mask[mask_idx]

            if mask_slice.shape != img.shape:
                mask_slice = cv2.resize(
                    mask_slice.astype(np.float32),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            display = apply_segmentation_overlay(img, mask_slice, self.opacity)
            h, w    = display.shape[:2]
            qimg    = QImage(display.tobytes(), w, h, w * 3, QImage.Format_BGR888)
        else:
            h, w = img.shape
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)

        pix = QPixmap.fromImage(qimg).scaled(
            self.dicom_label.width()  - 4,
            self.dicom_label.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.dicom_label.setPixmap(pix)
        self.lbl_slice.setText(f"Slice:  {self.current + 1} / {self.volume.shape[0]}")

    def _apply_window(self, raw):
        lo  = self.wl - self.ww / 2
        hi  = self.wl + self.ww / 2
        img = np.clip(raw, lo, hi)
        img = ((img - lo) / (hi - lo) * 255).astype(np.uint8)
        return img

    def _on_slice_change(self, val):
        self.current = val
        self._show_slice()

    def _prev_slice(self):
        if self.volume is not None and self.current > 0:
            self.current -= 1
            self.slice_slider.setValue(self.current)

    def _next_slice(self):
        if self.volume is not None and self.current < self.volume.shape[0] - 1:
            self.current += 1
            self.slice_slider.setValue(self.current)

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            self._prev_slice()
        else:
            self._next_slice()

    def _zoom_in(self):
        self.zoom = min(self.zoom + 0.1, 3.0)
        self._show_slice()

    def _zoom_out(self):
        self.zoom = max(self.zoom - 0.1, 1.0)
        self._show_slice()

    def _on_wl(self, val):
        self.wl = float(val)
        self.lbl_wl.setText(f"WL: {val}")
        self._show_slice()

    def _on_ww(self, val):
        self.ww = float(val)
        self.lbl_ww.setText(f"WW: {val}")
        self._show_slice()

    def _preset(self, wl, ww):
        self.slider_wl.setValue(wl)
        self.slider_ww.setValue(ww)

    def _reset_windowing(self):
        if self.volume is not None:
            self.wl = float(np.median(self.volume))
            self.ww = float(self.volume.max() - self.volume.min())
            self.slider_wl.setValue(int(self.wl))
            self.slider_ww.setValue(int(self.ww))

    def _reset_view(self):
        if self.volume is not None:
            self.current = self.volume.shape[0] // 2
            self.zoom    = 1.0
            self.slice_slider.setValue(self.current)
            self._reset_windowing()

    def _update_meta(self, meta):
        for k, v in meta.items():
            if k in self.meta_labels:
                self.meta_labels[k].setText(v)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._show_slice()

    def closeEvent(self, event):
        if self.gesture_engine:
            self.gesture_engine.stop()
            self.gesture_engine.wait()
        event.accept()

    def _apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background:#1a1a1a; color:#ddd;
                font-family:'Segoe UI', sans-serif; font-size:11px;
            }
            QSlider::groove:horizontal {
                height:4px; background:#2a2a2a; border-radius:2px;
            }
            QSlider::handle:horizontal {
                background:#3af; width:12px; height:12px;
                margin:-4px 0; border-radius:6px;
            }
            QSlider::sub-page:horizontal { background:#3af; border-radius:2px; }
            QSlider::groove:vertical {
                width:4px; background:#2a2a2a; border-radius:2px;
            }
            QSlider::handle:vertical {
                background:#3af; width:12px; height:12px;
                margin:0 -4px; border-radius:6px;
            }
            QSlider::sub-page:vertical { background:#3af; border-radius:2px; }
            QStatusBar { background:#111; color:#555; }
            QScrollArea { border:none; }
        """)

    def _btn_style(self, small=False, highlight=False):
        p  = "3px 6px" if small else "5px 10px"
        bg = "#1a3a1a" if highlight else "#222"
        bc = "#3a3" if highlight else "#333"
        hb = "#2a4a2a" if highlight else "#2a2a2a"
        return f"""
            QPushButton {{
                background:{bg}; color:#ccc; border:1px solid {bc};
                border-radius:4px; padding:{p};
            }}
            QPushButton:hover {{ background:{hb}; color:#fff; border-color:#3af; }}
            QPushButton:pressed {{ background:#111; }}
            QPushButton:disabled {{ background:#181818; color:#444; border-color:#222; }}
        """

    def _group_style(self):
        return """
            QGroupBox {
                border:1px solid #252525; border-radius:6px;
                margin-top:8px; padding:5px; color:#555; font-size:10px;
            }
            QGroupBox::title { subcontrol-origin:margin; left:8px; color:#3af; }
        """


# ──────────────────────────────────────────────
#  Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())