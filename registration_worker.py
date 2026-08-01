"""
registration_worker.py
Gesture-Controlled Medical Imaging Workstation

Rigid CT <-> MRI registration backend (SimpleITK, Mattes Mutual Information).
Ported from the Phase-4 registration prototype (k.py) into a standalone
module so it can be imported by main2.py without dragging along a second
copy of the DICOM loader / segmentation worker / gesture engine.

Public API
----------
RegistrationWorker(QThread)
    .progress(str)   -> log line
    .finished(dict)  -> {"ct_arr", "mri_arr", "metrics", "out_dir", "reg_dir"}
    .error(str)      -> error message

VIEW_MODES = ["CT", "MRI Registered", "Overlay (CT+MRI)"]
"""

from __future__ import annotations

import os
import json

import numpy as np
import SimpleITK as sitk
from PyQt5.QtCore import QThread, pyqtSignal


VIEW_MODES = ["CT", "MRI Registered", "Overlay (CT+MRI)"]


# ══════════════════════════════════════════════════════════════
#  Low level helpers
# ══════════════════════════════════════════════════════════════

def _normalize(img):
    arr = sitk.GetArrayFromImage(img).astype(np.float32)
    mn, mx = np.percentile(arr, (1, 99))
    arr = (arr - mn) / (mx - mn + 1e-9)
    arr = np.clip(arr, 0, 1)
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(img)
    return out


def _compute_mri_fov_bounds(mri_arr, margin=2):
    mask = mri_arr != 0
    if not mask.any():
        raise RuntimeError("MRI rigid volume is all zero; FOV cannot be computed.")
    coords = np.array(np.nonzero(mask))
    z_min, y_min, x_min = coords.min(axis=1)
    z_max, y_max, x_max = coords.max(axis=1)
    z_min = max(z_min - margin, 0)
    y_min = max(y_min - margin, 0)
    x_min = max(x_min - margin, 0)
    z_max = min(z_max + margin, mri_arr.shape[0] - 1)
    y_max = min(y_max + margin, mri_arr.shape[1] - 1)
    x_max = min(x_max + margin, mri_arr.shape[2] - 1)
    return int(z_min), int(z_max), int(y_min), int(y_max), int(x_min), int(x_max)


def _compute_nmi(fixed_img, moving_img, nbins=64):
    fa = sitk.GetArrayFromImage(fixed_img).astype(np.float32).ravel()
    ma = sitk.GetArrayFromImage(moving_img).astype(np.float32).ravel()
    mask = np.logical_and(np.isfinite(fa), np.isfinite(ma))
    fa, ma = fa[mask], ma[mask]
    if fa.size == 0:
        return 0.0
    h2d, _, _ = np.histogram2d(fa, ma, bins=nbins)
    pxy = h2d / np.sum(h2d)
    px, py = pxy.sum(axis=1), pxy.sum(axis=0)
    eps = 1e-12
    Hx = -np.sum(px * np.log(px + eps))
    Hy = -np.sum(py * np.log(py + eps))
    Hxy = -np.sum(pxy * np.log(pxy + eps))
    return float((Hx + Hy) / Hxy) if Hxy > 0 else 0.0


def _compute_mismatch(fixed_img, moving_img):
    fa = sitk.GetArrayFromImage(fixed_img).astype(np.float32)
    ma = sitk.GetArrayFromImage(moving_img).astype(np.float32)
    mask = np.logical_and(np.isfinite(fa), np.isfinite(ma))
    fa, ma = fa[mask], ma[mask]
    if fa.size == 0:
        return 0.0, 0.0, 0.0
    mad = float(np.mean(np.abs(fa - ma)))
    fz = fa - fa.mean()
    mz = ma - ma.mean()
    denom = np.linalg.norm(fz) * np.linalg.norm(mz)
    ncc = float(np.dot(fz, mz) / denom) if denom > 0 else 0.0
    return mad, ncc, 1.0 - ncc


def _compute_edge_dice(fixed_img, moving_img, sigma=1.0, lower=0.1, upper=0.3):
    def norm01(a):
        a = a.copy().astype(np.float32)
        a -= a.min()
        if a.max() > 0:
            a /= a.max()
        return a

    fa = norm01(sitk.GetArrayFromImage(fixed_img))
    ma = norm01(sitk.GetArrayFromImage(moving_img))
    fn = sitk.GetImageFromArray(fa); fn.CopyInformation(fixed_img)
    mn = sitk.GetImageFromArray(ma); mn.CopyInformation(moving_img)
    canny = sitk.CannyEdgeDetectionImageFilter()
    canny.SetVariance(sigma ** 2)
    canny.SetLowerThreshold(lower)
    canny.SetUpperThreshold(upper)
    fe = sitk.GetArrayFromImage(canny.Execute(fn)) > 0
    me = sitk.GetArrayFromImage(canny.Execute(mn)) > 0
    if fe.sum() == 0 or me.sum() == 0:
        return 0.0
    return float(2.0 * np.logical_and(fe, me).sum() / (fe.sum() + me.sum()))


def load_series(series_dir):
    """Load a DICOM series from a folder (drilling one level down if needed)."""
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(series_dir)
    if not ids:
        for sub in sorted(os.listdir(series_dir)):
            sp = os.path.join(series_dir, sub)
            if os.path.isdir(sp):
                ids = reader.GetGDCMSeriesIDs(sp)
                if ids:
                    series_dir = sp
                    break
    if not ids:
        raise RuntimeError(f"No DICOM series found in {series_dir}")
    files = reader.GetGDCMSeriesFileNames(series_dir, ids[0])
    reader.SetFileNames(files)
    return reader.Execute()


# ══════════════════════════════════════════════════════════════
#  RegistrationWorker
# ══════════════════════════════════════════════════════════════

class RegistrationWorker(QThread):
    """
    Full CT (fixed) <- MRI (moving) rigid registration pipeline, run off the
    UI thread. CT and MRI are read from separate DICOM folders (this keeps
    registration independent from whatever is currently loaded in the main
    viewer, matching clinical workflow where CT/MRI come from different
    studies).
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, ct_dir, mri_dir, out_dir, patient_id):
        super().__init__()
        self.ct_dir = ct_dir
        self.mri_dir = mri_dir
        self.out_dir = out_dir
        self.patient_id = patient_id or "Patient01"

    def _emit(self, msg):
        self.progress.emit(msg)

    def run(self):
        try:
            patient_dir = os.path.join(self.out_dir, self.patient_id)
            reg_dir = os.path.join(patient_dir, "registered")
            for d in (reg_dir,):
                os.makedirs(d, exist_ok=True)

            self._emit("Loading CT series...")
            ct = load_series(self.ct_dir)
            ct = sitk.DICOMOrient(ct, "RAI")
            self._emit(f"CT loaded: {ct.GetSize()} voxels")

            self._emit("Loading MRI series...")
            mri = load_series(self.mri_dir)
            mri = sitk.DICOMOrient(mri, "RAI")
            self._emit(f"MRI loaded: {mri.GetSize()} voxels")

            self._emit("Normalising intensities...")
            ct_n = _normalize(ct)
            mri_n = _normalize(mri)

            self._emit("Initialising rigid registration (Mattes MI)...")
            initial = sitk.CenteredTransformInitializer(
                ct_n, mri_n,
                sitk.VersorRigid3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY,
            )
            reg = sitk.ImageRegistrationMethod()
            reg.SetMetricAsMattesMutualInformation(50)
            reg.SetMetricSamplingStrategy(reg.RANDOM)
            reg.SetMetricSamplingPercentage(0.02)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsGradientDescent(
                learningRate=1.0, numberOfIterations=200,
                convergenceMinimumValue=1e-6, convergenceWindowSize=10,
            )
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetShrinkFactorsPerLevel([4, 2, 1])
            reg.SetSmoothingSigmasPerLevel([2, 1, 0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            reg.SetInitialTransform(initial, inPlace=False)

            self._emit("Running registration (multi-resolution: 4x -> 2x -> 1x)...")
            final_rigid = reg.Execute(ct_n, mri_n)
            metric_val = reg.GetMetricValue()
            self._emit(f"Registration complete. Final metric: {metric_val:.6f}")

            self._emit("Resampling MRI to CT grid...")
            mri_rigid = sitk.Resample(
                mri, ct, final_rigid, sitk.sitkLinear, 0.0, mri.GetPixelID()
            )

            self._emit("Computing MRI field-of-view and cropping...")
            mri_arr = sitk.GetArrayFromImage(mri_rigid)
            z0, z1, y0, y1, x0, x1 = _compute_mri_fov_bounds(mri_arr, margin=2)

            roi = sitk.RegionOfInterestImageFilter()
            roi.SetIndex([int(x0), int(y0), int(z0)])
            roi.SetSize([int(x1 - x0 + 1), int(y1 - y0 + 1), int(z1 - z0 + 1)])
            ct_crop = roi.Execute(ct)
            mri_crop = roi.Execute(mri_rigid)

            self._emit("Saving NIfTI outputs...")
            sitk.WriteImage(ct, os.path.join(reg_dir, "ct_fixed.nii.gz"))
            sitk.WriteImage(mri, os.path.join(reg_dir, "mri_original.nii.gz"))
            sitk.WriteImage(mri_rigid, os.path.join(reg_dir, "mri_rigid_to_ct.nii.gz"))
            sitk.WriteImage(ct_crop, os.path.join(reg_dir, "ct_cropped.nii.gz"))
            sitk.WriteImage(mri_crop, os.path.join(reg_dir, "mri_cropped.nii.gz"))
            sitk.WriteTransform(final_rigid, os.path.join(reg_dir, "rigid_transform.tfm"))

            self._emit("Computing registration quality metrics...")
            nmi = _compute_nmi(ct_crop, mri_crop)
            edge_dice = _compute_edge_dice(ct_crop, mri_crop)
            mad, ncc, mis = _compute_mismatch(ct_crop, mri_crop)
            ct_vox = np.prod(ct.GetSize())
            fov_vox = (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1)
            fov_ratio = float(fov_vox) / float(ct_vox) if ct_vox > 0 else 0.0

            metrics = {
                "nmi_rigid_cropped": round(nmi, 4),
                "edge_dice_rigid_cropped": round(edge_dice, 4),
                "mad_rigid_cropped": round(mad, 4),
                "ncc_rigid_cropped": round(ncc, 4),
                "mismatch_rigid_cropped": round(mis, 4),
                "fov_overlap_ratio": round(fov_ratio, 4),
                "final_metric_value": round(float(metric_val), 6),
            }
            with open(os.path.join(reg_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)

            self._emit(
                f"Metrics:\n"
                f"  NMI              : {nmi:.4f}\n"
                f"  Edge Dice        : {edge_dice:.4f}\n"
                f"  NCC              : {ncc:.4f}\n"
                f"  MAD              : {mad:.4f}\n"
                f"  FoV overlap ratio: {fov_ratio:.4f}"
            )

            ct_crop_arr = sitk.GetArrayFromImage(ct_crop).astype(np.float32)
            mri_crop_arr = sitk.GetArrayFromImage(mri_crop).astype(np.float32)

            self._emit("Registration pipeline complete!")
            self.finished.emit({
                "ct_arr": ct_crop_arr,
                "mri_arr": mri_crop_arr,
                "ct_sitk": ct_crop,          # kept for physical-coordinate use (path planning etc.)
                "metrics": metrics,
                "out_dir": patient_dir,
                "reg_dir": reg_dir,
            })

        except Exception as e:
            import traceback
            self.error.emit(f"Registration error:\n{e}\n{traceback.format_exc()}")
