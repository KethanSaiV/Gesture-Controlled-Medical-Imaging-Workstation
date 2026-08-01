"""
path_planning_worker.py
Gesture-Controlled Medical Imaging Workstation

Needle path planning backend - adapted from the standalone MooseZ-based
reference script into an in-process module that reuses the TotalSegmentator
organ mask already produced by the Segment tab (no second segmentation
model / no CUDA / no MooseZ dependency, no disk round-trip).

Pipeline (unchanged in spirit from the reference script):
  1. Build a body mask from the CT (HU threshold).
  2. Build organ / rib masks from the existing TotalSegmentator label volume.
  3. Coarse spherical search from the tumour point -> safest direction
     (maximises average clearance from organs/ribs before hitting one,
     leaving the body, or maxing out distance).
  4. Adaptive cone sampling around that safest direction -> candidate
     entry points on the skin.
  5. Filter candidates by needle angle (vs. gantry axis) and length.
  6. Remove near-duplicate entries.
  7. Full-path collision check against organs + ribs; keep only
     collision-free paths, ranked by needle length (shortest first).

All spatial math is done in physical mm using the SimpleITK image that
matches the loaded volume's geometry (spacing / origin / direction).
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk
from PyQt5.QtCore import QThread, pyqtSignal


# Reserved label id for the merged "Ribs" structure inside the shared
# TotalSegmentator label volume (kept out of the 1-50 organ id range
# used elsewhere in the app).
RIBS_LABEL = 51

RIB_NAME_FRAGMENTS = ("rib_left_", "rib_right_", "rib_")


# ══════════════════════════════════════════════════════════════
#  Geometry helpers
# ══════════════════════════════════════════════════════════════

def _is_inside_index(idx, size):
    return all(0 <= idx[d] < size[d] for d in range(3))


def sample_directions(num_theta=10, num_phi=18):
    dirs = []
    for i in range(num_theta):
        theta = np.pi * (i + 0.5) / num_theta
        for j in range(num_phi):
            phi = 2 * np.pi * j / num_phi
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            v = np.array([x, y, z], dtype=float)
            v /= np.linalg.norm(v)
            dirs.append(v)
    return dirs


def adaptive_cone_sampling(cone_axis, cone_angle_deg=30, num_theta=10, num_phi=20):
    cone_axis = np.asarray(cone_axis, dtype=float)
    cone_axis /= np.linalg.norm(cone_axis)

    helper = np.array([0, 0, 1], dtype=float) if abs(cone_axis[2]) < 0.9 else np.array([0, 1, 0], dtype=float)
    u = np.cross(cone_axis, helper); u /= np.linalg.norm(u)
    v = np.cross(cone_axis, u);      v /= np.linalg.norm(v)

    cone_angle = np.radians(cone_angle_deg)
    directions = []
    for i in range(num_theta):
        alpha = cone_angle * (i + 0.5) / num_theta
        for j in range(num_phi):
            beta = 2 * np.pi * j / num_phi
            d = np.cos(alpha) * cone_axis + np.sin(alpha) * (np.cos(beta) * u + np.sin(beta) * v)
            d /= np.linalg.norm(d)
            directions.append(d)
    return directions


def estimate_best_direction(tumor_point_mm, body_img, organs_img, ribs_img,
                             organ_distance_arr, rib_distance_arr,
                             num_theta=12, num_phi=24, step_mm=2.0, max_dist_mm=180.0):
    directions = sample_directions(num_theta, num_phi)
    organs_arr = sitk.GetArrayFromImage(organs_img)
    ribs_arr = sitk.GetArrayFromImage(ribs_img)
    body_arr = sitk.GetArrayFromImage(body_img)
    size = organs_img.GetSize()
    tumor = np.array(tumor_point_mm, dtype=float)

    best_direction, best_score = None, -1.0
    for direction in directions:
        clearance_score, samples = 0.0, 0
        for d in np.arange(0, max_dist_mm, step_mm):
            p = tumor - direction * d
            try:
                idx = organs_img.TransformPhysicalPointToIndex(tuple(p))
            except Exception:
                break
            if not _is_inside_index(idx, size):
                break
            z, y, x = idx[2], idx[1], idx[0]
            if body_arr[z, y, x] == 0:
                break
            if organs_arr[z, y, x] > 0 or ribs_arr[z, y, x] > 0:
                break
            clearance = min(organ_distance_arr[z, y, x], rib_distance_arr[z, y, x])
            clearance_score += clearance
            samples += 1
        if samples == 0:
            continue
        avg = clearance_score / samples
        if avg > best_score:
            best_score, best_direction = avg, direction

    return best_direction, best_score


def find_entry_point_along_direction(tumor_mm, direction, body_img, step_mm=1.0, max_dist_mm=200.0):
    size = body_img.GetSize()
    body_arr = sitk.GetArrayFromImage(body_img)
    tumor = np.array(tumor_mm, dtype=float)
    last_inside_point, last_inside, outside_count = None, False, 0

    for d in np.arange(0.0, max_dist_mm, step_mm):
        p = tumor + d * direction
        idx = body_img.TransformPhysicalPointToIndex(tuple(p))
        if not _is_inside_index(idx, size):
            return last_inside_point
        inside = body_arr[idx[2], idx[1], idx[0]] > 0
        if inside:
            last_inside_point, last_inside, outside_count = p, True, 0
        else:
            outside_count += 1
            if outside_count >= 5:
                if last_inside and last_inside_point is not None:
                    return last_inside_point
    return None


def compute_needle_angle_deg(entry_mm, target_mm, ref_img):
    entry = np.array(entry_mm, dtype=float)
    target = np.array(target_mm, dtype=float)
    v = target - entry
    norm_v = np.linalg.norm(v)
    if norm_v == 0:
        return None
    v /= norm_v
    direction = np.array(ref_img.GetDirection()).reshape(3, 3)
    axial_normal = direction[:, 2]
    norm_n = np.linalg.norm(axial_normal)
    axial_normal = axial_normal / norm_n if norm_n else np.array([0., 0., 1.])
    cos_theta = np.clip(abs(np.dot(v, axial_normal)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def evaluate_path(entry_mm, target_mm, organs_img, organs_arr, ribs_arr,
                   organ_distance_arr, rib_distance_arr, num_samples=800):
    entry = np.array(entry_mm, dtype=float)
    target = np.array(target_mm, dtype=float)
    size = organs_img.GetSize()

    min_organ_clearance, min_rib_clearance, collision = 9999.0, 9999.0, False
    for t in np.linspace(0, 1, num_samples):
        p = entry + t * (target - entry)
        idx = organs_img.TransformPhysicalPointToIndex(tuple(p))
        if not _is_inside_index(idx, size):
            collision = True
            break
        z, y, x = idx[2], idx[1], idx[0]
        if organs_arr[z, y, x] > 0 or ribs_arr[z, y, x] > 0:
            collision = True
            break
        min_organ_clearance = min(min_organ_clearance, organ_distance_arr[z, y, x])
        min_rib_clearance = min(min_rib_clearance, rib_distance_arr[z, y, x])

    return {
        "collision": collision,
        "organ_clearance": float(min_organ_clearance),
        "rib_clearance": float(min_rib_clearance),
    }


# ══════════════════════════════════════════════════════════════
#  Mask construction from the shared TotalSegmentator label volume
# ══════════════════════════════════════════════════════════════

def build_masks(raw_volume: np.ndarray, seg_mask: np.ndarray, ref_img: sitk.Image,
                 body_hu_threshold: float = -400.0):
    """
    Build (body_img, organs_img, ribs_img, organ_distance_arr, rib_distance_arr)
    as SimpleITK images / numpy arrays sharing ref_img's geometry.

    organs_img  : every non-zero label EXCEPT RIBS_LABEL
    ribs_img    : only RIBS_LABEL voxels
    """
    body_arr = (raw_volume > body_hu_threshold).astype(np.uint8)
    body_img = sitk.GetImageFromArray(body_arr)
    body_img.CopyInformation(ref_img)

    organs_arr = np.where((seg_mask > 0) & (seg_mask != RIBS_LABEL), 1, 0).astype(np.uint8)
    ribs_arr = np.where(seg_mask == RIBS_LABEL, 1, 0).astype(np.uint8)

    organs_img = sitk.GetImageFromArray(organs_arr)
    organs_img.CopyInformation(ref_img)
    ribs_img = sitk.GetImageFromArray(ribs_arr)
    ribs_img.CopyInformation(ref_img)

    organ_distance = sitk.SignedMaurerDistanceMap(
        organs_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
    rib_distance = sitk.SignedMaurerDistanceMap(
        ribs_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)

    return (body_img, organs_img, ribs_img,
            sitk.GetArrayFromImage(organ_distance), sitk.GetArrayFromImage(rib_distance))


# ══════════════════════════════════════════════════════════════
#  Worker thread
# ══════════════════════════════════════════════════════════════

class PathPlanWorker(QThread):
    """
    Runs candidate needle-path generation + collision filtering off the UI
    thread. Requires a CT volume with a completed TotalSegmentator mask and
    a tumour point (mm, physical coordinates).
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)   # {"valid": [...], "invalid": [...], "ranked": [...]}
    error = pyqtSignal(str)

    def __init__(self, raw_volume, seg_mask, ref_img, tumor_point_mm,
                 max_angle_deg=60.0, min_length_mm=20.0, max_length_mm=150.0,
                 cone_angle_deg=30.0, top_k=5):
        super().__init__()
        self.raw_volume = raw_volume
        self.seg_mask = seg_mask
        self.ref_img = ref_img
        self.tumor_point_mm = tuple(float(v) for v in tumor_point_mm)
        self.max_angle_deg = max_angle_deg
        self.min_length_mm = min_length_mm
        self.max_length_mm = max_length_mm
        self.cone_angle_deg = cone_angle_deg
        self.top_k = top_k

    def run(self):
        try:
            self.progress.emit("Building body / organ / rib masks...")
            body_img, organs_img, ribs_img, organ_dist_arr, rib_dist_arr = build_masks(
                self.raw_volume, self.seg_mask, self.ref_img)
            organs_arr = sitk.GetArrayFromImage(organs_img)
            ribs_arr = sitk.GetArrayFromImage(ribs_img)

            if not ribs_arr.any():
                self.progress.emit(
                    "Warning: no rib structures found in segmentation "
                    "(re-run Segment with a task that includes ribs for rib avoidance).")

            self.progress.emit("Searching for the safest coarse direction...")
            best_direction, best_score = estimate_best_direction(
                self.tumor_point_mm, body_img, organs_img, ribs_img,
                organ_dist_arr, rib_dist_arr, num_theta=14, num_phi=28)

            if best_direction is None:
                self.error.emit("Could not find any safe direction from the tumour point. "
                                 "Check that the tumour point lies inside the body.")
                return
            self.progress.emit(
                f"Safest coarse direction found (avg. clearance {best_score:.1f} mm).")

            self.progress.emit("Sampling candidate directions in adaptive cone...")
            directions = adaptive_cone_sampling(
                best_direction, cone_angle_deg=self.cone_angle_deg,
                num_theta=12, num_phi=24)

            self.progress.emit(f"Evaluating {len(directions)} candidate directions...")
            candidates = []
            entry_points = []
            tumor = np.array(self.tumor_point_mm, dtype=float)
            for direction in directions:
                entry = find_entry_point_along_direction(self.tumor_point_mm, direction, body_img)
                if entry is None:
                    continue
                entry = np.array(entry, dtype=float)

                angle_deg = compute_needle_angle_deg(entry, tumor, self.ref_img)
                if angle_deg is None or angle_deg > self.max_angle_deg:
                    continue

                length = float(np.linalg.norm(tumor - entry))
                if length < self.min_length_mm or length > self.max_length_mm:
                    continue

                if any(np.linalg.norm(entry - e) < 5.0 for e in entry_points):
                    continue

                entry_points.append(entry)
                candidates.append({
                    "entry": tuple(float(v) for v in entry),
                    "target": tuple(float(v) for v in tumor),
                    "length": length,
                    "angle_deg": angle_deg,
                })

            self.progress.emit(f"{len(candidates)} candidate paths passed angle/length checks.")
            if not candidates:
                self.error.emit("No candidate needle paths satisfied the angle/length constraints. "
                                 "Try relaxing Max Angle or Length limits.")
                return

            self.progress.emit("Checking full-path collisions against organs + ribs...")
            valid, invalid = [], []
            for p in candidates:
                result = evaluate_path(p["entry"], p["target"], organs_img,
                                        organs_arr, ribs_arr, organ_dist_arr, rib_dist_arr)
                if result["collision"]:
                    invalid.append(p)
                else:
                    p = dict(p)
                    p["organ_clearance"] = result["organ_clearance"]
                    p["rib_clearance"] = result["rib_clearance"]
                    valid.append(p)

            self.progress.emit(f"{len(valid)} collision-free paths, {len(invalid)} rejected.")

            ranked = sorted(valid, key=lambda p: p["length"])[: self.top_k]

            self.progress.emit("Path planning complete!")
            self.finished.emit({"valid": valid, "invalid": invalid, "ranked": ranked})

        except Exception as e:
            import traceback
            self.error.emit(f"Path planning error:\n{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
#  Export helper (Slicer-compatible FCSV) - optional convenience
# ══════════════════════════════════════════════════════════════

def save_paths_to_fcsv(paths, out_path, list_name="NeedlePaths"):
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Markups fiducial file version = 4.11\n")
        f.write(f"# name = {list_name}\n")
        f.write("# coordinateSystem = 0\n")
        f.write("# columns = label,x,y,z,ow,ox,oy,oz,vis,sel,lock,desc,associatedNodeID\n")
        for i, p in enumerate(paths):
            x, y, z = p["entry"]
            desc = (f"len={p['length']:.1f}mm,angle={p['angle_deg']:.1f},"
                    f"organ={p.get('organ_clearance', 0):.1f}mm,"
                    f"rib={p.get('rib_clearance', 0):.1f}mm")
            f.write(f"Entry_{i},{x},{y},{z},0,0,0,1,1,1,0,\"{desc}\",\n")
