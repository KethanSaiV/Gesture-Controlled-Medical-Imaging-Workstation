# Gesture-Controlled Medical Imaging Workstation

A touchless AI-assisted DICOM viewer for MRI and CT scans. Doctors and surgeons interact with medical scans using real-time hand gestures instead of mouse and keyboard.

---

## System Requirements

| Component | Requirement |
|-----------|-------------|
| OS | Windows 10/11 |
| Python | 3.11 |
| GPU | NVIDIA GTX 1650 Max-Q (4GB VRAM) or better |
| CUDA | 11.6+ |
| RAM | 8GB+ recommended |
| Webcam | Any USB or built-in webcam |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-repo/gesture-controlled-imaging.git
cd gesture-controlled-imaging
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install PyTorch with CUDA support

```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### 4. Install all other dependencies

```bash
pip install opencv-python mediapipe pydicom SimpleITK PyQt5 numpy scipy pillow monai TotalSegmentator
```

### 5. Verify GPU is detected

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce GTX 1650 with Max-Q Design
```

---

## Running the Application

```bash
python phase4_segmentation.py
```

---

## Project Structure

```
gesture-controlled-imaging/
│
├── phase4_segmentation.py     ← Main application (Phase 4 — current)
├── main.py                    ← Phase 3 (gesture + viewer, no segmentation)
├── dicom_viewer.py            ← Phase 1 (basic DICOM viewer)
├── gesture_recognition.py     ← Phase 2 (standalone gesture test)
├── hand_landmarker.task       ← MediaPipe model (auto-downloaded)
├── requirements.txt           ← All dependencies
└── README.md                  ← This file
```

---

## How to Use

### Loading a Scan

1. Click **📂 Open DICOM File** to load a single `.dcm` file
2. Click **📁 Open DICOM Folder** to load a full MRI/CT series
3. The scan appears in the centre panel with patient metadata on the left

### Navigating Slices

| Method | Action |
|--------|--------|
| Mouse wheel | Scroll through slices |
| Prev / Next buttons | Step one slice at a time |
| Vertical slider | Jump to any slice |
| Gesture: Swipe Up | Previous slice |
| Gesture: Swipe Down | Next slice |

### Windowing Presets

| Button | WL / WW | Best for |
|--------|---------|----------|
| Brain | 40 / 80 | Brain MRI/CT |
| Lung | -600 / 1500 | Chest CT |
| Bone | 400 / 1800 | Bone CT |

---

## Gesture Controls

Place your hands in front of the webcam (right panel shows live feed).

### Two-Hand Mode

| Hand | Gesture | Action |
|------|---------|--------|
| Left | Fist (hold 1 second) | Toggle AI Mode ON/OFF |
| Right | Swipe Up | Previous slice |
| Right | Swipe Down | Next slice |
| Right | Pinch (thumb + index close) | Zoom in |
| Right | Fist | Zoom out |
| Right | Open Palm (all fingers up) | Reset view |
| Right | Point Up (index only) | Scroll up |
| Right | Peace (index + middle) | Scroll down |

### Tips for Best Gesture Recognition

- Keep hand 30–60 cm from the webcam
- Ensure good lighting — avoid backlight
- Make gestures clearly and hold briefly
- The gesture name appears in the **Gesture Control** panel when detected

---

## AI Segmentation

### Running Segmentation

1. Load a DICOM file or folder
2. The app auto-detects modality (MRI or CT) and shows the task in the AI panel
3. Click **🤖 Run Segmentation**
4. First run downloads the model weights (~900MB for MRI, ~2.4GB for CT)
5. Segmentation completes in ~25 seconds on CPU (fast mode)
6. Coloured organ overlays appear automatically on the scan

### Segmentation Models

| Modality | Task | Structures |
|----------|------|------------|
| MRI | `total_mr` | 50 structures |
| CT | `total` | 117 structures |

### Organs Detected (MRI Pelvic Example)

Prostate, Urinary Bladder, Colon, Small Bowel, Femur (L/R), Hip (L/R), Sacrum, Iliopsoas (L/R), Gluteus Maximus/Medius/Minimus (L/R), Iliac Arteries/Veins, Aorta, Inferior Vena Cava, Vertebrae, Intervertebral Discs, Spinal Cord, and more.

### Overlay Controls

| Control | Description |
|---------|-------------|
| 👁 Toggle Overlay | Show/hide the segmentation overlay |
| Opacity slider | Blend overlay with the scan (10–90%) |
| AI Mode (left fist) | Auto-enables overlay when activated |
| Organ Legend | Shows colour key for detected structures only |

---

## Troubleshooting

### Webcam not opening
Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `TwoHandGestureEngine.run()`.

### CUDA not detected
```bash
pip uninstall torch torchvision -y
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

### ModuleNotFoundError on launch
```bash
pip install opencv-python mediapipe pydicom SimpleITK PyQt5 numpy TotalSegmentator
```

### No DICOM series found
Select the folder that **directly contains** the DICOM files (e.g. `vishwa/3/`), not the parent folder (`vishwa/`).

### Segmentation mask is empty
The scan's field of view may not cover the structures the model expects. Try the other modality task by checking that the **Modality** field in Patient Info shows the correct value (MR or CT).

### TensorFlow / NumPy warnings on startup
These are harmless warnings from unrelated packages. The app works correctly. To silence them permanently:
```bash
pip uninstall tensorflow tensorflow-intel keras -y
```

---

## Architecture Overview

```
phase4_segmentation.py
│
├── DicomLoader (QThread)
│   └── Loads DICOM file/folder → numpy volume + metadata
│
├── SegmentationWorker (QThread)
│   ├── Converts DICOM → NIfTI (SimpleITK)
│   ├── Runs TotalSegmentator as subprocess
│   ├── Merges per-organ masks → single label volume
│   └── Emits label array (Z, H, W) int32
│
├── TwoHandGestureEngine (QThread)
│   ├── MediaPipe HandLandmarker (2 hands)
│   ├── Classifies: SWIPE_UP/DOWN, PINCH, OPEN_PALM, FIST, POINT_UP, PEACE
│   └── Emits gesture + hand side signals
│
└── MainWindow (QMainWindow)
    ├── Left panel: controls, metadata, segmentation, gesture log
    ├── Centre panel: DICOM viewer with overlay
    └── Right panel: webcam feed + organ legend
```

---

## Completed Phases

| Phase | File | Status |
|-------|------|--------|
| 1 — Basic DICOM Viewer | `dicom_viewer.py` | ✅ Complete |
| 2 — Gesture Recognition | `gesture_recognition.py` | ✅ Complete |
| 3 — GUI Integration | `main.py` | ✅ Complete |
| 4 — AI Segmentation | `phase4_segmentation.py` | ✅ Complete |

## Planned Phases

| Phase | Description |
|-------|-------------|
| 5 | Multi-modality viewer (MRI + CT + PET side by side) |
| 6 | Image registration integration (CT-MRI overlay) |
| 7 | Preprocessing module (denoise, sharpen, normalize) |
| 8 | Tumour segmentation + surgical path planning (A*/Dijkstra) |
| 9 | Web interface with patient folder management |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyQt5 | 5.x | GUI framework |
| opencv-python | 4.x | Camera + image processing |
| mediapipe | 0.10.35 | Hand landmark detection |
| pydicom | 3.x | DICOM file reading |
| SimpleITK | 2.5+ | Medical image I/O + NIfTI conversion |
| numpy | 1.x / 2.x | Array operations |
| torch | 2.4.1+cu118 | GPU acceleration |
| TotalSegmentator | 2.13+ | AI organ segmentation |
| scipy | 1.x | Signal processing |
| pillow | 12.x | Image utilities |

---

## Citation

If you use TotalSegmentator in your work, please cite:

> Wasserthal, J., et al. "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images." *Radiology: Artificial Intelligence* (2023). https://pubs.rsna.org/doi/10.1148/ryai.230024

---


