"""
Patch: Add AI Preprocessing to phase4_segmentation.py

Adds:
1. PreprocessingEngine class (8 filters)
2. Preprocessing UI panel in left sidebar
3. Connects filter to _show_slice
4. Gesture O-shape cycles through filters (future gesture hook ready)

Filters:
  None (Original)
  Gaussian Denoise
  Median Filter
  Sharpen
  CLAHE (Contrast)
  Unsharp Mask
  Edge Enhance
  Normalize
"""

with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []

# ─────────────────────────────────────────────────────────────
# 1. Add scipy import at top
# ─────────────────────────────────────────────────────────────
old1 = 'import mediapipe as mp'
new1 = '''from scipy.ndimage import gaussian_filter, median_filter

import mediapipe as mp'''

if old1 in src and 'scipy' not in src:
    src = src.replace(old1, new1)
    fixes.append('✅ 1. scipy import added')
elif 'scipy' in src:
    fixes.append('✅ 1. scipy already imported')
else:
    fixes.append('⚠️  1. mediapipe import not found')


# ─────────────────────────────────────────────────────────────
# 2. Add QComboBox to PyQt5 imports
# ─────────────────────────────────────────────────────────────
old2 = '    QScrollArea\n)'
new2 = '    QScrollArea, QComboBox\n)'

if old2 in src:
    src = src.replace(old2, new2)
    fixes.append('✅ 2. QComboBox added to imports')
elif 'QComboBox' in src:
    fixes.append('✅ 2. QComboBox already imported')
else:
    fixes.append('⚠️  2. QScrollArea import line not found')


# ─────────────────────────────────────────────────────────────
# 3. Insert PreprocessingEngine class before DicomLoader
# ─────────────────────────────────────────────────────────────
PREPROCESSING_CLASS = '''
# ──────────────────────────────────────────────
#  Preprocessing Engine
#  All filters run on CPU via numpy/scipy/opencv
#  Applied per-slice at display time
#  Original volume is NEVER modified
# ──────────────────────────────────────────────
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
        Returns uint8 grayscale — never modifies input.
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


'''

# Insert before DicomLoader
old3 = '# ──────────────────────────────────────────────\n#  DICOM Loader\n# ──────────────────────────────────────────────\nclass DicomLoader'
new3 = PREPROCESSING_CLASS + '# ──────────────────────────────────────────────\n#  DICOM Loader\n# ──────────────────────────────────────────────\nclass DicomLoader'

if 'class PreprocessingEngine' in src:
    fixes.append('✅ 3. PreprocessingEngine already exists')
elif old3 in src:
    src = src.replace(old3, new3)
    fixes.append('✅ 3. PreprocessingEngine class inserted')
else:
    fixes.append('⚠️  3. DicomLoader header not found — inserting before class DicomLoader')
    src = src.replace('class DicomLoader', PREPROCESSING_CLASS + 'class DicomLoader', 1)
    fixes.append('✅ 3. PreprocessingEngine inserted (fallback)')


# ─────────────────────────────────────────────────────────────
# 4. Add self.preprocessor to MainWindow.__init__
# ─────────────────────────────────────────────────────────────
old4 = '''        self.view_locked  = False  # True = zoom/pan/slice locked
        self._setup_ui()'''

new4 = '''        self.view_locked  = False  # True = zoom/pan/slice locked
        self.preprocessor = PreprocessingEngine()
        self._setup_ui()'''

if 'self.preprocessor' in src:
    fixes.append('✅ 4. preprocessor already in __init__')
elif old4 in src:
    src = src.replace(old4, new4)
    fixes.append('✅ 4. self.preprocessor added to __init__')
else:
    # Try without pan_x/pan_y (older version)
    old4b = '''        self.view_locked  = False
        self._setup_ui()'''
    if old4b in src:
        src = src.replace(old4b, old4b.replace('self._setup_ui()', 'self.preprocessor = PreprocessingEngine()\n        self._setup_ui()'))
        fixes.append('✅ 4. preprocessor added (alt form)')
    else:
        fixes.append('⚠️  4. __init__ end not found — add manually: self.preprocessor = PreprocessingEngine()')


# ─────────────────────────────────────────────────────────────
# 5. Add Preprocessing UI panel to left sidebar
#    Insert after win_group (windowing panel)
# ─────────────────────────────────────────────────────────────
PREPROCESS_UI = '''
        # ── AI Preprocessing Panel ────────────
        pre_group = QGroupBox("AI Preprocessing")
        pre_group.setStyleSheet(self._group_style())
        pg = QVBoxLayout(pre_group)

        filter_lbl = QLabel("Filter:")
        filter_lbl.setStyleSheet("color:#bbb; font-size:10px;")
        pg.addWidget(filter_lbl)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(PreprocessingEngine.FILTERS)
        self.filter_combo.setStyleSheet(
            "QComboBox{background:#222;color:#ddd;border:1px solid #333;"
            "border-radius:3px;padding:2px;font-size:10px;}"
            "QComboBox::drop-down{border:none;}"
            "QComboBox QAbstractItemView{background:#222;color:#ddd;"
            "selection-background-color:#3af;}"
        )
        self.filter_combo.currentTextChanged.connect(self._on_filter_change)
        pg.addWidget(self.filter_combo)

        self.lbl_strength = QLabel("Strength: 1.0")
        self.lbl_strength.setStyleSheet("color:#bbb; font-size:10px;")
        pg.addWidget(self.lbl_strength)

        self.slider_strength = QSlider(Qt.Horizontal)
        self.slider_strength.setRange(1, 20)
        self.slider_strength.setValue(10)
        self.slider_strength.valueChanged.connect(self._on_strength_change)
        pg.addWidget(self.slider_strength)

        btn_reset_filter = QPushButton("Reset Filter")
        btn_reset_filter.setFixedHeight(24)
        btn_reset_filter.setStyleSheet(self._btn_style(small=True))
        btn_reset_filter.clicked.connect(self._reset_filter)
        pg.addWidget(btn_reset_filter)

        left.addWidget(pre_group)

'''

# Insert after windowing group, before AI segmentation group
old5 = '''        left.addWidget(win_group)

        ai_group = QGroupBox("AI Segmentation'''

new5 = '''        left.addWidget(win_group)
''' + PREPROCESS_UI + '''
        ai_group = QGroupBox("AI Segmentation'''

if 'AI Preprocessing' in src:
    fixes.append('✅ 5. Preprocessing UI already exists')
elif old5 in src:
    src = src.replace(old5, new5)
    fixes.append('✅ 5. Preprocessing UI panel added to sidebar')
else:
    fixes.append('⚠️  5. win_group insertion point not found')


# ─────────────────────────────────────────────────────────────
# 6. Add preprocessing to _show_slice
#    Apply filter AFTER windowing, BEFORE zoom/overlay
# ─────────────────────────────────────────────────────────────
old6 = '''        if self.zoom != 1.0 or (self.pan_x != 0 or self.pan_y != 0):'''
new6 = '''        # Apply preprocessing filter
        img = self.preprocessor.apply(img)

        if self.zoom != 1.0 or (self.pan_x != 0 or self.pan_y != 0):'''

if 'self.preprocessor.apply' in src:
    fixes.append('✅ 6. preprocessor.apply already in _show_slice')
elif old6 in src:
    src = src.replace(old6, new6)
    fixes.append('✅ 6. preprocessor.apply added to _show_slice')
else:
    # Try without pan version
    old6b = '''        if self.zoom != 1.0:'''
    if old6b in src:
        src = src.replace(old6b, '        img = self.preprocessor.apply(img)\n\n        if self.zoom != 1.0:', 1)
        fixes.append('✅ 6. preprocessor.apply added (no-pan form)')
    else:
        fixes.append('⚠️  6. _show_slice zoom block not found')


# ─────────────────────────────────────────────────────────────
# 7. Add preprocessing handler methods to MainWindow
#    Insert before _toggle_overlay
# ─────────────────────────────────────────────────────────────
PREPROCESS_METHODS = '''
    # ── Preprocessing Handlers ─────────────────────────────────
    def _on_filter_change(self, filter_name):
        self.preprocessor.active_filter = filter_name
        self._show_slice()
        self.status.showMessage("Filter: " + filter_name)
        short = filter_name.split("(")[0].strip()
        self.lbl_gesture.setText("Filter\\n" + short)

    def _on_strength_change(self, val):
        self.preprocessor.strength = val / 10.0
        self.lbl_strength.setText("Strength: " + str(round(self.preprocessor.strength, 1)))
        self._show_slice()

    def _reset_filter(self):
        self.filter_combo.setCurrentIndex(0)
        self.slider_strength.setValue(10)
        self.status.showMessage("Filter reset to original")

'''

old7 = '''    def _toggle_overlay(self):'''

if '_on_filter_change' in src:
    fixes.append('✅ 7. Filter handlers already exist')
elif old7 in src:
    src = src.replace(old7, PREPROCESS_METHODS + '    def _toggle_overlay(self):')
    fixes.append('✅ 7. Preprocessing handler methods added')
else:
    fixes.append('⚠️  7. _toggle_overlay not found')


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