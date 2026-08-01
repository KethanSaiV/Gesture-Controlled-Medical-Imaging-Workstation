with open('viewer_state.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = []

# Check if PATH_PLAN exists
if 'PATH_PLAN' not in src:
    # Add PATH_PLAN to Mode enum
    old = '''class Mode(Enum):
    DICOM    = auto()
    FILTER   = auto()
    SEGMENT  = auto()
    REGISTER = auto()
    MEASURE  = auto()
    REVIEW   = auto()'''
    new = '''class Mode(Enum):
    DICOM     = auto()
    FILTER    = auto()
    SEGMENT   = auto()
    REGISTER  = auto()
    PATH_PLAN = auto()
    MEASURE   = auto()
    REVIEW    = auto()'''
    if old in src:
        src = src.replace(old, new)
        fixes.append('✅ PATH_PLAN added to Mode enum')
    else:
        fixes.append('⚠️  Mode enum not found exactly')

# Check MODE_LABELS
if 'PATH_PLAN' in src and 'Path Plan' not in src:
    old_labels = '    Mode.REGISTER: "Register",'
    new_labels = '    Mode.REGISTER : "Register",\n    Mode.PATH_PLAN: "Path Plan",'
    if old_labels in src:
        src = src.replace(old_labels, new_labels)
        fixes.append('✅ PATH_PLAN added to MODE_LABELS')

# Check MODE_ICONS
if 'PATH_PLAN' in src and ('PATH_PLAN' not in src.split('MODE_ICONS')[1][:200] if 'MODE_ICONS' in src else True):
    old_icons = '    Mode.REGISTER: "🔗",'
    new_icons = '    Mode.REGISTER : "🔗",\n    Mode.PATH_PLAN: "🧭",'
    if old_icons in src:
        src = src.replace(old_icons, new_icons)
        fixes.append('✅ PATH_PLAN added to MODE_ICONS')

# Add path planning state slots if missing
if 'tumor_voxel' not in src:
    old_measure = '        self.measure_points: list = []'
    new_measure = '''        self.measure_points: list = []

        # ── Path planning state ────────────────
        self.sitk_image       = None
        self.tumor_point_mm   = None
        self.tumor_voxel      = None
        self.path_candidates  = []
        self.path_ranked      = []
        self.selected_path_idx = None

        # ── Registration result state ──────────
        self.reg_ct_arr   = None
        self.reg_mri_arr  = None
        self.reg_ct_sitk  = None
        self.reg_metrics  = {}
        self.reg_out_dir  = ""
        self.view_mode    = 0'''
    if old_measure in src:
        src = src.replace(old_measure, new_measure)
        fixes.append('✅ Path planning + registration state slots added')

with open('viewer_state.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print('\n'.join(fixes) if fixes else '✅ viewer_state.py already complete')
    print('✅ viewer_state.py syntax OK')
except SyntaxError as e:
    print(f'❌ Syntax error at line {e.lineno}: {e.msg}')
