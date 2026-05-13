with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

fixes = [
    (
        'self.lbl_gesture.setText("✌️✌️ BOTH PEACE\n🔒 View Locked")',
        'self.lbl_gesture.setText("BOTH PEACE\\n🔒 View Locked")'
    ),
    (
        'self.lbl_gesture.setText("👍☝️ BOTH L\n🔓 View Unlocked")',
        'self.lbl_gesture.setText("BOTH L-SHAPE\\n🔓 View Unlocked")'
    ),
    (
        'self.lbl_gesture.setText("👊👊 BOTH FIST\n↺ View Reset")',
        'self.lbl_gesture.setText("BOTH FIST\\n↺ View Reset")'
    ),
    (
        'self.lbl_gesture.setText("🖐🖐 BOTH PALM\n👁 Overlay Toggled")',
        'self.lbl_gesture.setText("BOTH PALM\\n👁 Overlay Toggled")'
    ),
]

applied = 0
for old, new in fixes:
    if old in src:
        src = src.replace(old, new)
        applied += 1
        print(f'✅ Fixed: {old[:40]}...')
    else:
        print(f'⚠️  Not found: {old[:40]}...')

with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print(f'\n✅ Syntax OK — {applied} fixes applied')
    print('Run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'\n❌ Still broken at line {e.lineno}: {e.msg}')
    print(f'   Text: {repr(e.text)}')