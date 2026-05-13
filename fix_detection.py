with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    c = f.read()

fixes = [
    # Lower detection thresholds so both hands are caught
    (
        'min_hand_detection_confidence = 0.4,\n            min_hand_presence_confidence  = 0.4,\n            min_tracking_confidence       = 0.4,',
        'min_hand_detection_confidence = 0.2,\n            min_hand_presence_confidence  = 0.2,\n            min_tracking_confidence       = 0.2,'
    ),
    # Remove histogram equalisation — it distorts skin tone and hurts MediaPipe
    (
        '            # ── Histogram equalisation on V channel ──\n            # Makes hand detection lighting-robust\n            hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)\n            hsv[:,:,2]  = cv2.equalizeHist(hsv[:,:,2])\n            enhanced    = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)\n\n            rgb    = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)',
        '            # Mild CLAHE on luminance — gentler than full equalisation\n            lab         = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)\n            clahe       = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))\n            lab[:,:,0]  = clahe.apply(lab[:,:,0])\n            enhanced    = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)\n\n            rgb    = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)'
    ),
    # Increase camera resolution for better detection
    (
        '        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)\n        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)',
        '        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)\n        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)'
    ),
    # Lower vote window so gestures fire faster
    (
        '    VOTE_WINDOW       = 6     # frames gesture must be consistent before firing',
        '    VOTE_WINDOW       = 4     # frames gesture must be consistent before firing'
    ),
    # Lower vote threshold
    (
        '    VOTE_THRESHOLD    = 0.70  # fraction of frames that must agree (70%)',
        '    VOTE_THRESHOLD    = 0.60  # fraction of frames that must agree (60%)'
    ),
    # Lower MIN_VISIBILITY so partially occluded hands still work
    (
        '    MIN_VISIBILITY = 0.5',
        '    MIN_VISIBILITY = 0.1'
    ),
]

applied = 0
for old, new in fixes:
    if old in c:
        c = c.replace(old, new)
        applied += 1
        print(f'✅ Fix {applied} applied')
    else:
        print(f'⚠️  Fix {applied+1} not found — skipping')

with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(c)

print(f'\nDone — {applied}/{len(fixes)} fixes applied')

import ast
try:
    ast.parse(c)
    print('Syntax OK ✅')
except SyntaxError as e:
    print(f'Syntax error at line {e.lineno}: {e.msg}')