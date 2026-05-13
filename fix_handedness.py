with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    c = f.read()

old = '                    if handedness == "Left":\n                        right_landmarks = lm_list\n                        right_vis       = vis_list\n                    else:\n                        left_landmarks  = lm_list\n                        left_vis        = vis_list'
new = '                    if handedness == "Left":\n                        left_landmarks  = lm_list\n                        left_vis        = vis_list\n                    else:\n                        right_landmarks = lm_list\n                        right_vis       = vis_list'

if old in c:
    c = c.replace(old, new)
    with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
        f.write(c)
    print('FIXED — handedness swap corrected')
else:
    print('NOT FOUND — showing landmark assignment lines:')
    for i, line in enumerate(c.splitlines(), 1):
        if 'landmarks = lm_list' in line or 'vis = vis_list' in line:
            print(f'  Line {i}: {repr(line)}')