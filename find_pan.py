with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Find the pan detection section and show what's there
lines = src.splitlines()
for i, line in enumerate(lines):
    if 'pan_moved.emit' in line or 'PAN_SCALE' in line or 'pan_prev_pos' in line.lower():
        print(f"Line {i+1}: {repr(line)}")