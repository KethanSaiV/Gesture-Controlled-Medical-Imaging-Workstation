with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Show lines 1915-1930 to see the problem
print("=== Lines 1910-1935 ===")
for i, line in enumerate(lines[1909:1935], 1910):
    print(f"{i}: {repr(line)}")