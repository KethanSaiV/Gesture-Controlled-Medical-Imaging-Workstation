with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Show lines 1870-1940 to see full _show_slice structure
print("=== Lines 1870-1940 ===")
for i, line in enumerate(lines[1869:1940], 1870):
    print(f"{i}: {repr(line)}")