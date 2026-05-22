with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# The problem: line 1910 is the outer if, but lines 1911-1920
# are indented as if inside an inner if block that doesn't exist.
# The inner "if self.zoom..." block is missing its header line.
# Fix: insert the missing inner if header before line 1911.

fixed = []
for i, line in enumerate(lines):
    lineno = i + 1
    if lineno == 1910:
        # This is the outer: "if self.zoom != 1.0 or (self.pan_x..."
        # It should be followed by an inner if for the mask
        # But actually looking at the structure, line 1910 IS
        # "if self.zoom != 1.0..." for the MASK block,
        # but it's missing the "if self.zoom..." wrapper.
        # The fix: change indentation of 1911-1920 from 16 to 12 spaces
        fixed.append(line)
    elif lineno in range(1911, 1921):
        # These are over-indented by 4 spaces (16 instead of 12)
        if line.startswith('                '):  # 16 spaces
            fixed.append('            ' + line[16:])  # reduce to 12
        else:
            fixed.append(line)
    else:
        fixed.append(line)

with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.writelines(fixed)

# Verify
import ast
with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

try:
    ast.parse(src)
    print('✅ Syntax OK — run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'❌ Still broken at line {e.lineno}: {e.msg}')
    print(f'   Text: {repr(e.text)}')
    # Show context
    lines = src.splitlines()
    start = max(0, e.lineno - 4)
    end   = min(len(lines), e.lineno + 4)
    for i, l in enumerate(lines[start:end], start + 1):
        marker = ' <<<' if i == e.lineno else ''
        print(f'  {i}: {repr(l)}{marker}')