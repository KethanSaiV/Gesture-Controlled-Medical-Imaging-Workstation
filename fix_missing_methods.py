with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

METHODS = '''
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

# Insert before _toggle_overlay
target = '    def _toggle_overlay(self):'
if target in src:
    src = src.replace(target, METHODS + target, 1)
    print('✅ Methods inserted')
else:
    print('⚠️  _toggle_overlay not found — inserting before _on_opacity')
    target2 = '    def _on_opacity(self, val):'
    if target2 in src:
        src = src.replace(target2, METHODS + target2, 1)
        print('✅ Methods inserted before _on_opacity')
    else:
        print('❌ Could not find insertion point')

with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print('✅ Syntax OK — run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'❌ Syntax error at line {e.lineno}: {e.msg}')