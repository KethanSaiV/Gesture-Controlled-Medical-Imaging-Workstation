with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    c = f.read()

fixes = []

# ── FIX 1: Handedness swap ─────────────────────────────────────────────
# MediaPipe reports Left=Left, Right=Right after cv2.flip — NO swap needed
old1 = '''                    # FIX: swap because cv2.flip mirrors display
                    if handedness == "Left":
                        right_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (0, 200, 100))
                    else:
                        left_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (200, 100, 0))'''

new1 = '''                    # After cv2.flip: Left=Left, Right=Right (confirmed by debug)
                    if handedness == "Left":
                        left_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (200, 100, 0))
                    else:
                        right_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (0, 200, 100))'''

if old1 in c:
    c = c.replace(old1, new1)
    fixes.append('✅ Fix 1: Handedness swap corrected')
else:
    fixes.append('⚠️  Fix 1: Handedness block not found — checking alternate form...')
    # Try alternate (no comment line)
    old1b = '''                    if handedness == "Left":
                        right_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (0, 200, 100))
                    else:
                        left_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (200, 100, 0))'''
    new1b = '''                    if handedness == "Left":
                        left_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (200, 100, 0))
                    else:
                        right_landmarks = lm_list
                        self._draw_landmarks(frame, hand_lms, w, h, (0, 200, 100))'''
    if old1b in c:
        c = c.replace(old1b, new1b)
        fixes.append('  ✅ Fix 1 (alternate form): Handedness swap corrected')
    else:
        fixes.append('  ❌ Fix 1 not applied — print lines manually')

# ── FIX 2: Zoom + mask misalignment ───────────────────────────────────
# The bug: zoom crops image pixels but mask is at FULL resolution coords.
# Fix: apply zoom to BOTH image and mask slice together, then overlay.
old2 = '''        if self.show_overlay and self.seg_mask is not None:
            mz=self.seg_mask.shape[0]; vz=self.volume.shape[0]
            mask_idx=min(int(self.current*mz/vz), mz-1)
            mask_slice=self.seg_mask[mask_idx]
            if mask_slice.shape != img.shape:
                mask_slice=cv2.resize(mask_slice.astype(np.float32),(img.shape[1],img.shape[0]),interpolation=cv2.INTER_NEAREST).astype(np.int32)
            display=apply_segmentation_overlay(img,mask_slice,self.opacity)
            h,w=display.shape[:2]; qimg=QImage(display.tobytes(),w,h,w*3,QImage.Format_BGR888)
        else:
            h,w=img.shape; qimg=QImage(img.tobytes(),w,h,w,QImage.Format_Grayscale8)'''

new2 = '''        if self.show_overlay and self.seg_mask is not None:
            mz=self.seg_mask.shape[0]; vz=self.volume.shape[0]
            mask_idx=min(int(self.current*mz/vz), mz-1)
            mask_slice=self.seg_mask[mask_idx]
            # Resize mask to match original volume slice dimensions first
            raw_h, raw_w = self.volume[self.current].shape
            if mask_slice.shape != (raw_h, raw_w):
                mask_slice=cv2.resize(mask_slice.astype(np.float32),
                                      (raw_w, raw_h),
                                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
            # Apply zoom to MASK the same way as the image
            # (crop same region, resize to same display size)
            if self.zoom != 1.0:
                h0, w0 = mask_slice.shape
                new_h = int(h0 / self.zoom); new_w = int(w0 / self.zoom)
                cy, cx = h0//2, w0//2
                y1=max(0,cy-new_h//2); y2=min(h0,cy+new_h//2)
                x1=max(0,cx-new_w//2); x2=min(w0,cx+new_w//2)
                mask_slice = cv2.resize(mask_slice[y1:y2, x1:x2].astype(np.float32),
                                        (w0, h0),
                                        interpolation=cv2.INTER_NEAREST).astype(np.int32)
            # Now img and mask_slice are both at display resolution — overlay them
            if mask_slice.shape != img.shape:
                mask_slice=cv2.resize(mask_slice.astype(np.float32),
                                      (img.shape[1],img.shape[0]),
                                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
            display=apply_segmentation_overlay(img,mask_slice,self.opacity)
            h,w=display.shape[:2]; qimg=QImage(display.tobytes(),w,h,w*3,QImage.Format_BGR888)
        else:
            h,w=img.shape; qimg=QImage(img.tobytes(),w,h,w,QImage.Format_Grayscale8)'''

if old2 in c:
    c = c.replace(old2, new2)
    fixes.append('✅ Fix 2: Zoom+mask alignment corrected')
else:
    fixes.append('⚠️  Fix 2: Overlay block not found — may already be patched or formatting differs')

# ── FIX 3: Lower detection thresholds for better two-hand detection ───
old3 = '            min_hand_detection_confidence=0.5,\n            min_hand_presence_confidence=0.5,\n            min_tracking_confidence=0.5,'
new3 = '            min_hand_detection_confidence=0.2,\n            min_hand_presence_confidence=0.2,\n            min_tracking_confidence=0.2,'
if old3 in c:
    c = c.replace(old3, new3)
    fixes.append('✅ Fix 3: Detection thresholds lowered (0.5 → 0.2)')
else:
    fixes.append('⚠️  Fix 3: Thresholds already patched or not found')

# ── Write fixed file ───────────────────────────────────────────────────
with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(c)

print('\n'.join(fixes))
print()

import ast
try:
    ast.parse(c)
    print('✅ Syntax OK — run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'❌ Syntax error at line {e.lineno}: {e.msg}')
    print(f'   Text: {e.text}')