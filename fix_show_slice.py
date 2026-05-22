with open('phase4_segmentation.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Find and replace the entire _show_slice method
# We'll find it by its def line and replace until _apply_window

old = '''    def _show_slice(self):
        if self.volume is None: return
        raw = self.volume[self.current]; img = self._apply_window(raw)
        # Apply preprocessing filter
        img = self.preprocessor.apply(img)

        if self.zoom != 1.0 or (self.pan_x != 0 or self.pan_y != 0):
            h, w   = img.shape
            new_h  = int(h / self.zoom)
            new_w  = int(w / self.zoom)
            # Pan offsets shift the crop centre
            cy = h//2 + self.pan_y
            cx = w//2 + self.pan_x
            y1 = max(0, cy - new_h//2); y2 = min(h, y1 + new_h)
            x1 = max(0, cx - new_w//2); x2 = min(w, x1 + new_w)
            # Clamp so we never crop outside image
            if y2 > h: y1 = max(0, h - new_h); y2 = h
            if x2 > w: x1 = max(0, w - new_w); x2 = w
            img = cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)
        if self.show_overlay and self.seg_mask is not None:
            mz=self.seg_mask.shape[0]; vz=self.volume.shape[0]
            mask_idx=min(int(self.current*mz/vz), mz-1)
            mask_slice=self.seg_mask[mask_idx]
            # Resize mask to match original volume slice dimensions first
            raw_h, raw_w = self.volume[self.current].shape
            if mask_slice.shape != (raw_h, raw_w):
                mask_slice=cv2.resize(mask_slice.astype(np.float32),
                                      (raw_w, raw_h),
                                      interpolation=cv2.INTER_NEAREST).astype(np.int32)
            # Apply SAME zoom + pan to mask as to the image
            # Apply preprocessing filter
        img = self.preprocessor.apply(img)

        if self.zoom != 1.0 or (self.pan_x != 0 or self.pan_y != 0):
            h0, w0 = mask_slice.shape
            new_h  = int(h0 / self.zoom); new_w = int(w0 / self.zoom)
            cy = h0//2 + self.pan_y; cx = w0//2 + self.pan_x
            y1 = max(0, cy - new_h//2); y2 = min(h0, y1 + new_h)
            x1 = max(0, cx - new_w//2); x2 = min(w0, x1 + new_w)
            if y2 > h0: y1 = max(0, h0 - new_h); y2 = h0
            if x2 > w0: x1 = max(0, w0 - new_w); x2 = w0
            mask_slice = cv2.resize(mask_slice[y1:y2, x1:x2].astype(np.float32),
                                    (w0, h0),
                                    interpolation=cv2.INTER_NEAREST).astype(np.int32)
            # Now img and mask_slice are both at display resolution — overlay them'''

new = '''    def _show_slice(self):
        if self.volume is None:
            return

        # 1. Window
        raw = self.volume[self.current]
        img = self._apply_window(raw)

        # 2. Preprocessing filter
        img = self.preprocessor.apply(img)

        # 3. Zoom + pan crop (image)
        if self.zoom != 1.0 or self.pan_x != 0 or self.pan_y != 0:
            h, w  = img.shape
            new_h = int(h / self.zoom)
            new_w = int(w / self.zoom)
            cy = h//2 + self.pan_y
            cx = w//2 + self.pan_x
            y1 = max(0, cy - new_h//2); y2 = min(h, y1 + new_h)
            x1 = max(0, cx - new_w//2); x2 = min(w, x1 + new_w)
            if y2 > h: y1 = max(0, h - new_h); y2 = h
            if x2 > w: x1 = max(0, w - new_w); x2 = w
            img = cv2.resize(img[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)

        # 4. Overlay
        if self.show_overlay and self.seg_mask is not None:
            mz = self.seg_mask.shape[0]
            vz = self.volume.shape[0]
            mask_idx   = min(int(self.current * mz / vz), mz - 1)
            mask_slice = self.seg_mask[mask_idx]

            # Resize mask to raw slice dimensions
            raw_h, raw_w = self.volume[self.current].shape
            if mask_slice.shape != (raw_h, raw_w):
                mask_slice = cv2.resize(
                    mask_slice.astype(np.float32),
                    (raw_w, raw_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            # Apply same zoom + pan to mask
            if self.zoom != 1.0 or self.pan_x != 0 or self.pan_y != 0:
                h0, w0 = mask_slice.shape
                new_h  = int(h0 / self.zoom)
                new_w  = int(w0 / self.zoom)
                cy = h0//2 + self.pan_y
                cx = w0//2 + self.pan_x
                y1 = max(0, cy - new_h//2); y2 = min(h0, y1 + new_h)
                x1 = max(0, cx - new_w//2); x2 = min(w0, x1 + new_w)
                if y2 > h0: y1 = max(0, h0 - new_h); y2 = h0
                if x2 > w0: x1 = max(0, w0 - new_w); x2 = w0
                mask_slice = cv2.resize(
                    mask_slice[y1:y2, x1:x2].astype(np.float32),
                    (w0, h0),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            # Final size match
            if mask_slice.shape != img.shape:
                mask_slice = cv2.resize(
                    mask_slice.astype(np.float32),
                    (img.shape[1], img.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)

            display = apply_segmentation_overlay(img, mask_slice, self.opacity)
            h, w    = display.shape[:2]
            qimg    = QImage(display.tobytes(), w, h, w * 3, QImage.Format_BGR888)
        else:
            h, w = img.shape
            qimg = QImage(img.tobytes(), w, h, w, QImage.Format_Grayscale8)

        pix = QPixmap.fromImage(qimg).scaled(
            self.dicom_label.width() - 4,
            self.dicom_label.height() - 4,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.dicom_label.setPixmap(pix)
        self.lbl_slice.setText(f"Slice:  {self.current+1} / {self.volume.shape[0]}")

    def _PLACEHOLDER_(self):
        pass'''

if old in src:
    # Also need to remove the leftover tail that comes after the broken block
    # Find what comes after the old block
    tail_start = src.find(old) + len(old)
    # Find next method def after the broken section
    tail = src[tail_start:]
    # The broken section continues until the next proper method
    # Find the closing lines of the broken _show_slice
    broken_tail_end = tail.find('\n    def _apply_window')
    if broken_tail_end >= 0:
        broken_tail = tail[:broken_tail_end]
        print(f"Removing broken tail ({len(broken_tail)} chars):")
        print(repr(broken_tail[:200]))
        src = src[:src.find(old)] + new + tail[broken_tail_end:]
        src = src.replace('    def _PLACEHOLDER_(self):\n        pass\n', '')
        print('✅ _show_slice fully replaced')
    else:
        print('⚠️  Could not find _apply_window after broken section')
else:
    print('⚠️  Exact old block not found — trying line-based replacement')
    # Fall back: find _show_slice def and _apply_window def, replace everything between
    start_marker = '    def _show_slice(self):\n'
    end_marker   = '    def _apply_window(self, raw):\n'
    start_idx = src.find(start_marker)
    end_idx   = src.find(end_marker)
    if start_idx >= 0 and end_idx >= 0 and end_idx > start_idx:
        clean_new = new.replace('    def _PLACEHOLDER_(self):\n        pass', '')
        src = src[:start_idx] + clean_new + '\n' + src[end_idx:]
        print('✅ _show_slice replaced via line-based method')
    else:
        print(f'❌ Could not find markers. start={start_idx}, end={end_idx}')

with open('phase4_segmentation.py', 'w', encoding='utf-8') as f:
    f.write(src)

import ast
try:
    ast.parse(src)
    print('✅ Syntax OK — run: python phase4_segmentation.py')
except SyntaxError as e:
    print(f'❌ Syntax error at line {e.lineno}: {e.msg}')
    print(f'   Text: {repr(e.text)}')