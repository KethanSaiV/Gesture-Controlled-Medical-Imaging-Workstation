"""
Phase 2 - Gesture Recognition System
Gesture-Controlled Medical Imaging Workstation
Compatible with MediaPipe 0.10+ (new API)
"""

import cv2
import mediapipe as mp
import numpy as np
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ──────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────
GESTURE_COOLDOWN = 1.0

COL_GREEN  = (0, 255, 120)
COL_BLUE   = (255, 180, 0)
COL_RED    = (0, 80, 255)
COL_WHITE  = (255, 255, 255)
COL_YELLOW = (0, 220, 255)

GESTURE_ACTIONS = {
    "SWIPE_UP"   : "Previous Slice",
    "SWIPE_DOWN" : "Next Slice",
    "PINCH"      : "Zoom",
    "OPEN_PALM"  : "Reset Viewer",
    "FIST"       : "Activate AI",
    "POINT_UP"   : "Scroll Up",
    "PEACE"      : "Scroll Down",
}


# ──────────────────────────────────────────────
#  Gesture Detector
# ──────────────────────────────────────────────
class GestureDetector:

    def __init__(self):
        model_path = self._get_model()
        base_opts  = mp_python.BaseOptions(model_asset_path=model_path)
        options    = vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.detector       = vision.HandLandmarker.create_from_options(options)
        self.prev_positions = []
        self.smooth_window  = 5
        self.last_time      = {}

    def _get_model(self):
        import urllib.request, os
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            print("Downloading hand landmark model (~8 MB)...")
            url = (
                "https://storage.googleapis.com/mediapipe-models/"
                "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            )
            urllib.request.urlretrieve(url, model_path)
            print("Download complete.")
        return model_path

    def process(self, frame):
        h, w = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        gesture   = None
        landmarks = []

        if result.hand_landmarks:
            hand_lms  = result.hand_landmarks[0]
            landmarks = [(lm.x, lm.y) for lm in hand_lms]
            self._draw_landmarks(frame, hand_lms, w, h)
            gesture = self._classify(landmarks)

        return frame, gesture, landmarks

    def _draw_landmarks(self, frame, hand_lms, w, h):
        connections = [
            (0,1),(1,2),(2,3),(3,4),
            (0,5),(5,6),(6,7),(7,8),
            (5,9),(9,10),(10,11),(11,12),
            (9,13),(13,14),(14,15),(15,16),
            (13,17),(17,18),(18,19),(19,20),(0,17),
        ]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for a, b in connections:
            cv2.line(frame, pts[a], pts[b], (0, 200, 100), 2)
        for pt in pts:
            cv2.circle(frame, pt, 4, (0, 255, 180), -1)
        for tip in [4, 8, 12, 16, 20]:
            cv2.circle(frame, pts[tip], 8, COL_YELLOW, -1)

    def _classify(self, lm):
        fingers_up = self._fingers_up(lm)
        num_up     = sum(fingers_up)

        wrist = lm[0]
        self.prev_positions.append(wrist)
        if len(self.prev_positions) > self.smooth_window:
            self.prev_positions.pop(0)

        gesture = None

        if num_up == 5:
            gesture = "OPEN_PALM"
        elif num_up == 0:
            gesture = "FIST"
        elif fingers_up == [0, 1, 0, 0, 0]:
            gesture = "POINT_UP"
        elif fingers_up == [0, 1, 1, 0, 0]:
            gesture = "PEACE"
        elif self._is_pinch(lm):
            gesture = "PINCH"
        elif len(self.prev_positions) >= self.smooth_window:
            dy = self.prev_positions[-1][1] - self.prev_positions[0][1]
            if dy < -0.06:
                gesture = "SWIPE_UP"
            elif dy > 0.06:
                gesture = "SWIPE_DOWN"

        if gesture:
            now  = time.time()
            last = self.last_time.get(gesture, 0)
            if now - last < GESTURE_COOLDOWN:
                gesture = None
            else:
                self.last_time[gesture] = now

        return gesture

    def _fingers_up(self, lm):
        tips = [4, 8, 12, 16, 20]
        pip  = [3, 6, 10, 14, 18]
        fingers = [1 if lm[tips[0]][0] < lm[pip[0]][0] else 0]
        for i in range(1, 5):
            fingers.append(1 if lm[tips[i]][1] < lm[pip[i]][1] else 0)
        return fingers

    def _is_pinch(self, lm):
        tx, ty = lm[4]
        ix, iy = lm[8]
        return np.hypot(tx - ix, ty - iy) < 0.06


# ──────────────────────────────────────────────
#  Overlay
# ──────────────────────────────────────────────
def draw_overlay(frame, gesture, landmarks, fps):
    h, w = frame.shape[:2]

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL_GREEN, 2)

    if gesture:
        action = GESTURE_ACTIONS.get(gesture, gesture)
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, action, (20, h - 30),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, COL_YELLOW, 2)
        cv2.putText(frame, gesture, (20, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_GREEN, 1)
    else:
        cv2.rectangle(frame, (0, h - 40), (w, h), (20, 20, 20), -1)
        cv2.putText(frame, "No gesture — show your hand to the webcam",
                    (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

    guide = [
        "GESTURES:",
        "Swipe Up    -> Prev Slice",
        "Swipe Down  -> Next Slice",
        "Pinch       -> Zoom",
        "Open Palm   -> Reset",
        "Fist        -> AI Mode",
        "Point Up    -> Scroll Up",
        "Peace sign  -> Scroll Down",
    ]
    x0 = w - 290
    for i, line in enumerate(guide):
        col = COL_BLUE if i == 0 else (180, 180, 180)
        cv2.putText(frame, line, (x0, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, col, 1)

    if landmarks:
        tx, ty = landmarks[4]
        ix, iy = landmarks[8]
        pt1  = (int(tx * w), int(ty * h))
        pt2  = (int(ix * w), int(iy * h))
        dist = np.hypot(tx - ix, ty - iy)
        col  = COL_RED if dist < 0.06 else COL_WHITE
        cv2.line(frame, pt1, pt2, col, 2)
        mid  = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(frame, f"{dist:.2f}", mid,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

    return frame


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    print("\nInitialising gesture detector...")
    detector = GestureDetector()
    print("Ready!\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time   = time.time()
    gesture_log = []

    print("Show your hand to the webcam.")
    print("Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame, gesture, landmarks = detector.process(frame)

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-9)
        prev_time = now

        if gesture:
            action = GESTURE_ACTIONS.get(gesture, gesture)
            print(f"  [{gesture}]  ->  {action}")
            gesture_log.append((gesture, action))

        frame = draw_overlay(frame, gesture, landmarks, fps)
        cv2.imshow("Gesture Recognition — Phase 2  (Q to quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n── Session Summary ──────────────")
    print(f"Total gestures: {len(gesture_log)}")
    for g, a in gesture_log[-10:]:
        print(f"  {g:15s} -> {a}")


if __name__ == "__main__":
    main()