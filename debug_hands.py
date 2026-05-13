import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import urllib.request, os

model_path = "hand_landmarker.task"
if not os.path.exists(model_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        model_path
    )

base_opts = mp_python.BaseOptions(model_asset_path=model_path)
options   = vision.HandLandmarkerOptions(
    base_options=base_opts, num_hands=2,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

print("Show BOTH hands - watching for 5 seconds...")
import time
start = time.time()
while time.time() - start < 15:
    ret, frame = cap.read()
    if not ret: continue
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if result.hand_landmarks and result.handedness:
        for i, hand_lms in enumerate(result.hand_landmarks):
            h = result.handedness[i][0].category_name
            score = result.handedness[i][0].score
            print(f"  Detected: '{h}' (score={score:.2f}) — show LEFT fist + RIGHT palm")
    cv2.imshow("debug", frame)
    if cv2.waitKey(1) == 27: break

cap.release()
cv2.destroyAllWindows()