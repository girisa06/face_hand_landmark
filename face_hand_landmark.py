import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download hand model
hand_model = 'hand_landmarker.task'
if not os.path.exists(hand_model):
    print("Downloading hand model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        hand_model)
    print("Hand model done.")

# Download face model
face_model = 'face_landmarker.task'
if not os.path.exists(face_model):
    print("Downloading face model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        face_model)
    print("Face model done.")

# Hand skeleton connections
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# Face contour connections (simplified, key edges only)
FACE_CONNECTIONS = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
    # Eyes
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),
    (362,398),(398,384),(384,385),(385,386),(386,387),(387,388),(388,466),(466,263),
    # Lips
    (61,185),(185,40),(40,39),(39,37),(37,0),(0,267),(267,269),(269,270),(270,409),(409,291),
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),
]

# Setup hand detector
hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=hand_model),
    num_hands=2
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# Setup face detector
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=face_model),
    num_faces=2
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # --- Hands ---
    hand_result = hand_detector.detect(mp_image)
    if hand_result.hand_landmarks:
        for hand in hand_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (255, 255, 255), 2)
            for cx, cy in pts:
                cv2.circle(frame, (cx, cy), 6, (0, 255, 0), -1)

        cv2.putText(frame, f'Hands: {len(hand_result.hand_landmarks)}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # --- Face ---
    face_result = face_detector.detect(mp_image)
    if face_result.face_landmarks:
        for face in face_result.face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face]
            for a, b in FACE_CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], (255, 255, 0), 1)
            for cx, cy in pts:
                cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)

        cv2.putText(frame, f'Faces: {len(face_result.face_landmarks)}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame, 'Press Q to quit', (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Hand + Face Landmarks', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()