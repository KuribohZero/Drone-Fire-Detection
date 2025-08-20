import cv2
import os
import torch
from ultralytics import YOLO
import numpy as np

# ====================
# CONFIG
# ====================
MODEL_PATH = "models/best_nano_111.pt"  # your YOLO11n .pt model
TEST_IMAGE_PATH = "Test_images/campfire.jpg"       # change to your test image
CONF_THRESHOLD = 0.3

# ====================
# DEVICE SETUP
# ====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚡ Using device: {device}")

# ====================
# LOAD MODEL
# ====================
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found:", MODEL_PATH)
    exit()

print("✅ Loading YOLO model...")
model = YOLO(MODEL_PATH)  # YOLO class handles .pt models
model.fuse()  # optional: fuse Conv+BN layers for speed

# ====================
# TEST SINGLE IMAGE
# ====================
if os.path.exists(TEST_IMAGE_PATH):
    print(f"🖼️ Testing image: {TEST_IMAGE_PATH}")
    results = model.predict(TEST_IMAGE_PATH, conf=CONF_THRESHOLD, device=device)
    img = cv2.imread(TEST_IMAGE_PATH)
    
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()  # xyxy boxes
        scores = r.boxes.conf.cpu().numpy() # confidence
        class_ids = r.boxes.cls.cpu().numpy() # class indices

        fire_detected = False
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if int(cls_id) == 0:  # assuming fire class is 0
                fire_detected = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, f"FIRE {score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if fire_detected:
            cv2.putText(img, "🔥 FIRE DETECTED!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        else:
            cv2.putText(img, "✅ No Fire Detected", (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)
    cv2.imshow("Test Image Fire Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("⚠️ Test image not found, skipping test image detection")

# ====================
# WEBCAM FIRE DETECTION
# ====================
print("📷 Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🎥 Starting live fire detection. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_THRESHOLD, device=device)
    fire_detected = False
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        class_ids = r.boxes.cls.cpu().numpy()
        for box, score, cls_id in zip(boxes, scores, class_ids):
            if int(cls_id) == 0:
                fire_detected = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
                cv2.putText(frame, f"FIRE {score:.2f}", (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

    if fire_detected:
        cv2.putText(frame, "🔥 FIRE DETECTED!", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
    else:
        cv2.putText(frame, "✅ No Fire Detected", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3)

    cv2.imshow("Live Fire Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
