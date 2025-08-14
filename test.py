import cv2
import os
import onnxruntime as ort
import numpy as np

MODEL_PATH = "models/yolo11n.onnx"
TEST_IMAGE = "test_images/campfire.jpg"  # <- put a sample fire image here
CONF_THRESHOLD = 0.3
FIRE_CLASS_ID = 0  # adjust if fire is labeled differently

def preprocess(frame, input_shape):
    img = cv2.resize(frame, (input_shape[2], input_shape[3]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def postprocess(outputs, orig_shape):
    preds = np.squeeze(outputs[0])
    boxes = []
    for pred in preds:
        x_center, y_center, w, h = pred[0:4]
        obj_conf = pred[4]
        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        class_conf = class_scores[class_id]
        score = obj_conf * class_conf
        if class_id == FIRE_CLASS_ID and score >= CONF_THRESHOLD:
            x1 = int((x_center - w / 2) * orig_shape[1])
            y1 = int((y_center - h / 2) * orig_shape[0])
            x2 = int((x_center + w / 2) * orig_shape[1])
            y2 = int((y_center + h / 2) * orig_shape[0])
            boxes.append((x1, y1, x2, y2, score))
    return boxes

# 1️⃣ Check model exists
if not os.path.exists(MODEL_PATH):
    print("❌ Model file not found:", MODEL_PATH)
    exit()

# 2️⃣ Check test image exists
if not os.path.exists(TEST_IMAGE):
    print("❌ Test image not found:", TEST_IMAGE)
    exit()

# 3️⃣ Load model
print("✅ Loading model on CPU...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"📏 Model input shape: {input_shape}")

# 4️⃣ Read & preprocess image
frame = cv2.imread(TEST_IMAGE)
img_input = preprocess(frame, input_shape)

# 5️⃣ Run inference
outputs = session.run(None, {input_name: img_input})

# 6️⃣ Postprocess detections
detections = postprocess(outputs, frame.shape)

# 7️⃣ Draw results
fire_detected = False
for (x1, y1, x2, y2, conf) in detections:
    fire_detected = True
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, f"FIRE {conf:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

if fire_detected:
    print("🔥 FIRE detected in test image!")
    cv2.putText(frame, "🔥 FIRE DETECTED!", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
else:
    print("✅ No fire detected in test image.")
    cv2.putText(frame, "✅ No Fire Detected", (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

# 8️⃣ Show image
cv2.imshow("Fire Model Test", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
