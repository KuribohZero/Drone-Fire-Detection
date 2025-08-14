import cv2
import os
import onnxruntime as ort
import numpy as np

MODEL_PATH = "models/yolo11n.onnx"
CONF_THRESHOLD = 0.3
FIRE_CLASS_ID = 0  # Change if needed

# ====================
# TEST MODEL FUNCTION
# ====================
def test_model(model_path):
    if not os.path.exists(model_path):
        print("❌ Model file not found:", model_path)
        return False
    try:
        print("✅ Loading model for test...")
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"📏 Model input shape: {input_shape}")

        # Dummy image (black)
        dummy = np.zeros((input_shape[2], input_shape[3], 3), dtype=np.float32)
        dummy = np.transpose(dummy, (2, 0, 1))  # CHW
        dummy = np.expand_dims(dummy, axis=0)   # Batch
        outputs = session.run(None, {input_name: dummy})
        print("✅ Model ran inference successfully!")
        return True
    except Exception as e:
        print("❌ Model test failed:", e)
        return False

# Run model test first
if not test_model(MODEL_PATH):
    exit()

# ====================
# FIRE DETECTION
# ====================
print("✅ Model found, loading on CPU...")
session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape

def preprocess(frame):
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

print("📷 Opening webcam...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

print("🎥 Starting CPU fire detection (press 'q' to quit)...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_input = preprocess(frame)
    outputs = session.run(None, {input_name: img_input})
    detections = postprocess(outputs, frame.shape)

    fire_detected = False
    for (x1, y1, x2, y2, conf) in detections:
        fire_detected = True
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"FIRE {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if fire_detected:
        cv2.putText(frame, "🔥 FIRE DETECTED!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "✅ No Fire Detected", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Fire Detection (CPU)", frame)

    # EXIT PROGRAM
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting program...")
        break

cap.release()
cv2.destroyAllWindows()
