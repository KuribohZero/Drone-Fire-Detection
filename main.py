import cv2
from detectors.fire_detector import FireDetector

# Use CPU locally
fire = FireDetector("models/yolo11n.onnx", conf_threshold=0.3, use_npu=False)

cap = cv2.VideoCapture(0)  # webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = fire.detect(frame)

    # Draw boxes
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{det['class_id']} {det['confidence']:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
