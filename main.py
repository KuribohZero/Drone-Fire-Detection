import cv2
from ultralytics import YOLO

# Load ONNX model
model = YOLO("models/best_nano_111.onnx")

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Optional: set frame size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Starting real-time fire and smoke detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run ONNX inference on frame
    results = model.predict(
        source=frame,
        imgsz=448,   # same as export size
        conf=0.1,    # lower confidence threshold to catch faint detections
        device=0
    )

    # Draw bounding boxes
    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            # Determine label based on class
            if int(cls) == 0:
                label = f"Fire {conf:.2f}"
                color = (0, 0, 255)  # Red for fire
            elif int(cls) == 1:
                label = f"Smoke {conf:.2f}"
                color = (0, 255, 255)  # Yellow for smoke
            else:
                label = f"Unknown {conf:.2f}"
                color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display the frame
    cv2.imshow("Real-Time Fire and Smoke Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
