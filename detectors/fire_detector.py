import numpy as np
from ultralytics import YOLO

class FireDetector:
    def __init__(self, model_path, conf_threshold=0.3, use_npu=False):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        self.use_npu = use_npu

    def detect(self, frame):
        results = self.model.predict(frame, verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()      # shape: (N, 4)
            confs = r.boxes.conf.cpu().numpy()      # shape: (N,)
            classes = r.boxes.cls.cpu().numpy()     # shape: (N,)

            # Loop through each detection instead of doing array-wide if
            for box, conf, cls in zip(boxes, confs, classes):
                if float(conf) >= self.conf_threshold:
                    detections.append({
                        "box": box,
                        "confidence": float(conf),
                        "class_id": int(cls)
                    })

        return detections
