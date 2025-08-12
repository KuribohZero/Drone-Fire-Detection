from ultralytics import YOLO

visualModel = YOLO("yolov11n.pt")  #Will train custom model for visual camera
theramlModel = YOLO("yolov11n.pt")  #Will train custom model for thermal camera

