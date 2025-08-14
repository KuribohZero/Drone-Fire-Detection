FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# System dependencies for OpenCV and building packages
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 ffmpeg git curl wget unzip \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Ensure Ultralytics config folder is writable
RUN mkdir -p /root/.config/Ultralytics && chmod -R 777 /root/.config

# Install pip packages
# Using python -m pip ensures it installs in the right env
RUN python -m pip install --upgrade pip

# Install required Python packages explicitly
RUN python -m pip install ultralytics opencv-python onnxruntime numpy

# Copy project code
COPY . .

COPY models/yolo11n.onnx /app/models/yolo11n.onnx

# Env variable for YOLO config
ENV YOLO_CONFIG_DIR=/root/.config/Ultralytics

CMD ["python", "main.py"]
