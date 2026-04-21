from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")  # classification model

model.train(
    data="dataset",
    epochs=10,
    imgsz=224
)