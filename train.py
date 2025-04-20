from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # or yolov8s.pt for slightly bigger

model.train(
    data="data.yaml",
    epochs=20,
    imgsz=224,
    batch=16
)
