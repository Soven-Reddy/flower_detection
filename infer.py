from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Load the YOLOv8 model

# Replace this with your test image
results = model("images/train/daisy_100.jpg", save=True, imgsz=224)
results.show()
