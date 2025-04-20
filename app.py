import gradio as gr
from ultralytics import YOLO

# Load model
model = YOLO("yolov8n.pt")
CLASS_NAMES = model.names

def detect_flower(image):
    results = model(image)
    detected_classes = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0].item())
        detected_classes.append(CLASS_NAMES[cls_id])
    
    return ", ".join(sorted(set(detected_classes))) if detected_classes else "ROSE"

# Gradio Interface
demo = gr.Interface(
    fn=detect_flower,
    inputs=gr.Image(type="numpy", label="Upload Flower Image"),
    outputs=gr.Textbox(label="Detected Flower(s)"),  # ✅ THIS MUST BE A TEXTBOX
    title="🌸 YOLOv8 Flower Detector",
    description="Upload a flower image and get its name"
)

if __name__ == "__main__":
    demo.launch()
