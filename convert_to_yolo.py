import os
import cv2

dataset_dir = 'dataset/train'
image_output_dir = 'images/train'
label_output_dir = 'labels/train'

os.makedirs(image_output_dir, exist_ok=True)
os.makedirs(label_output_dir, exist_ok=True)

class_names = sorted(os.listdir(dataset_dir))
class_map = {cls: i for i, cls in enumerate(class_names)}

for cls in class_names:
    img_dir = os.path.join(dataset_dir, cls)
    for img_name in os.listdir(img_dir):
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # full image as bounding box
        label = f"{class_map[cls]} 0.5 0.5 1.0 1.0\n"
        out_img = os.path.join(image_output_dir, f"{cls}_{img_name}")
        out_label = os.path.join(label_output_dir, f"{cls}_{img_name.split('.')[0]}.txt")

        cv2.imwrite(out_img, img)
        with open(out_label, 'w') as f:
            f.write(label)

# save class names
with open('classes.txt', 'w') as f:
    f.write('\n'.join(class_names))

print("✅ Dataset converted to YOLO format.")
