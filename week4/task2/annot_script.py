from ultralytics import YOLO
import os

# Load a pretrained model
model = YOLO('yolov8n.pt')  # nano model, fast

# Folders
image_dir = 'dataset/images/val'
label_dir = 'dataset/labels/val'
os.makedirs(label_dir, exist_ok=True)

# Run inference on every image
for img_name in os.listdir(image_dir):
    if not img_name.endswith(('.jpg', '.png')):
        continue

    img_path = os.path.join(image_dir, img_name)
    results = model(img_path)

    # Save label file
    label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    with open(label_path, 'w') as f:
        for box in results[0].boxes:
            cls = int(box.cls)
            x, y, w, h = box.xywhn[0].tolist()  # normalised xywh
            f.write(f'{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n')

print("Annotation done!")