# Run this separately to see what classes your images contain
from ultralytics import YOLO
import os

model = YOLO('yolov8n.pt')

found = set()
for split in ['train', 'val']:
    for f in os.listdir(f'dataset/images/{split}'):
        if f.endswith(('.jpg', '.png')):
            r = model(f'dataset/images/{split}/{f}', verbose=False)
            for box in r[0].boxes:
                found.add((int(box.cls), model.names[int(box.cls)]))

print("Classes found:")
for cls_id, name in sorted(found):
    print(f"  {cls_id} → {name}")
