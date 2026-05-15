import os

COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck',
    'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
    'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard',
    'sports ball','kite','baseball bat','baseball glove','skateboard','surfboard',
    'tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza',
    'donut','cake','chair','couch','potted plant','bed','dining table','toilet',
    'tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven',
    'toaster','sink','refrigerator','book','clock','vase','scissors',
    'teddy bear','hair drier','toothbrush'
]

# ── Step 1: Scan all label files ──────────────────────────
class_ids_found = set()
for split in ['train', 'val']:
    label_dir = f'dataset/labels/{split}'
    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith('.txt'):
            continue
        with open(os.path.join(label_dir, txt_file), 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_ids_found.add(int(line.split()[0]))

class_ids_found = sorted(class_ids_found)

# ── Step 2: Map IDs to names ──────────────────────────────
detected_names = [COCO_CLASSES[i] for i in class_ids_found]

print("Classes found:", class_ids_found)
print("Class names  :", detected_names)
print("Total classes:", len(detected_names))

# ── Step 3: Remap label files to 0-indexed IDs ────────────
# IMPORTANT: YOLO needs class IDs starting from 0
# e.g. if you found IDs [2, 7] they must become [0, 1]
id_remap = {old: new for new, old in enumerate(class_ids_found)}
# id_remap = {2: 0, 7: 1}  ← example

for split in ['train', 'val']:
    label_dir = f'dataset/labels/{split}'
    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith('.txt'):
            continue
        filepath = os.path.join(label_dir, txt_file)
        with open(filepath, 'r') as f:
            lines = f.readlines()
        with open(filepath, 'w') as f:
            for line in lines:
                parts = line.strip().split()
                if parts:
                    parts[0] = str(id_remap[int(parts[0])])
                    f.write(' '.join(parts) + '\n')

print("\nLabel files remapped to 0-indexed class IDs")

# ── Step 4: Write data.yaml automatically ─────────────────
yaml_content = f"""path: ./dataset
train: images/train
val:   images/val
test:  images/test

nc: {len(detected_names)}
names: {detected_names}
"""
with open('dataset/data.yaml', 'w') as f:
    f.write(yaml_content)

print("data.yaml written successfully!")
print("\nFinal data.yaml contents:")
print(yaml_content)