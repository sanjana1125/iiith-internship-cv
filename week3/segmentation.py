from ultralytics import YOLO
import cv2
import os

# ── 1. Load model ──────────────────────────────────────────────────────────────
# Use a real YOLOv8 seg model (yolo26 doesn't exist — pick the right size):
#   yolov8n-seg.pt  ← nano  (fastest, least accurate)
#   yolov8s-seg.pt  ← small
#   yolov8m-seg.pt  ← medium
#   yolov8l-seg.pt  ← large
#   yolov8x-seg.pt  ← xlarge (slowest, most accurate)
model = YOLO("yolov8x-seg.pt")   # auto-downloads on first run

# ── 2. Run inference on your video ─────────────────────────────────────────────
VIDEO_PATH = "video.mp4"          # path to your uploaded video
OUTPUT_DIR = "runs/segment/video_result"

results = model.predict(
    source=VIDEO_PATH,
    save=True,                    # saves annotated output video
    save_txt=True,                # saves per-frame label .txt files
    save_conf=True,               # includes confidence scores in labels
    project="runs/segment",       # root output folder
    name="video_result",          # subfolder name
    conf=0.10,                    # confidence threshold (lower = detect more)
    iou=0.35,                     # NMS IoU threshold
    augment = True,
    imgsz=1280,  
    classes=None,                 # None = segment ALL classes
    retina_masks=True,            # higher quality masks (slower)
    stream=True,                  # memory-efficient streaming for video
    lr0=0.001,            # lower initial LR for stability
    lrf=0.01,             # final LR ratio
    mosaic=1.0,           # mosaic augmentation (helps rare classes)
    mixup=0.1,            # mixup augmentation
    copy_paste=0.3,       # copy-paste augmentation (great for seg)
    degrees=10.0,         # rotation augmentation
    patience=20,          # early stopping
)

# ── 3. Collect per-frame metrics ───────────────────────────────────────────────
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

frame_ids      = []
obj_counts     = []
avg_confs      = []
class_counts   = defaultdict(list)   # class_name → [count per frame]

for i, result in enumerate(results):
    boxes = result.boxes
    n_objs = len(boxes) if boxes is not None else 0
    confs  = boxes.conf.cpu().numpy() if boxes is not None and len(boxes) else np.array([0.0])

    frame_ids.append(i)
    obj_counts.append(n_objs)
    avg_confs.append(float(confs.mean()))

    # Per-class counts
    if boxes is not None and len(boxes):
        names = result.names
        for cls_id in boxes.cls.cpu().numpy().astype(int):
            class_counts[names[cls_id]].append(i)

# ── 4. Plot & save metrics graphs ──────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle("YOLOv8 Segmentation — Video Metrics", fontsize=14, fontweight="bold")

# Graph 1: Objects detected per frame
axes[0].plot(frame_ids, obj_counts, color="steelblue", linewidth=1.5)
axes[0].fill_between(frame_ids, obj_counts, alpha=0.2, color="steelblue")
axes[0].set_title("Objects Detected per Frame")
axes[0].set_xlabel("Frame")
axes[0].set_ylabel("Count")
axes[0].grid(True, alpha=0.3)

# Graph 2: Average confidence per frame
axes[1].plot(frame_ids, avg_confs, color="darkorange", linewidth=1.5)
axes[1].fill_between(frame_ids, avg_confs, alpha=0.2, color="darkorange")
axes[1].set_title("Average Confidence per Frame")
axes[1].set_xlabel("Frame")
axes[1].set_ylabel("Confidence")
axes[1].set_ylim(0, 1)
axes[1].grid(True, alpha=0.3)

# Graph 3: Class frequency (bar chart across whole video)
if class_counts:
    cls_names = list(class_counts.keys())
    cls_freqs  = [len(v) for v in class_counts.values()]
    axes[2].bar(cls_names, cls_freqs, color="mediumseagreen", edgecolor="black")
    axes[2].set_title("Total Detections per Class (all frames)")
    axes[2].set_xlabel("Class")
    axes[2].set_ylabel("Detection Count")
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(True, alpha=0.3, axis='y')
else:
    axes[2].text(0.5, 0.5, "No detections", ha='center', va='center')

plt.tight_layout()
graph_path = os.path.join(OUTPUT_DIR, "metrics.png")
plt.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"\n✅ Metrics saved to: {graph_path}")
print(f"✅ Annotated video saved in: {OUTPUT_DIR}/")