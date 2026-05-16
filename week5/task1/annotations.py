import cv2
import os

# ── CONFIG — change these to match your setup ──────────────────────────
DATASET_DIR  = './dataset'
CLASS_NAMES  = ['car', 'airplane', 'truck', 'cat', 'sports ball', 'apple', 'orange', 'donut', 'cake', 'potted plant', 'vase']   # must match your data.yaml order
COLORS = [
    (0,   0,   255),  # red   → class 0
    (255, 0,   0  ),  # blue  → class 1
    (0,   255, 0  ),  # green → class 2
    (0,   255, 255),  # yellow → class 3
    (255, 0,   255),  # magenta → class 4
    (255, 128, 0  ),  # orange → class 5
]
# ───────────────────────────────────────────────────────────────────────

def annotate_split(split):
    image_dir    = os.path.join(DATASET_DIR, 'images',    split)
    label_dir    = os.path.join(DATASET_DIR, 'labels',    split)
    output_dir   = os.path.join(DATASET_DIR, 'annotated', split)

    os.makedirs(output_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    if not images:
        print(f"  No images found in {image_dir}")
        return

    print(f"\nProcessing {split} — {len(images)} images")

    skipped  = 0
    no_label = 0

    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        img      = cv2.imread(img_path)

        if img is None:
            print(f"  Could not read: {img_name} — skipping")
            skipped += 1
            continue

        h, w = img.shape[:2]

        # Find matching label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            no_label += 1
            # Save image as-is with a note
            cv2.imwrite(os.path.join(output_dir, img_name), img)
            continue

        with open(label_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id   = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            bw       = float(parts[3])
            bh       = float(parts[4])

            # Convert normalised YOLO coords → pixel coords
            x1 = int((x_center - bw / 2) * w)
            y1 = int((y_center - bh / 2) * h)
            x2 = int((x_center + bw / 2) * w)
            y2 = int((y_center + bh / 2) * h)

            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Pick color
            color = COLORS[cls_id % len(COLORS)]

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Draw filled label background
            label_text = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)

            # Draw label text in white
            cv2.putText(img, label_text, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Save annotated image
        out_path = os.path.join(output_dir, img_name)
        cv2.imwrite(out_path, img)

    print(f"  Done — saved to {output_dir}")
    print(f"  Annotated : {len(images) - skipped - no_label}")
    print(f"  No label  : {no_label}  (saved as-is)")
    print(f"  Skipped   : {skipped}  (unreadable)")

# ── Run for both splits ─────────────────────────────────────────────────
annotate_split('train')
annotate_split('val')

print("\nAll done! Annotated images saved to dataset/annotated/")
