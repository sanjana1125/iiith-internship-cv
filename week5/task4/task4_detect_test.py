"""
05WT4# – Run inference with the newly trained weights on test images.

What this script does
─────────────────────
1. Loads best_saved.pt produced by Task 3.
2. Runs detection on every image inside the test folder.
3. Saves each output image WITH bounding boxes drawn into
   runs/vehicles_v1/detect_test/
4. Saves a detections.csv log with all detection details.

Prerequisites
─────────────
    pip install ultralytics opencv-python

Usage
─────
    python3 task4_detect_test.py

Outputs
───────
    runs/vehicles_v1/detect_test/
        ├── image_XXXX.jpg    ← annotated images with bounding boxes
        └── detections.csv    ← per-detection log
"""

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS  = '/Users/sanjana/Sanjana/internship/week5/task3/runs/detect/runs/vehicles_v1/weights/best.pt'
DEFAULT_TEST_DIR = '/Users/sanjana/Sanjana/internship/week5/task3/dataset/images_scaled/test_scaled'
DEFAULT_OUT_DIR  = '/Users/sanjana/Sanjana/internship/week5/task3/runs/detect/runs/vehicles_v1/detect_test'

CONF_THRESHOLD   = 0.25
IMG_SIZE         = 384
CLASS_NAMES      = ['bicycle', 'car', 'person']
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description='YOLOv8 inference on test images')
    p.add_argument('--weights',  default=DEFAULT_WEIGHTS)
    p.add_argument('--test-dir', default=DEFAULT_TEST_DIR)
    p.add_argument('--output',   default=DEFAULT_OUT_DIR)
    p.add_argument('--conf',     type=float, default=CONF_THRESHOLD)
    return p.parse_args()


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    test_dir     = Path(args.test_dir)
    out_dir      = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Validate inputs ────────────────────────────────────────────────────────
    if not weights_path.exists():
        # Try alternate common paths
        alternates = [
            Path('/Users/sanjana/Sanjana/internship/week5/task3/runs/detect/runs/vehicles_v1/weights/best.pt'),
            Path('/Users/sanjana/Sanjana/internship/week5/task3/runs/vehicles_v1/weights/best.pt'),
        ]
        for alt in alternates:
            if alt.exists():
                weights_path = alt
                print(f"Using weights: {weights_path}")
                break
        else:
            raise FileNotFoundError(
                f"Weights not found. Run task3_train_model.py first.\n"
                f"Then update DEFAULT_WEIGHTS in this script."
            )

    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test image directory not found: {test_dir}\n"
            "Update DEFAULT_TEST_DIR to your test folder."
        )

    # ── Load model ─────────────────────────────────────────────────────────────
    print(f"Loading weights: {weights_path}")
    model = YOLO(str(weights_path))

    # ── Collect test images ────────────────────────────────────────────────────
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in extensions:
        image_paths.extend(sorted(test_dir.glob(ext)))

    if not image_paths:
        raise RuntimeError(f"No images found in {test_dir}")

    print(f"Found {len(image_paths)} test images in {test_dir}")
    print(f"Saving annotated outputs to: {out_dir}")
    print("-" * 55)

    # ── Run inference ──────────────────────────────────────────────────────────
    csv_path = out_dir / 'detections.csv'
    total_detections = 0

    import cv2

    with open(csv_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['image', 'class_id', 'class_name',
                         'confidence', 'x1', 'y1', 'x2', 'y2'])

        for img_path in image_paths:
            preds = model.predict(
                source  = str(img_path),
                imgsz   = IMG_SIZE,
                conf    = args.conf,
                save    = False,
                verbose = False,
            )

            result    = preds[0]
            annotated = result.plot()
            save_path = out_dir / img_path.name
            cv2.imwrite(str(save_path), annotated)

            boxes = result.boxes
            n = len(boxes) if boxes is not None else 0
            total_detections += n

            if boxes is not None and n > 0:
                for box in boxes:
                    cls_id   = int(box.cls[0])
                    cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
                    conf     = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    writer.writerow([img_path.name, cls_id, cls_name,
                                     f'{conf:.4f}',
                                     f'{x1:.1f}', f'{y1:.1f}',
                                     f'{x2:.1f}', f'{y2:.1f}'])

            status = f"{n} detection(s)" if n else "no detections"
            print(f"  {img_path.name:<35} → {status}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print("-" * 55)
    print(f"\n✔  Inference complete!")
    print(f"   Images processed : {len(image_paths)}")
    print(f"   Total detections : {total_detections}")
    print(f"   Annotated images : {out_dir}/")
    print(f"   Detection log    : {csv_path}")


if __name__ == '__main__':
    main()
