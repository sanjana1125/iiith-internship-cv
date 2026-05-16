import subprocess
import os

# ── UPDATE THIS PATH TO WHERE YOUR IMAGES ARE ──────────────
DATASET_DIR = '/Users/sanjana/Sanjana/internship/week5/task2/images'
# e.g. if your images are directly in task2/train/, task2/val/
# set DATASET_DIR = '/Users/sanjana/Sanjana/internship/week5/task2'
# ────────────────────────────────────────────────────────────

SCALE       = '384:-1'
OUTPUT_BASE = os.path.join(DATASET_DIR, 'images_scaled')

for split in ['train', 'val', 'test']:
    input_dir  = os.path.join(DATASET_DIR, split)  # try without 'images' subfolder
    output_dir = os.path.join(OUTPUT_BASE, f'{split}_scaled')

    if not os.path.exists(input_dir):
        print(f"Skipping {split} — folder not found: {input_dir}")
        continue

    os.makedirs(output_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith(('.jpg', '.jpeg', '.png'))
    ])

    print(f"\nScaling {split} — {len(images)} images → {output_dir}")

    for img_name in images:
        input_path  = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)

        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-vf', f'scale={SCALE}',
            output_path,
            '-y', '-loglevel', 'error'
        ])
        print(f"  Scaled: {img_name}")

print("\nAll done!")
