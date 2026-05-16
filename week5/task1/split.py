import os
import shutil

# Paths
EXPORTED_LABELS = 'exported/labels'   # where you extracted the zip
DATASET_DIR     = 'dataset'

for split in ['train', 'val']:
    image_dir = os.path.join(DATASET_DIR, 'images', split)
    label_dir = os.path.join(DATASET_DIR, 'labels', split)
    os.makedirs(label_dir, exist_ok=True)

    # Get image names in this split
    images = [os.path.splitext(f)[0] for f in os.listdir(image_dir)
              if f.endswith(('.jpg', '.png', '.jpeg'))]

    matched   = 0
    unmatched = 0

    for img_stem in images:
        src = os.path.join(EXPORTED_LABELS, img_stem + '.txt')
        dst = os.path.join(label_dir, img_stem + '.txt')

        if os.path.exists(src):
            shutil.copy(src, dst)
            matched += 1
        else:
            # Create empty label file if no annotation
            open(dst, 'w').close()
            unmatched += 1

    print(f"{split}: {matched} matched, {unmatched} unmatched")

print("\nDone! Labels sorted into train and val.")
