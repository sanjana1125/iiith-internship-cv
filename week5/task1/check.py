import os

for split in ['train', 'val']:
    imgs   = len([f for f in os.listdir(f'dataset/images/{split}') if f.endswith(('.jpg','.png'))])
    labels = len([f for f in os.listdir(f'dataset/labels/{split}') if f.endswith('.txt')])
    print(f"{split}: {imgs} images, {labels} label files")
    if imgs != labels:
        print(f"  WARNING — mismatch! Check missing files")
    else:
        print(f"  OK — counts match")
