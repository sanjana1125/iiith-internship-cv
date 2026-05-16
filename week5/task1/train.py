import os

for split in ['train', 'val']:
    image_dir = f'dataset/images/{split}'  
    with open(f'dataset/{split}.txt', 'w') as f:  
        for img in sorted(os.listdir(image_dir)):  
            f.write(f'./images/{split}/{img}\n') 