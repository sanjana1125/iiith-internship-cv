import os, shutil, random

frames = sorted(os.listdir('frames'))
random.shuffle(frames)

splits = {'train': frames[:100], 'val': frames[100:140], 'test': frames[140:]}

for split, files in splits.items():
    os.makedirs(f'dataset/images/{split}', exist_ok=True)
    for f in files:
        shutil.copy(f'frames/{f}', f'dataset/images/{split}/{f}')