"""
05WT5# – Create a 1-minute video from detected images with optional music.
Uses OpenCV + ffmpeg only. No moviepy needed.

Usage
─────
    # Without music:
    python3 task5_create_video.py

    # With music:
    python3 task5_create_video.py --music /path/to/song.mp3
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# ── CONFIG ─────────────────────────────────────────────────────────────────────
DEFAULT_DETECT_DIR = '/Users/sanjana/Sanjana/internship/week5/task4'
DEFAULT_OUTPUT     = '/Users/sanjana/Sanjana/internship/week5/task5/final_video.mp4'
DEFAULT_FPS        = 10
DEFAULT_DURATION   = 0.15   # seconds per image → 399 × 0.15 ≈ 60s
# ──────────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--detect-dir', default=DEFAULT_DETECT_DIR)
    p.add_argument('--output',     default=DEFAULT_OUTPUT)
    p.add_argument('--fps',        type=int,   default=DEFAULT_FPS)
    p.add_argument('--duration',   type=float, default=DEFAULT_DURATION)
    p.add_argument('--music',      default=None)
    return p.parse_args()


def collect_images(detect_dir: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    return sorted([p for p in detect_dir.iterdir()
                   if p.suffix.lower() in exts])


def main():
    args = parse_args()

    detect_dir = Path(args.detect_dir)
    out_path   = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not detect_dir.exists():
        print(f"ERROR: Folder not found: {detect_dir}")
        sys.exit(1)

    image_paths = collect_images(detect_dir)
    if not image_paths:
        print(f"ERROR: No images found in {detect_dir}")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    first = cv2.imread(str(image_paths[0]))
    H, W  = first.shape[:2]
    frames_per_image = max(1, int(args.duration * args.fps))
    total_duration   = len(image_paths) * args.duration

    print(f"Frame size  : {W} x {H}")
    print(f"FPS         : {args.fps}")
    print(f"Duration    : ~{total_duration:.0f} seconds")

    # ── Write video ────────────────────────────────────────────────────────────
    silent_path = out_path.parent / ('_silent_' + out_path.name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(silent_path), fourcc, args.fps, (W, H))

    print("Writing frames ...")
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        for _ in range(frames_per_image):
            writer.write(img)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(image_paths)} images done ...")

    writer.release()

    # ── Re-encode / add music with ffmpeg ──────────────────────────────────────
    if args.music and Path(args.music).exists():
        print(f"Adding music: {Path(args.music).name}")
        cmd = [
            'ffmpeg', '-y',
            '-i', str(silent_path),
            '-i', str(args.music),
            '-map', '0:v', '-map', '1:a',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-pix_fmt', 'yuv420p',
            '-shortest', '-af', 'volume=0.4',
            str(out_path)
        ]
    else:
        print("Re-encoding with ffmpeg ...")
        cmd = [
            'ffmpeg', '-y',
            '-i', str(silent_path),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(out_path)
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    silent_path.unlink(missing_ok=True)

    if result.returncode == 0:
        print(f"\n✔  Video created!")
        print(f"   Output   : {out_path}")
        print(f"   Duration : ~{total_duration:.0f}s")
        print(f"   FPS      : {args.fps}")
    else:
        print(f"ERROR: ffmpeg failed:\n{result.stderr[-500:]}")


if __name__ == '__main__':
    main()
