"""
05WT3# – Fine-tune YOLOv8n on the fruit/vegetable dataset.

What this script does
─────────────────────
1. Loads the pretrained YOLOv8n backbone.
2. Fine-tunes it on your scaled dataset for up to 100 epochs.
3. After training, reads the CSV results file produced by Ultralytics
   and prints the epoch at which validation loss (val/box_loss) was
   lowest — i.e. the last epoch before the model started to overfit.
4. Copies the best weights to  runs/vegetables_v1/weights/best_saved.pt
   so they are clearly separated from the auto-saved checkpoints.

Prerequisites
─────────────
    pip install ultralytics

Usage
─────
    python task3_train_model.py

Outputs (inside  runs/vegetables_v1/)
──────────────────────────────────────
    weights/best.pt          ← Ultralytics "best" checkpoint
    weights/best_saved.pt    ← your explicitly saved copy
    weights/last.pt          ← weights after the final epoch
    results.csv              ← per-epoch metrics (loss, mAP, …)
    *.png                    ← loss-curve and metric plots
"""

import shutil
from pathlib import Path

from ultralytics import YOLO
import pandas as pd   # already installed with ultralytics

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Update DATA_YAML to match the absolute path on your machine.
DATA_YAML   = '/Users/sanjana/Sanjana/internship/week5/task3/dataset/data.yaml'
MODEL       = 'yolov8n.pt'      # pretrained COCO backbone
EPOCHS      = 100
IMGSZ       = 384               # matches your scaled images
BATCH       = 16
PROJECT     = 'runs'
RUN_NAME    = 'vegetables_v1'
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Load pretrained model ──────────────────────────────────────────────
    model = YOLO(MODEL)

    # ── 2. Train ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("Starting fine-tuning …")
    print("=" * 60)

    results = model.train(
        data        = DATA_YAML,
        epochs      = EPOCHS,
        imgsz       = IMGSZ,
        batch       = BATCH,
        project     = PROJECT,
        name        = RUN_NAME,
        plots       = True,         # saves loss-curve PNGs
        save        = True,
        save_period = 10,           # checkpoint every 10 epochs
        patience    = 20,           # early-stop if no val improvement for 20 epochs
        verbose     = True,
        cache = False,
    )

    run_dir = Path(PROJECT) / RUN_NAME

    # ── 3. Analyse training results ───────────────────────────────────────────
    results_csv = run_dir / 'results.csv'
    if results_csv.exists():
        df = pd.read_csv(results_csv)
        df.columns = df.columns.str.strip()   # remove accidental whitespace

        # Ultralytics stores val box-loss under the column 'val/box_loss'
        val_loss_col = 'val/box_loss'
        if val_loss_col in df.columns:
            best_epoch = int(df[val_loss_col].idxmin()) + 1   # 1-indexed
            min_val_loss = df[val_loss_col].min()

            print("\n" + "=" * 60)
            print("OVERFITTING ANALYSIS")
            print("=" * 60)
            print(f"  Lowest val/box_loss : {min_val_loss:.4f}  @ epoch {best_epoch}")
            print(f"  Total epochs run    : {len(df)}")
            if len(df) > best_epoch:
                print(f"  ⚠  Validation loss started rising after epoch {best_epoch}.")
                print(f"     Model likely began overfitting from epoch {best_epoch + 1} onward.")
            else:
                print("  ✔  Training stopped before significant overfitting was detected.")
            print("=" * 60)
        else:
            print(f"[WARN] Column '{val_loss_col}' not found in results.csv.")
            print(f"       Available columns: {list(df.columns)}")
    else:
        print("[WARN] results.csv not found – skipping overfitting analysis.")

    # ── 4. Save best weights separately ──────────────────────────────────────
    best_src  = run_dir / 'weights' / 'best.pt'
    best_dst  = run_dir / 'weights' / 'best_saved.pt'

    if best_src.exists():
        shutil.copy(best_src, best_dst)
        print(f"\n✔  Best weights saved separately → {best_dst}")
    else:
        print("[WARN] best.pt not found; training may have been interrupted.")

    print(f"\nTraining complete!")
    print(f"  Best weights  : {best_src}")
    print(f"  Saved copy    : {best_dst}")
    print(f"  Last weights  : {run_dir / 'weights' / 'last.pt'}")
    print(f"  Loss plots    : {run_dir}/*.png")


if __name__ == '__main__':
    main()
