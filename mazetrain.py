#!/usr/bin/env python3
"""
train.py — YOLO training script for start/goal dot detection on synthetic maze data.

Usage:
  python train.py --data /path/to/maze_yolo_dataset/data.yaml
  # or point to the dataset root (auto-finds data.yaml):
  python train.py --data /path/to/maze_yolo_dataset

  # optional flags:
  python train.py --model yolov8n.pt --epochs 50 --imgsz 640 --batch 16 --device 0
"""

import argparse
import sys
from pathlib import Path

# Ultralytics >=8.0.0
try:
    from ultralytics import YOLO
except ImportError as e:
    print("Ultralytics not found. Install with:\n  pip install ultralytics\n")
    raise


def resolve_data_yaml(data_arg: str) -> str:
    """
    Accepts either:
      - a path to data.yaml, or
      - a path to the dataset root that contains data.yaml
    Returns absolute path to data.yaml.
    """
    p = Path(data_arg).expanduser().resolve()
    if p.is_dir():
        yaml_path = p / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(f"Could not find data.yaml in: {p}")
        return str(yaml_path)
    elif p.is_file():
        if p.name.lower() != "data.yaml":
            raise ValueError(f"If passing a file, it must be data.yaml. Got: {p.name}")
        return str(p)
    else:
        raise FileNotFoundError(f"Path does not exist: {p}")


def main():
    ap = argparse.ArgumentParser(description="Train YOLO on maze start/goal dots.")
    ap.add_argument(
        "--data",
        required=True,
        help="Path to data.yaml OR the dataset root containing data.yaml.",
    )
    ap.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base model to start from (e.g., yolov8n.pt, yolov8s.pt, path to .pt).",
    )
    ap.add_argument(
        "--imgsz", type=int, default=640, help="Training image size (square)."
    )
    ap.add_argument("--epochs", type=int, default=50, help="Epochs.")
    ap.add_argument("--batch", type=int, default=16, help="Batch size.")
    ap.add_argument("--workers", type=int, default=8, help="Dataloader workers.")
    ap.add_argument(
        "--device",
        default=None,
        help="Device string for PyTorch (e.g., '0' for GPU0, 'cpu' for CPU).",
    )
    ap.add_argument(
        "--project", default="runs/maze_yolo", help="Root folder for training runs."
    )
    ap.add_argument("--name", default="exp", help="Run name (subfolder under project).")
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume the most recent run in this project/name.",
    )
    ap.add_argument(
        "--predict_samples",
        type=int,
        default=12,
        help="After training, run predictions on N validation images and save.",
    )
    ap.add_argument(
        "--export",
        default="onnx",
        choices=[
            "none",
            "onnx",
            "torchscript",
            "openvino",
            "engine",
            "coreml",
            "tflite",
            "pb",
            "saved_model",
        ],
        help="Export format after training/val. Use 'none' to skip.",
    )
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)
    print(f"[INFO] Using data.yaml: {data_yaml}")

    # Load model (will download if needed)
    model = YOLO(args.model)

    # Train
    # NOTE: you can add more hyperparams here (lr0, lrf, momentum, weight_decay, augment, hsv_h/s/v, translate, scale, flipud, fliplr, mosaic, mixup, etc.)
    train_results = model.train(
        data=data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        resume=args.resume,
        patience=50,  # early stopping patience
        verbose=True,
    )

    # Validate on the best.pt produced by training
    print("\n[INFO] Running validation on best checkpoint...")
    val_metrics = model.val(
        data=data_yaml,
        imgsz=args.imgsz,
        device=args.device,
        split="val",  # ensure we measure on the validation set
    )
    print(f"[VAL] metrics: {val_metrics}")

    # Optional: run a few predictions on the val set and save images
    if args.predict_samples > 0:
        print(
            f"\n[INFO] Running sample predictions on {args.predict_samples} val images..."
        )
        # If data.yaml follows the standard structure, val images live at the path in data.yaml;
        # Ultralytics handles loading if we pass 'data=' and 'split=' OR we can pass a folder/glob.
        # Here we call predict on the 'val' split directly.
        model.predict(
            data=data_yaml,
            imgsz=args.imgsz,
            device=args.device,
            split="val",
            max_det=10,
            save=True,
            save_txt=False,
            save_conf=True,
            conf=0.25,
            iou=0.45,
            project=args.project,
            name=f"{args.name}_val_preds",
            exist_ok=True,
            stream=False,  # set True to iterate, False saves all in one go
            vid_stride=1,
        )

    # Optional: export model
    if args.export and args.export.lower() != "none":
        print(f"\n[INFO] Exporting model to {args.export}...")
        export_file = model.export(format=args.export)  # returns path to exported file
        print(f"[INFO] Exported: {export_file}")

    print("\n[DONE] Training complete.")
    print(f" Run folder: {Path(args.project)/args.name}")


if __name__ == "__main__":
    sys.exit(main())
