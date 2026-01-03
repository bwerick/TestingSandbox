"""
auto_label_xo_rotations.py

Pseudo-label real images of tic-tac-toe boards using a trained Ultralytics YOLO model (2 classes: 0=X, 1=O).
For each input image, generate four labeled variants at 0°, 90°, 180°, and 270° rotations.

Outputs under --out (default: dataset_real):
  - images/train/*.jpg         (clean images, including rotated variants)
  - labels/train/*.txt         (YOLO txt labels matching images)
  - preview/*.jpg              (annotated previews for quick review)
  - data.yaml                  (points train to images/train)
  - manifest.csv               (filename, angle, num_dets, mean_conf)

Usage:
  python auto_label_xo_rotations.py \
    --model runs/detect/ttt_xo_synth/weights/best.pt \
    --input real_boards \
    --out dataset_real \
    --conf 0.35 --iou 0.5
"""

import argparse
from pathlib import Path
import csv

import cv2
import numpy as np
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def yolo_norm_from_xyxy(x1, y1, x2, y2, W, H):
    """xyxy (abs px) -> YOLO normalized (xc,yc,w,h)."""
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return xc / W, yc / H, w / W, h / H


def draw_box(img, x1, y1, x2, y2, label, conf, color):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)
    cv2.putText(
        img,
        f"{label} {conf:.2f}",
        (int(x1), max(0, int(y1) - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def rotate_image(img, angle):
    """Rotate image by 0/90/180/270 degrees and return rotated image + (new_W,new_H)."""
    if angle == 0:
        rot = img
    elif angle == 90:
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        rot = cv2.rotate(img, cv2.ROTATE_180)
    elif angle == 270:
        rot = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("angle must be one of {0,90,180,270}")
    H, W = rot.shape[:2]
    return rot, W, H


def ensure_dirs(root: Path):
    img_out = root / "images" / "train"
    lbl_out = root / "labels" / "train"
    prev_out = root / "preview"
    for p in (img_out, lbl_out, prev_out):
        p.mkdir(parents=True, exist_ok=True)
    return img_out, lbl_out, prev_out


def write_data_yaml(root: Path):
    (root / "data.yaml").write_text(
        "path: {}\ntrain: images/train\nval: images/train\nnc: 2\nnames: [X, O]\n".format(
            root.resolve()
        )
    )


def process_one_image(
    model,
    src_img_path: Path,
    img_out: Path,
    lbl_out: Path,
    prev_out: Path,
    conf: float,
    iou: float,
):
    """
    For one source image, run YOLO on 4 rotations and save:
      - rotated image
      - YOLO labels
      - annotated preview
    Returns list of (filename, angle, num_dets, mean_conf).
    """
    img = cv2.imread(str(src_img_path))
    if img is None:
        print(f"[!] Could not read {src_img_path}")
        return []

    base = src_img_path.stem
    rows = []

    for angle in (0, 90, 180, 270):
        suffix = "" if angle == 0 else f"_r{angle}"
        rot_img, W, H = rotate_image(img, angle)

        # Run inference on the rotated image directly (simplest & safest)
        res = model.predict(rot_img, conf=conf, iou=iou, verbose=False)[0]

        # Save clean image
        dst_img_path = img_out / f"{base}{suffix}.jpg"
        cv2.imwrite(str(dst_img_path), rot_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Build YOLO labels from detections for this rotation
        lines = []
        anno = rot_img.copy()
        confs = []

        for b in res.boxes:
            cls_id = int(b.cls)
            confv = float(b.conf)
            x1, y1, x2, y2 = map(float, b.xyxy[0])

            # Clamp to image
            x1 = max(0, min(W - 1, x1))
            x2 = max(0, min(W - 1, x2))
            y1 = max(0, min(H - 1, y1))
            y2 = max(0, min(H - 1, y2))

            # YOLO normalized
            xc, yc, w, h = yolo_norm_from_xyxy(x1, y1, x2, y2, W, H)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            confs.append(confv)

            # Preview annotation
            color = (0, 255, 255) if cls_id == 0 else (255, 200, 0)
            label = "X" if cls_id == 0 else "O"
            draw_box(anno, x1, y1, x2, y2, label, confv, color)

        # Labels file (even if empty, write blank file so it's tracked)
        dst_lbl_path = lbl_out / f"{base}{suffix}.txt"
        with open(dst_lbl_path, "w") as f:
            if lines:
                f.write("\n".join(lines))

        # Preview image
        dst_prev_path = prev_out / f"{base}{suffix}.jpg"
        cv2.imwrite(str(dst_prev_path), anno, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        mean_conf = float(np.mean(confs)) if confs else 0.0
        rows.append((dst_img_path.name, angle, len(lines), f"{mean_conf:.4f}"))

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to YOLO weights (best.pt)")
    ap.add_argument("--input", required=True, help="Folder with real board images")
    ap.add_argument("--out", default="dataset_real", help="Output dataset root")
    ap.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.50, help="NMS IoU threshold")
    args = ap.parse_args()

    model = YOLO(args.model)

    input_dir = Path(args.input)
    out_dir = Path(args.out)
    img_out, lbl_out, prev_out = ensure_dirs(out_dir)
    write_data_yaml(out_dir)

    image_paths = [p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    image_paths.sort()
    if not image_paths:
        print(f"[!] No images found under {input_dir}")
        return

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "angle_deg", "num_dets", "mean_conf"])

        for i, src in enumerate(image_paths, 1):
            rows = process_one_image(
                model, src, img_out, lbl_out, prev_out, args.conf, args.iou
            )
            for r in rows:
                writer.writerow(r)

            if i % 10 == 0:
                print(f"[i] Processed {i}/{len(image_paths)} images (with rotations)")

    print("\n✅ Pseudo-label dataset written to:", out_dir)
    print("   - images:", img_out)
    print("   - labels:", lbl_out)
    print("   - preview:", prev_out)
    print("   - data.yaml:", out_dir / "data.yaml")
    print("   - manifest:", manifest_path)
    print("\nNext, fine-tune with:")
    print(
        f"  yolo detect train model={args.model} data={out_dir/'data.yaml'} imgsz=640 epochs=100 batch=16 project=runs name=ttt_xo_real_rot_ft\n"
    )


if __name__ == "__main__":
    main()
