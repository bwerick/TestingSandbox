"""
train_ttt_xo_yolo.py

- Generates a synthetic tic-tac-toe dataset (images + YOLO .txt labels)
- Saves data.yaml
- Trains a YOLO model from scratch (Ultralytics) with nc=2 (X, O)
- Saves weights under runs/detect/ttt_xo_synth/weights/

Classes:
  0 -> X
  1 -> O
"""

import os
import random
import math
import yaml
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageEnhance


# -----------------------
# Config
# -----------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Dataset sizes
NUM_TRAIN = 800
NUM_VAL = 200

# Image / board settings
IMG_SIZE = 640
GRID_THICKNESS = (4, 10)  # px range
LINE_COLOR = (0, 0, 0)
BG_COLOR = (230, 230, 230)

# Mark settings
P_X = 0.45  # probability a cell is X
P_O = 0.45  # probability a cell is O (rest empty)
X_STROKE = (6, 20)  # px range for X stroke width
O_STROKE = (6, 20)  # px range for O stroke width
MARGIN_RATIO = 0.12  # inner margin of cell for marks
JITTER_RATIO = 0.06  # jitter in center
SCALE_RANGE = (0.75, 1.15)  # relative mark scale in cell

# Output
DATASET_DIR = Path("dataset_xo")
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_SUB = "train"
VAL_SUB = "val"
DATA_YAML = DATASET_DIR / "data.yaml"

# Training
PROJECT = "runs"
RUN_NAME = "ttt_xo_mixed"
EPOCHS = 60
BATCH = 16
IMGSZ = 640


# -----------------------
# Utilities
# -----------------------
def ensure_ultralytics_installed():
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        print("[i] ultralytics not found. Installing...")
        subprocess.check_call(["pip", "install", "-q", "ultralytics"])
        print("[i] ultralytics installed.")


def yolo_bbox_from_xyxy(
    img_w: int, img_h: int, x1: float, y1: float, x2: float, y2: float
) -> Tuple[float, float, float, float]:
    """Convert absolute xyxy to YOLO normalized (xc, yc, w, h)."""
    w = x2 - x1
    h = y2 - y1
    xc = x1 + w / 2.0
    yc = y1 + h / 2.0
    return xc / img_w, yc / img_h, w / img_w, h / img_h


def clamp_xyxy(x1, y1, x2, y2, W, H):
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    if x2 <= x1:
        x2 = min(W - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(H - 1, y1 + 1)
    return x1, y1, x2, y2


def rand_between(lo, hi):
    return lo + random.random() * (hi - lo)


def add_camera_like_artifacts(img: Image.Image) -> Image.Image:
    """Add light blur, brightness/contrast jitter, vignetting, and JPEG-like soften."""
    # Brightness/contrast
    img = ImageEnhance.Brightness(img).enhance(rand_between(0.9, 1.1))
    img = ImageEnhance.Contrast(img).enhance(rand_between(0.9, 1.1))

    # Slight color cast
    img = ImageEnhance.Color(img).enhance(rand_between(0.9, 1.1))

    # Mild blur sometimes
    if random.random() < 0.4:
        img = img.filter(ImageFilter.GaussianBlur(radius=rand_between(0.3, 1.2)))

    # Vignette
    if random.random() < 0.5:
        w, h = img.size
        vignette = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(vignette)
        maxrad = math.hypot(w, h) / 1.5
        for r in range(int(maxrad), 0, -20):
            alpha = int(255 * (1 - r / maxrad) * 0.25)
            bbox = (w // 2 - r, h // 2 - r, w // 2 + r, h // 2 + r)
            draw.ellipse(bbox, fill=alpha)
        img = Image.composite(
            img,
            ImageOps.colorize(vignette, (0, 0, 0), (0, 0, 0)),
            vignette.convert("L"),
        )

    return img


def draw_grid(draw: ImageDraw.ImageDraw, W: int, H: int) -> List[Tuple[int, int]]:
    """Draws 3x3 grid; returns list of (vertical_lines_x, horizontal_lines_y)."""
    # Grid lines positions
    xs = [int(W * 1 / 3), int(W * 2 / 3)]
    ys = [int(H * 1 / 3), int(H * 2 / 3)]
    thickness = int(rand_between(*GRID_THICKNESS))

    for x in xs:
        draw.line([(x, 0), (x, H)], fill=LINE_COLOR, width=thickness)
    for y in ys:
        draw.line([(0, y), (W, y)], fill=LINE_COLOR, width=thickness)

    return xs, ys


def cell_bounds(idx: int, W: int, H: int) -> Tuple[int, int, int, int]:
    """Get xyxy pixel bounds for a cell index 0..8 in row-major order."""
    r = idx // 3
    c = idx % 3
    x0 = int(c * W / 3)
    y0 = int(r * H / 3)
    x1 = int((c + 1) * W / 3)
    y1 = int((r + 1) * H / 3)
    return x0, y0, x1, y1


def draw_X(
    draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Draw an X inside bbox, return tight xyxy bounds used."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    # inner rect to avoid touching gridlines
    margin = int(min(w, h) * MARGIN_RATIO)
    ix0, iy0, ix1, iy1 = x0 + margin, y0 + margin, x1 - margin, y1 - margin

    # jitter center & scale
    cx = (ix0 + ix1) / 2 + rand_between(-JITTER_RATIO, JITTER_RATIO) * w
    cy = (iy0 + iy1) / 2 + rand_between(-JITTER_RATIO, JITTER_RATIO) * h
    scale = rand_between(*SCALE_RANGE)

    half_w = (ix1 - ix0) * 0.4 * scale
    half_h = (iy1 - iy0) * 0.4 * scale

    stroke = int(rand_between(*X_STROKE))
    theta = rand_between(-12, 12) * math.pi / 180.0

    # base diagonals relative to center
    dx = half_w
    dy = half_h
    corners = np.array([[-dx, -dy], [dx, dy], [-dx, dy], [dx, -dy]], dtype=np.float32)

    # rotate
    rot = np.array(
        [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]],
        dtype=np.float32,
    )
    c = corners @ rot.T
    a1 = (cx + c[0, 0], cy + c[0, 1])
    a2 = (cx + c[1, 0], cy + c[1, 1])
    b1 = (cx + c[2, 0], cy + c[2, 1])
    b2 = (cx + c[3, 0], cy + c[3, 1])

    draw.line([a1, a2], fill=LINE_COLOR, width=stroke)
    draw.line([b1, b2], fill=LINE_COLOR, width=stroke)

    # compute tight bbox around both lines
    xs = [a1[0], a2[0], b1[0], b2[0]]
    ys = [a1[1], a2[1], b1[1], b2[1]]
    bx0, by0, bx1, by1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    # expand a bit to cover stroke fully
    pad = int(stroke * 0.6)
    return bx0 - pad, by0 - pad, bx1 + pad, by1 + pad


def draw_O(
    draw: ImageDraw.ImageDraw, bbox: Tuple[int, int, int, int]
) -> Tuple[int, int, int, int]:
    """Draw an O (circle outline) inside bbox, return tight xyxy."""
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    margin = int(min(w, h) * MARGIN_RATIO)
    ix0, iy0, ix1, iy1 = x0 + margin, y0 + margin, x1 - margin, y1 - margin

    cx = (ix0 + ix1) / 2 + rand_between(-JITTER_RATIO, JITTER_RATIO) * w
    cy = (iy0 + iy1) / 2 + rand_between(-JITTER_RATIO, JITTER_RATIO) * h
    scale = rand_between(*SCALE_RANGE)

    rx = (ix1 - ix0) * 0.35 * scale
    ry = (iy1 - iy0) * 0.35 * scale

    stroke = int(rand_between(*O_STROKE))

    # ellipse bounding box
    bx0, by0 = int(cx - rx), int(cy - ry)
    bx1, by1 = int(cx + rx), int(cy + ry)
    draw.ellipse([bx0, by0, bx1, by1], outline=LINE_COLOR, width=stroke)

    pad = int(stroke * 0.6)
    return bx0 - pad, by0 - pad, bx1 + pad, by1 + pad


def synth_image_and_labels(W=IMG_SIZE, H=IMG_SIZE):
    """Create one synthetic board image and labels for X & O."""
    # Background with subtle texture
    base = Image.new("RGB", (W, H), BG_COLOR)
    # Add faint paper/desk texture via noise
    if random.random() < 0.8:
        noise = (np.random.randn(H, W, 3) * rand_between(2, 7)).astype(np.int16)
        arr = np.array(base, dtype=np.int16) + noise
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        base = Image.fromarray(arr, mode="RGB")

    draw = ImageDraw.Draw(base)
    draw_grid(draw, W, H)

    labels = []  # list of (class_id, x_center_rel, y_center_rel, w_rel, h_rel)

    # For each cell, choose X/O/empty and draw + record bbox
    for idx in range(9):
        cb = cell_bounds(idx, W, H)
        r = random.random()
        if r < P_X:
            x1, y1, x2, y2 = draw_X(draw, cb)
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)
            xc, yc, bw, bh = yolo_bbox_from_xyxy(W, H, x1, y1, x2, y2)
            labels.append((0, xc, yc, bw, bh))
        elif r < P_X + P_O:
            x1, y1, x2, y2 = draw_O(draw, cb)
            x1, y1, x2, y2 = clamp_xyxy(x1, y1, x2, y2, W, H)
            xc, yc, bw, bh = yolo_bbox_from_xyxy(W, H, x1, y1, x2, y2)
            labels.append((1, xc, yc, bw, bh))
        else:
            # empty
            pass

    # Camera-like artifacts
    base = add_camera_like_artifacts(base)

    return base, labels


def save_yolo_label(path: Path, labels: List[Tuple[int, float, float, float, float]]):
    with open(path, "w") as f:
        for cls, xc, yc, bw, bh in labels:
            f.write(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")


def make_dataset():
    if DATASET_DIR.exists():
        print(f"[i] Removing existing {DATASET_DIR} ...")
        shutil.rmtree(DATASET_DIR)

    (IMAGES_DIR / TRAIN_SUB).mkdir(parents=True, exist_ok=True)
    (IMAGES_DIR / VAL_SUB).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / TRAIN_SUB).mkdir(parents=True, exist_ok=True)
    (LABELS_DIR / VAL_SUB).mkdir(parents=True, exist_ok=True)

    print("[i] Generating training images...")
    for i in range(NUM_TRAIN):
        img, labels = synth_image_and_labels()
        img_path = IMAGES_DIR / TRAIN_SUB / f"img_{i:05d}.jpg"
        lbl_path = LABELS_DIR / TRAIN_SUB / f"img_{i:05d}.txt"
        img.save(img_path, quality=95)
        save_yolo_label(lbl_path, labels)

    print("[i] Generating validation images...")
    for i in range(NUM_VAL):
        img, labels = synth_image_and_labels()
        img_path = IMAGES_DIR / VAL_SUB / f"img_{i:05d}.jpg"
        lbl_path = LABELS_DIR / VAL_SUB / f"img_{i:05d}.txt"
        img.save(img_path, quality=95)
        save_yolo_label(lbl_path, labels)

    # data.yaml
    data = {
        "path": str(DATASET_DIR.resolve()),
        "train": f"images/{TRAIN_SUB}",
        "val": f"images/{VAL_SUB}",
        "nc": 2,
        "names": ["X", "O"],
    }
    with open(DATA_YAML, "w") as f:
        yaml.safe_dump(data, f)
    print(f"[i] Wrote {DATA_YAML}")


def write_model_yaml(model_yaml_path: Path, base: str = "yolo11n.yaml", nc: int = 2):
    """
    Create a local copy of the model yaml with nc=2.
    We read the base yaml from ultralytics package and write a modified copy.
    """
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG_DICT

    # Try to load the base model YAML config from ultralytics assets
    # Fallback: construct a minimal dict
    try:
        from ultralytics.utils import yaml_load

        # Attempt common base path
        import ultralytics

        base_dir = Path(ultralytics.__file__).parent / "cfg" / "models" / "11"
        base_yaml = base_dir / base
        if not base_yaml.exists():
            # Older versions might store v8 configs
            base_dir_v8 = Path(ultralytics.__file__).parent / "cfg" / "models" / "v8"
            base_yaml = base_dir_v8 / "yolov8n.yaml"
        cfg = yaml.safe_load(open(base_yaml, "r"))
    except Exception:
        # Minimal fallback config (YOLOv5/8-like)
        cfg = dict(
            nc=nc,
            depth_multiple=0.33,
            width_multiple=0.25,
            anchors=3,
            # The rest will be filled by ultralytics defaults; this is just a fallback.
        )

    cfg["nc"] = nc
    with open(model_yaml_path, "w") as f:
        yaml.safe_dump(cfg, f)
    print(f"[i] Wrote model config: {model_yaml_path}")


from ultralytics import YOLO


def train_yolo():
    ensure_ultralytics_installed()

    # Try YOLOv11; fall back to YOLOv8 if needed
    base_arch = "yolo11n.yaml"
    try:
        model = YOLO(base_arch)  # build model from official YAML
    except Exception:
        base_arch = "yolov8n.yaml"
        model = YOLO(base_arch)

    print(f"[i] Using base arch: {base_arch} (training from scratch)")
    model.train(
        data=str(DATA_YAML),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        project=PROJECT,
        name=RUN_NAME,
        pretrained=False,  # <-- train from scratch
        task="detect",  # <-- silence the warning and be explicit
        verbose=True,
    )
    print(f"[i] Training finished. Check: {PROJECT}/{RUN_NAME}/weights/best.pt")


def main():
    if not os.path.exists("dataset_xo/images/train"):
        print("[i] Generating synthetic dataset...")
        make_dataset()
    else:
        print("[i] Using existing dataset_xo/ images and labels.")
    train_yolo()


if __name__ == "__main__":
    main()
