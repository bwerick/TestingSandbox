#!/usr/bin/env python3
"""
cam_detect.py — Live camera detection for YOLO start/goal dots.

Usage:
  python cam_detect.py --weights runs/maze_yolo/exp/weights/best.pt
  # optional:
  python cam_detect.py --weights best.pt --imgsz 640 --conf 0.25 --device cpu --camera 0
"""

import argparse
import time
from pathlib import Path
import os

import cv2
import numpy as np
from ultralytics import YOLO

# BGR colors for drawing
COLOR_START = (0, 255, 0)  # green
COLOR_GOAL = (0, 0, 255)  # red
COLOR_TEXT = (0, 0, 0)  # black text
COLOR_BG = (255, 255, 255)  # label background


def draw_label(img, text, x, y, color_box, font_scale=0.5, thickness=1, pad=3):
    """Draws a filled background with text on top (for readability)."""
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    cv2.rectangle(
        img, (x, y - th - 2 * pad), (x + tw + 2 * pad, y + baseline), COLOR_BG, -1
    )
    cv2.putText(
        img,
        text,
        (x + pad, y - pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        COLOR_TEXT,
        thickness,
        cv2.LINE_AA,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained weights .pt")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--device", default=None, help="PyTorch device, e.g. 0 or 'cpu'")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (0 is default)")
    ap.add_argument(
        "--width", type=int, default=0, help="Request capture width (0=leave default)"
    )
    ap.add_argument(
        "--height", type=int, default=0, help="Request capture height (0=leave default)"
    )
    ap.add_argument("--save_dir", default="runs/cam", help="Folder to save snapshots")
    args = ap.parse_args()

    # Load model
    print(f"[INFO] Loading model: {args.weights}")
    model = YOLO(args.weights)

    # Camera
    cap = cv2.VideoCapture(args.camera)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    os.makedirs(args.save_dir, exist_ok=True)

    last_t = time.time()
    fps = 0.0

    print("[INFO] Press 'S' to save a snapshot, 'Q' or ESC to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("[WARN] Failed to grab frame.")
            break

        # Run YOLO on this frame (Ultralytics handles BGR numpy arrays)
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            verbose=False,
        )

        # Parse predictions for this single frame
        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                # xyxy in pixels
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy.tolist()
                cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else -1
                conf = float(box.conf[0].cpu().item()) if box.conf is not None else 0.0

                # Classes: 0=start (green), 1=goal (red)
                if cls_id == 0:
                    color = COLOR_START
                    label = f"start {conf:.2f}"
                elif cls_id == 1:
                    color = COLOR_GOAL
                    label = f"goal {conf:.2f}"
                else:
                    color = (255, 255, 0)  # unexpected class (yellow)
                    label = f"cls{cls_id} {conf:.2f}"

                # Draw box + label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                draw_label(frame, label, x1, y1, color)

        # FPS overlay
        now = time.time()
        dt = now - last_t
        last_t = now
        fps = (0.9 * fps + 0.1 * (1.0 / dt)) if dt > 0 else fps
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (50, 200, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("YOLO Cam — start/goal", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):  # ESC or Q
            break
        if key in (ord("s"), ord("S")):
            out_path = Path(args.save_dir) / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(out_path), frame)
            print(f"[INFO] Saved {out_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
