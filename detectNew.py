#!/usr/bin/env python3
# detect.py — Snapshot-based ROI tester with grid overlay + YOLO predictions

import sys
import time
import json
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

MODEL_PATH = "/Users/erickduarte/git/TestingSandbox/runs/ttt_xo_mixed/weights/best.pt"
ROI_FILE = "roi.json"
CONF_THRESH = 0.5


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def open_cam(indexes=(0, 1, 2), backends=None, width=1280, height=720, fps=30):
    if backends is None:
        if sys.platform == "darwin":
            backends = [cv2.CAP_AVFOUNDATION]
        elif sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW]
        else:
            backends = [cv2.CAP_V4L2, 0]
    for be in backends:
        for idx in indexes:
            cap = cv2.VideoCapture(idx, be) if be != 0 else cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[i] Camera opened (index={idx}, backend={be})")
                return cap
            cap.release()
    return None


def load_roi():
    if Path(ROI_FILE).exists():
        with open(ROI_FILE, "r") as f:
            return json.load(f)
    return None


def save_roi(x, y, w, h):
    roi = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    with open(ROI_FILE, "w") as f:
        json.dump(roi, f)
    return roi


def select_roi(frame):
    r = cv2.selectROI("Select ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, r)
    if w > 0 and h > 0:
        return save_roi(x, y, w, h)
    return None


def rects_from_roi(roi):
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    cell_w, cell_h = w // 3, h // 3
    rects = []
    for r in range(3):
        for c in range(3):
            X1 = x + c * cell_w
            Y1 = y + r * cell_h
            X2 = X1 + cell_w
            Y2 = Y1 + cell_h
            rects.append((X1, Y1, X2, Y2))
    return rects


def draw_grid(frame, roi):
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 3)
    for i in range(1, 3):
        cv2.line(overlay, (x + i * w // 3, y), (x + i * w // 3, y + h), (0, 255, 0), 2)
        cv2.line(overlay, (x, y + i * h // 3), (x + w, y + i * h // 3), (0, 255, 0), 2)
    return overlay


def assign_to_cells(res, rects, conf_thresh=CONF_THRESH):
    labels = [""] * 9
    best_conf = np.zeros(9)
    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        if conf < conf_thresh:
            continue
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        for i, (X1, Y1, X2, Y2) in enumerate(rects):
            if X1 <= cx < X2 and Y1 <= cy < Y2:
                if conf > best_conf[i]:
                    labels[i] = "X" if cls_id == 0 else "O"
                    best_conf[i] = conf
    return labels


def visualize(frame, res, rects, labels):
    img = frame.copy()
    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        color = (0, 255, 255) if cls_id == 0 else (255, 180, 0)
        label = "X" if cls_id == 0 else "O"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img,
            f"{label} {conf:.2f}",
            (x1, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    for i, (X1, Y1, X2, Y2) in enumerate(rects):
        cv2.putText(
            img,
            f"{i+1}:{labels[i] or '-'}",
            (X1 + 5, Y1 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
    return img


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("[i] Loading YOLO model...")
    model = YOLO(MODEL_PATH)

    print("[i] Opening camera...")
    cap = open_cam()
    if cap is None:
        print("[!] Could not open any camera.")
        return

    print("[i] Press 's' to take a snapshot, or 'q' to quit.")
    frame = None
    while True:
        ok, frame_live = cap.read()
        if not ok:
            continue
        cv2.imshow("Live Feed", frame_live)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            frame = frame_live.copy()
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyWindow("Live Feed")

    # ROI
    roi = load_roi()
    if roi is None:
        print("[i] No ROI found. Please select a 3x3 area on the board.")
        roi = select_roi(frame)
        if roi is None:
            print("[!] ROI selection cancelled.")
            return

    rects = rects_from_roi(roi)
    grid_img = draw_grid(frame, roi)

    res = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
    labels = assign_to_cells(res, rects)
    vis = visualize(grid_img, res, rects, labels)

    print("\n--- Board State ---")
    for r in range(3):
        print("|".join(labels[r * 3 : (r + 1) * 3]))
    print("-------------------")

    cv2.imshow("Snapshot + Grid + Detections", vis)
    print("[i] Press 'r' to reselect ROI, 'q' to quit.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            roi = select_roi(frame)
            if roi is not None:
                rects = rects_from_roi(roi)
                grid_img = draw_grid(frame, roi)
                res = model.predict(frame, conf=CONF_THRESH, verbose=False)[0]
                labels = assign_to_cells(res, rects)
                vis = visualize(grid_img, res, rects, labels)
                cv2.imshow("Snapshot + Grid + Detections", vis)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
