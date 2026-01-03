import cv2, sys, time
import numpy as np
from ultralytics import YOLO

# ===================== CONFIG =====================
MODEL_PATH = "/Users/erickduarte/git/TestingSandbox/runs/ttt_xo_mixed/weights/best.pt"  # <-- your YOLO model path
CONF_THRESH = 0.45
# ===================================================


import cv2, sys, time, json
import numpy as np
from ultralytics import YOLO
from pathlib import Path

ROI_FILE = "roi.json"


def open_cam(indexes=(0, 1, 2), backends=None, width=1280, height=1024, fps=25):
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
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # Warm up
            for _ in range(20):
                ok, _ = cap.read()
                if not ok:
                    break
                time.sleep(0.01)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size > 0:
                print(f"[OK] Camera {idx} via backend {be}, frame {frame.shape}")
                return cap
            cap.release()
    return None


def snapshot_with_preview():
    cap = open_cam()
    if cap is None:
        raise RuntimeError("Could not open camera (permissions/index/backend).")
    print("[i] Press 's' to snapshot, 'q' to quit.")
    frame = None
    while True:
        ok, f = cap.read()
        if not ok:
            continue
        cv2.imshow("Live Preview", f)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            frame = f.copy()
            break
        elif key == ord("q"):
            break
    cap.release()
    cv2.destroyWindow("Live Preview")
    if frame is None:
        raise RuntimeError("No snapshot taken.")
    return frame


def select_board_roi_on(frame):
    disp = frame.copy()
    max_side = 1200
    h, w = disp.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / float(max(h, w))
        disp = cv2.resize(disp, (int(w * scale), int(h * scale)))
    r = cv2.selectROI("Select Board ROI", disp, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Select Board ROI")
    if r == (0, 0, 0, 0):
        return None
    x, y, rw, rh = r
    x = int(x / scale)
    y = int(y / scale)
    rw = int(rw / scale)
    rh = int(rh / scale)
    return (x, y, rw, rh)


def save_roi(roi):
    with open(ROI_FILE, "w") as f:
        json.dump({"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]}, f)
    print(f"[i] ROI saved to {ROI_FILE}")


def load_roi():
    path = Path(ROI_FILE)
    if not path.exists():
        return None
    try:
        data = json.load(open(path))
        return (data["x"], data["y"], data["w"], data["h"])
    except Exception:
        return None


def read_board_by_simple_grid(model_path, conf_thresh=0.45, interactive_roi=True):
    model = YOLO(model_path)
    frame = snapshot_with_preview()

    # Load saved ROI if exists, otherwise ask user once
    roi_xywh = load_roi()
    if roi_xywh is None:
        print("[i] No saved ROI found — please select the board region.")
        roi_xywh = select_board_roi_on(frame)
        if roi_xywh is not None:
            save_roi(roi_xywh)

    # Fallback if ROI still missing
    if roi_xywh is None:
        h, w = frame.shape[:2]
        roi_xywh = (0, 0, w, h)

    # Divide ROI into 9 cells
    x, y, w, h = roi_xywh
    cw, ch = w // 3, h // 3
    rects = [
        (x + c * cw, y + r * ch, x + (c + 1) * cw, y + (r + 1) * ch)
        for r in range(3)
        for c in range(3)
    ]

    # Run detection
    res = model.predict(frame, conf=conf_thresh, verbose=False)[0]

    labels = [""] * 9
    best = np.zeros(9, dtype=np.float32)
    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        x1, y1, x2, y2 = map(float, b.xyxy[0])
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        for i, (X1, Y1, X2, Y2) in enumerate(rects):
            if X1 <= u < X2 and Y1 <= v < Y2:
                if conf > best[i]:
                    labels[i] = "X" if cls_id == 0 else "O"
                    best[i] = conf
                break

    # Visualization (optional)
    vis = frame.copy()
    v1, v2 = rects[1][0], rects[2][0]
    h1, h2 = rects[3][1], rects[6][1]
    cv2.line(vis, (v1, y), (v1, y + h), (0, 255, 0), 3)
    cv2.line(vis, (v2, y), (v2, y + h), (0, 255, 0), 3)
    cv2.line(vis, (x, h1), (x + w, h1), (0, 255, 0), 3)
    cv2.line(vis, (x, h2), (x + w, h2), (0, 255, 0), 3)

    for b in res.boxes:
        cls_id = int(b.cls)
        conf = float(b.conf)
        X1, Y1, X2, Y2 = map(int, b.xyxy[0])
        color = (0, 255, 255) if cls_id == 0 else (255, 200, 0)
        cv2.rectangle(vis, (X1, Y1), (X2, Y2), color, 3)
        lab = "X" if cls_id == 0 else "O"
        cv2.putText(
            vis,
            f"{lab} {conf:.2f}",
            (X1, max(0, Y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

    for i, (X1, Y1, X2, Y2) in enumerate(rects):
        t = labels[i] if labels[i] else "-"
        cv2.putText(
            vis,
            f"{i+1}:{t}",
            (X1 + 6, Y1 + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    grid = [labels[0:3], labels[3:6], labels[6:9]]
    print("\n--- Board State ---")
    for row in grid:
        print("|".join(c if c else " " for c in row))
    print("--------------------")

    return grid, labels, rects, frame


if __name__ == "__main__":
    grid, labels, rects, frame = read_board_by_simple_grid(
        model_path=MODEL_PATH, conf_thresh=CONF_THRESH, interactive_roi=True
    )
