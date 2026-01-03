import cv2
import time
import argparse
import numpy as np

# pip install easyocr opencv-python
import easyocr


def draw_polygons_with_labels(frame, results, conf_thresh=0.5):
    """
    results: list of (box, text, conf) from EasyOCR
    Draws polygon, then renders the text centered within its axis-aligned bounding rect.
    """
    import numpy as np
    import cv2

    for box, text, conf in results:
        if conf < conf_thresh or not text.strip():
            continue

        pts = np.array(box).astype(int)
        # 1) Draw the polygon
        cv2.polylines(frame, [pts], isClosed=True, thickness=2, color=(0, 255, 0))

        # 2) Axis-aligned bounding rect (easy, fast)
        x, y, w, h = cv2.boundingRect(pts)

        # 3) Choose a font scale that fits the word inside the rect
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        # Get size at scale=1, then compute scale to fit
        (tw, th), _ = cv2.getTextSize(text, font_face, 1.0, 2)
        if tw == 0 or th == 0:
            continue

        # padding to keep text off the edges
        pad = 6
        scale_w = (w - 2 * pad) / max(1, tw)
        scale_h = (h - 2 * pad) / max(1, th)
        scale = max(0.3, min(scale_w, scale_h))  # clamp to avoid tiny/huge text

        thickness = max(1, int(2 * scale))

        # Recompute exact text size at chosen scale
        (tw, th), baseline = cv2.getTextSize(text, font_face, scale, thickness)

        # 4) Center the text inside the rect
        tx = x + (w - tw) // 2
        ty = y + (h + th) // 2 - baseline

        # 5) Optional: semi-transparent background just behind the text for contrast
        bg_x1, bg_y1 = max(0, tx - 3), max(0, ty - th - 3)
        bg_x2, bg_y2 = min(frame.shape[1] - 1, tx + tw + 3), min(
            frame.shape[0] - 1, ty + baseline + 3
        )
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), thickness=-1
        )
        alpha = 0.35
        frame[bg_y1 : bg_y2 + 1, bg_x1 : bg_x2 + 1] = cv2.addWeighted(
            overlay[bg_y1 : bg_y2 + 1, bg_x1 : bg_x2 + 1],
            alpha,
            frame[bg_y1 : bg_y2 + 1, bg_x1 : bg_x2 + 1],
            1 - alpha,
            0,
        )

        # 6) Draw text with an outline for legibility
        cv2.putText(
            frame,
            text,
            (tx, ty),
            font_face,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            text,
            (tx, ty),
            font_face,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="Camera index")
    ap.add_argument(
        "--langs",
        type=str,
        default="en,es",
        help="Comma-separated languages (e.g., 'en,es')",
    )
    ap.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    ap.add_argument(
        "--resize",
        type=float,
        default=1.0,
        help="Scale input frame (e.g., 0.75 for speed)",
    )
    ap.add_argument(
        "--ocr_interval", type=float, default=0.1, help="Seconds between OCR runs"
    )
    ap.add_argument("--roi", action="store_true", help="Select ROI on first frame")
    args = ap.parse_args()

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    reader = easyocr.Reader(
        langs, gpu=True
    )  # set gpu=True if you have a CUDA-capable GPU and EasyOCR set up

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    roi = None
    last_ocr_time = 0.0
    last_results = []

    print("[q] quit  |  [r] reselect ROI  |  [s] save frame")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Optional downscale for speed
        if args.resize != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=args.resize,
                fy=args.resize,
                interpolation=cv2.INTER_LINEAR,
            )

        # Select ROI once (or on demand)
        if args.roi and roi is None:
            # SelectROI returns (x, y, w, h)
            r = cv2.selectROI(
                "Select ROI (press ENTER/SPACE to confirm, c to cancel)",
                frame,
                fromCenter=False,
                showCrosshair=True,
            )
            cv2.destroyWindow("Select ROI (press ENTER/SPACE to confirm, c to cancel)")
            if r is not None and r[2] > 0 and r[3] > 0:
                roi = tuple(map(int, r))

        # Crop if ROI is set
        if roi is not None:
            x, y, w, h = roi
            sub = frame[y : y + h, x : x + w]
        else:
            sub = frame

        # Throttle OCR to every N seconds to keep FPS reasonable
        now = time.time()
        if (now - last_ocr_time) >= args.ocr_interval:
            # EasyOCR expects RGB
            rgb = cv2.cvtColor(sub, cv2.COLOR_BGR2RGB)
            # detail=1 returns (box, text, conf); paragraph=False keeps line-level boxes faster
            last_results = reader.readtext(rgb, detail=1, paragraph=False)
            last_ocr_time = now

        # Draw results (adjust coords if ROI applied)
        display = frame.copy()
        if roi is None:
            draw_polygons_with_labels(display, last_results, conf_thresh=args.conf)
        else:
            shifted = []
            for box, text, conf in last_results:
                shifted_box = [(int(pt[0] + x), int(pt[1] + y)) for pt in box]
                shifted.append((shifted_box, text, conf))
            draw_polygons_with_labels(display, shifted, conf_thresh=args.conf)

        # Show FPS
        cv2.putText(
            display,
            f"OCR every {args.ocr_interval:.2f}s | ROI: {'ON' if roi else 'OFF'}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Live OCR", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            roi = None
            args.roi = True
        elif key == ord("s"):
            ts = int(time.time())
            fname = f"ocr_frame_{ts}.png"
            cv2.imwrite(fname, display)
            print(f"Saved {fname}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
