import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ===================== USER SETTINGS =====================
image_path = "/home/ji541/Shared/Takes/captureGe1.png"
output_csv = "/home/ji541/Shared/ekg_digitized.csv"

# Warp options:
#   "none"    : do nothing
#   "manual4" : click 4 corners (TL, TR, BR, BL) to warp
warp_mode = "none"

# Band size (pixels): main knob for missing peaks vs. jumping to noise
BAND_PX = 80  # try 80–180

# Detection mode:
#   "contrast" : grayscale-based (recommended; works for any waveform color)
#   "green"    : HSV green mask (only if your trace is green)
#   "auto"     : tries contrast first; falls back to green if contrast fails
DETECT_MODE = "green"

# --- Contrast-based segmentation knobs (tune if needed) ---
# Background removal kernel size: larger removes slow background (grid/glare) more aggressively
BG_KERNEL = 31         # try 21–51
ADAPTIVE_BLOCK = 31    # must be odd; try 21–51
ADAPTIVE_C = -5        # more negative -> more pixels included; try -2 to -12
# ---------------------------------------------------------

# --- Green HSV thresholds (only used in green/auto fallback) ---
HSV_LOWER = (40, 80, 80)
HSV_UPPER = (95, 255, 255)
# -------------------------------------------------------------

# Morphology (keep gentle; avoid rounding peaks)
KERNEL_SIZE = 3
OPEN_ITERS = 2
CLOSE_ITERS = 2

# If True, prompt ROI each run; ROI saved to file for reuse
INTERACTIVE_ROI = True
ROI_FILE = os.path.splitext(image_path)[0] + "_roi.txt"

# Spike tracking knobs (polarity-agnostic)
SPAN_THR_PX = 10       # lower -> more columns treated as spike; try 8–14
BASELINE_WIN = 25      # baseline smoothing; try 15–60

# =========================================================


def manual_warp_4points(img):
    clone = img.copy()
    points = []
    win = "Warp: click 4 corners TL,TR,BR,BL then press 'w' (q=cancel)"

    def cb(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(clone, (x, y), 6, (0, 0, 255), -1)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, cb)

    while True:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            return img
        if k == ord('w'):
            if len(points) != 4:
                print(f"Need 4 points; currently {len(points)}")
                continue
            break

    cv2.destroyAllWindows()

    pts = np.array(points, dtype=np.float32)
    (tl, tr, br, bl) = pts

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxW, maxH))
    return warped


def choose_or_load_roi(img, roi_file, interactive=True):
    if (not interactive) and os.path.exists(roi_file):
        try:
            with open(roi_file, "r") as f:
                x, y, w, h = [int(v) for v in f.read().strip().split(",")]
            return (x, y, w, h)
        except Exception:
            pass

    win = "Select ROI: drag rectangle, press ENTER/SPACE to confirm, 'c' to cancel"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    r = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(win)

    x, y, w, h = [int(v) for v in r]
    if w <= 0 or h <= 0:
        raise RuntimeError("ROI selection cancelled or invalid.")

    with open(roi_file, "w") as f:
        f.write(f"{x},{y},{w},{h}")

    return (x, y, w, h)


def skeletonize(binary_mask):
    """Morphological skeletonization (works without opencv-contrib)."""
    skel = np.zeros(binary_mask.shape, np.uint8)
    #element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    img = binary_mask.copy()

    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def mask_quality_ok(mask):
    """Heuristic check: reject masks that are too empty or too full."""
    frac = float(np.count_nonzero(mask)) / float(mask.size)
    return (0.0005 <= frac <= 0.20), frac


def build_mask_contrast(roi_bgr):
    """
    Contrast-based segmentation designed to work for any waveform color:
      - convert to gray
      - remove slow background (grid/glare) via morphological opening
      - adaptive threshold on the high-pass image
      - light horizontal close to connect the trace
    """
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    # Mild denoise while preserving edges
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Background estimation and subtraction (high-pass)
    k = BG_KERNEL if BG_KERNEL % 2 == 1 else BG_KERNEL + 1
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray_blur, cv2.MORPH_OPEN, bg_kernel)
    hp = cv2.subtract(gray_blur, bg)

    # Normalize to use full range
    hp = cv2.normalize(hp, None, 0, 255, cv2.NORM_MINMAX)

    # Adaptive threshold: try both polarities and choose better
    block = ADAPTIVE_BLOCK if ADAPTIVE_BLOCK % 2 == 1 else ADAPTIVE_BLOCK + 1

    th1 = cv2.adaptiveThreshold(
        hp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        block, ADAPTIVE_C
    )
    th2 = cv2.adaptiveThreshold(
        hp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, ADAPTIVE_C
    )

    ok1, frac1 = mask_quality_ok(th1)
    ok2, frac2 = mask_quality_ok(th2)

    # Choose mask with "reasonable" fill; prefer the smaller fill if both reasonable
    if ok1 and ok2:
        mask = th1 if frac1 <= frac2 else th2
    elif ok1:
        mask = th1
    else:
        mask = th2

    # Gentle cleanup: connect along x without rounding peaks too much
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    if OPEN_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)

    if CLOSE_ITERS > 0:
        # Horizontal close is less likely to round vertical spike tips
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=CLOSE_ITERS)

    return gray, hp, mask


def build_mask_green(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (KERNEL_SIZE, KERNEL_SIZE))
    if OPEN_ITERS > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITERS)
    if CLOSE_ITERS > 0:
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h, iterations=CLOSE_ITERS)

    return mask


def fill_gaps_nearest(y_positions):
    nan_mask = np.isnan(y_positions)
    good_x = np.where(~nan_mask)[0]
    if good_x.size < 20:
        raise RuntimeError("Too few trace points detected. Tighten ROI or adjust segmentation knobs.")

    missing = np.where(nan_mask)[0]
    for idx in missing:
        j = good_x[np.argmin(np.abs(good_x - idx))]
        y_positions[idx] = y_positions[j]
    return y_positions


def digitize_trace_from_mask(img, roi_rect, mask_bin, band_px):
    """Polarity-agnostic extreme selection using baseline distance + spike span."""
    x, y, w, h = roi_rect
    roi_h, roi_w = mask_bin.shape[:2]

    y_positions = np.full(roi_w, np.nan, dtype=float)
    prev_y = roi_h // 2
    baseline_buf = []

    for xi in range(roi_w):
        ys = np.where(mask_bin[:, xi] > 0)[0]
        if ys.size == 0:
            continue

        y_band = ys[(ys >= prev_y - band_px) & (ys <= prev_y + band_px)]
        if y_band.size == 0:
            y_band = ys

        span = int(y_band.max() - y_band.min())

        if len(baseline_buf) >= 5:
            baseline = float(np.median(baseline_buf[-BASELINE_WIN:]))
        else:
            baseline = float(prev_y)

        if span >= SPAN_THR_PX:
            y_top = float(y_band.min())
            y_bot = float(y_band.max())
            yi = int(y_top if abs(y_top - baseline) >= abs(y_bot - baseline) else y_bot)
        else:
            yi = int(np.median(y_band))

        y_positions[xi] = yi
        prev_y = yi

        baseline_buf.append(yi)
        if len(baseline_buf) > 5 * BASELINE_WIN:
            baseline_buf = baseline_buf[-5 * BASELINE_WIN:]

    y_positions = fill_gaps_nearest(y_positions)

    # Normalize amplitude
    y_min, y_max = float(np.min(y_positions)), float(np.max(y_positions))
    y_center = 0.5 * (y_min + y_max)
    signal = -(y_positions - y_center) / (0.5 * (y_max - y_min) + 1e-8)

    time = np.linspace(0.0, 1.0, roi_w)

    # Overlay on full image
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)
    for xi in range(roi_w):
        cv2.circle(overlay, (x + xi, y + int(y_positions[xi])), 1, (0, 0, 255), -1)

    return time, signal, overlay


def main():
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load image: {image_path}")
        raise SystemExit(1)

    # Optional warp stage
    if warp_mode == "manual4":
        img2 = manual_warp_4points(img)
    else:
        img2 = img

    # Interactive ROI selection
    roi_rect = choose_or_load_roi(img2, ROI_FILE, interactive=INTERACTIVE_ROI)
    x, y, w, h = roi_rect
    roi = img2[y:y + h, x:x + w].copy()

    print(f"ROI (x,y,w,h) = {roi_rect}")
    print(f"BAND_PX = {BAND_PX}  (increase if peaks missed; decrease if jumping)")

    # Build mask
    debug_gray = None
    debug_hp = None

    if DETECT_MODE in ("contrast", "auto"):
        debug_gray, debug_hp, mask = build_mask_contrast(roi)
        ok, frac = mask_quality_ok(mask)
        print(f"Contrast mask fill fraction = {frac:.4f} (ok={ok})")

        if (DETECT_MODE == "auto") and (not ok):
            print("Contrast mask looks poor; falling back to green HSV mask.")
            mask = build_mask_green(roi)
    elif DETECT_MODE == "green":
        mask = build_mask_green(roi)
    else:
        raise ValueError("DETECT_MODE must be 'contrast', 'green', or 'auto'")

    # Skeletonize to sharpen peaks
    mask_thin = skeletonize(mask)

    # Digitize
    time, signal, overlay = digitize_trace_from_mask(img2, roi_rect, mask_thin, BAND_PX)

    data = np.column_stack((time, signal))
    np.savetxt(output_csv, data, delimiter=",", header="time,value", comments="")
    print(f"Saved digitized signal to: {output_csv}")

    # Visualize (OpenCV)
    cv2.imshow("Image (post-warp if used)", img2)
    cv2.imshow("ROI", roi)
    if debug_gray is not None:
        cv2.imshow("ROI gray", debug_gray)
    if debug_hp is not None:
        cv2.imshow("ROI high-pass (grid/glare suppressed)", debug_hp)
    cv2.imshow("Mask (binary)", mask)
    cv2.imshow("Mask (thin/skeleton)", mask_thin)
    cv2.imshow("Overlay (ROI + trace)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Quick plot (Matlab-style '.-')
    plt.figure(figsize=(10, 3))
    plt.plot(time, signal, ".-", linewidth=1, markersize=3)
    plt.xlabel("Time (normalized)")
    plt.ylabel("Amplitude (normalized)")
    plt.title("Digitized waveform (from ROI)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
