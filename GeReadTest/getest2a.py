import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# ===================== USER SETTINGS =====================
# image_path = "/home/ji541/Shared/Takes/captureGe0.png"
image_path = "/home/ji541/Shared/Takes/captureGe1.png"
output_csv = "/home/ji541/Shared/ekg_digitized_green.csv"

# Warp options:
#   "none"            : do nothing
#   "manual4"         : click 4 corners (TL, TR, BR, BL) to warp
#   "auto_placeholder": stub for future automatic warp
warp_mode = "none"   # change to "manual4" if you want manual warping now

# Band size (pixels): main knob for missing peaks vs. jumping to noise
BAND_PX = 140  #chatGPT suggested 110 initially # try 80–160
###Missing peaks → increase to 140–180\
###Tracker jumps to other UI elements → decrease to 70–110

# HSV green threshold (adjust if needed)
HSV_LOWER = (40, 80, 80)
HSV_UPPER = (95, 255, 255)

# Morphology
KERNEL_SIZE = 1
OPEN_ITERS = 0
CLOSE_ITERS = 0

# If True, you will be prompted to draw ROI each run.
# If False, and ROI_FILE exists, it will reuse saved ROI.
INTERACTIVE_ROI = True

# ROI save file (so you can reuse it)
ROI_FILE = os.path.splitext(image_path)[0] + "_roi.txt"
# =========================================================


def manual_warp_4points(img):
    """
    Manual warp by clicking 4 points: TL, TR, BR, BL. Press 'w' to warp, 'q' to cancel.
    """
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


def auto_warp_placeholder(img):
    """
    Placeholder for future automatic warp.
    Idea (to implement later):
      - Detect monitor screen region by finding the largest quadrilateral contour.
      - Use edge detection + contour approximation (approxPolyDP) to get 4 corners.
      - Apply cv2.getPerspectiveTransform + cv2.warpPerspective.
    For now: return input unchanged.
    """
    # TODO: implement robust quad detection, corner ordering, perspective transform
    return img


def choose_or_load_roi(img, roi_file, interactive=True):
    """
    Interactive ROI selector using cv2.selectROI.
    Saves ROI to roi_file for reuse.
    """
    if (not interactive) and os.path.exists(roi_file):
        try:
            with open(roi_file, "r") as f:
                x, y, w, h = [int(v) for v in f.read().strip().split(",")]
            return (x, y, w, h)
        except Exception:
            pass

    # Interactive selection
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


def digitize_green_trace(img, roi_rect, band_px, hsv_lower, hsv_upper,
                         kernel_size=3, open_iters=1, close_iters=2):
    """
    Digitize green trace inside ROI.
    Returns: time (0..1), signal (normalized), overlay_full, roi_img, mask
    """
    x, y, w, h = roi_rect
    roi = img[y:y+h, x:x+w].copy()

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iters)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iters)

    roi_h, roi_w = mask.shape[:2]
    y_positions = np.full(roi_w, np.nan, dtype=float)

    prev_y = roi_h // 2

    # --- polarity-agnostic spike picking knobs ---
    SPAN_THR_PX = 12       # lower -> more columns treated as spike
    BASELINE_WIN = 25      # baseline smoothing window
    baseline_buf = []      # recent y picks to estimate baseline
    # --------------------------------------------

    for xi in range(roi_w):
        ys = np.where(mask[:, xi] > 0)[0]
        if ys.size == 0:
            continue

        # Band-limit around previous y to avoid jumping to other green UI elements
        y_band = ys[(ys >= prev_y - band_px) & (ys <= prev_y + band_px)]
        if y_band.size == 0:
            y_band = ys

        span = int(y_band.max() - y_band.min())

        # Robust local baseline from recent picks (fallback to prev_y at the start)
        if len(baseline_buf) >= 5:
            baseline = float(np.median(baseline_buf[-BASELINE_WIN:]))
        else:
            baseline = float(prev_y)

        if span >= SPAN_THR_PX:
            # Spike-like column: pick extreme (top OR bottom) farther from baseline
            y_top = float(y_band.min())
            y_bot = float(y_band.max())
            yi = int(y_top if abs(y_top - baseline) >= abs(y_bot - baseline) else y_bot)
        else:
            # Non-spike: median is stable
            yi = int(np.median(y_band))

        y_positions[xi] = yi
        prev_y = yi

        baseline_buf.append(yi)
        if len(baseline_buf) > 5 * BASELINE_WIN:
            baseline_buf = baseline_buf[-5 * BASELINE_WIN:]

    nan_mask = np.isnan(y_positions)
    good_x = np.where(~nan_mask)[0]
    if good_x.size < 20:
        raise RuntimeError("Too few trace pixels detected. Adjust ROI or HSV thresholds.")

    # Interpolate missing columns
    y_positions[nan_mask] = np.interp(np.where(nan_mask)[0], good_x, y_positions[good_x])

    # Normalize amplitude
    y_min, y_max = float(np.min(y_positions)), float(np.max(y_positions))
    y_center = 0.5 * (y_min + y_max)
    signal = -(y_positions - y_center) / (0.5 * (y_max - y_min) + 1e-8)

    # Time axis (normalized). Later you’ll calibrate to seconds.
    time = np.linspace(0.0, 1.0, roi_w)

    # Build overlay on full image
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x+w, y+h), (255, 0, 0), 2)
    for xi in range(roi_w):
        cv2.circle(overlay, (x + xi, y + int(y_positions[xi])), 1, (0, 0, 255), -1)

    return time, signal, overlay, roi, mask



def main():
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not load image: {image_path}")
        raise SystemExit(1)

    # 0) Optional warp stage
    if warp_mode == "manual4":
        img2 = manual_warp_4points(img)
    elif warp_mode == "auto_placeholder":
        img2 = auto_warp_placeholder(img)
    else:
        img2 = img

    # 1) Interactive ROI selection (drag box)
    roi_rect = choose_or_load_roi(img2, ROI_FILE, interactive=INTERACTIVE_ROI)
    print(f"ROI (x,y,w,h) = {roi_rect}")
    print(f"Band size BAND_PX = {BAND_PX}  (increase if peaks missed; decrease if jumping to noise)")

    # 2) Digitize
    time, signal, overlay, roi_img, mask = digitize_green_trace(
        img2, roi_rect,
        band_px=BAND_PX,
        hsv_lower=HSV_LOWER,
        hsv_upper=HSV_UPPER,
        kernel_size=KERNEL_SIZE,
        open_iters=OPEN_ITERS,
        close_iters=CLOSE_ITERS
    )

    data = np.column_stack((time, signal))
    np.savetxt(output_csv, data, delimiter=",", header="time,value", comments="")
    print(f"Saved digitized signal to: {output_csv}")
    
    

		

    # 3) Visualize
    cv2.imshow("Image (post-warp if used)", img2)
    cv2.imshow("ROI", roi_img)
    cv2.imshow("Green mask", mask)
    cv2.imshow("Overlay (ROI + trace)", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    plt.figure(figsize=(10, 3))
    plt.plot(time, signal, '.-', linewidth=1, markersize=4)
    plt.xlabel("Time (normalized)")
    plt.ylabel("Amplitude (normalized)")
    plt.title("Digitized EKG Trace")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
