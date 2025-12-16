import cv2
import numpy as np

# ===== Parameters you may tweak =====
image_path = "/home/ji541/Shared/ekg_original.png"  # <-- put full path to your PNG here
#image_path = "/home/ji541/Shared/ekg_warped1.png"  # <-- put full path to your PNG here
output_csv = "/home/ji541/Shared/ekg_digitized2.txt"

# Edge detection parameters
blur_ksize = (5, 5)
canny_low = 50
canny_high = 150

# ====================================

# 1. Load image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: could not load image: {image_path}")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Preprocess & edge detection
blur = cv2.GaussianBlur(gray, blur_ksize, 0)
edges = cv2.Canny(blur, canny_low, canny_high)

height, width = edges.shape

# 3. For each x column, find the y position of the ECG trace
y_positions = np.full(width, np.nan)

# Start tracking near middle of image
prev_y = height // 2

for x in range(width):
    column = edges[:, x]
    ys = np.where(column > 0)[0]  # rows where an edge is present

    if ys.size == 0:
        # no edge in this column; keep previous y (optional)
        y_positions[x] = np.nan
        continue

    # Choose the y closest to previous y to follow a continuous trace
    idx = np.argmin(np.abs(ys - prev_y))
    y = ys[idx]
    y_positions[x] = y
    prev_y = y

# 4. Interpolate missing values (NaNs) if any
nan_mask = np.isnan(y_positions)
if np.any(~nan_mask):  # at least some good points
    good_x = np.where(~nan_mask)[0]
    good_y = y_positions[good_x]
    # Linear interpolation over missing columns
    y_positions[nan_mask] = np.interp(np.where(nan_mask)[0], good_x, good_y)
else:
    print("Error: no trace detected in any column.")
    exit(1)

# 5. Convert pixel coordinates to normalized signal
# In images, y=0 is top; signals usually are plotted with up = positive.
# We'll invert and normalize to roughly [-1, 1].
y_min = np.min(y_positions)
y_max = np.max(y_positions)
y_center = 0.5 * (y_min + y_max)

# Center and scale
signal = -(y_positions - y_center) / (0.5 * (y_max - y_min) + 1e-8)

# 6. Create a normalized time axis from 0 to 1
time = np.linspace(0.0, 1.0, width)

# 7. Save to CSV: columns = time, value
data = np.column_stack((time, signal))
np.savetxt(output_csv, data, delimiter=",", header="time,value", comments="")

print(f"Saved digitized signal to: {output_csv}")
print("First 5 samples:")
print(data[:5, :])

# 8. Optional: visualize overlay of detected trace on original image
overlay = img.copy()
for x in range(width):
    y = int(y_positions[x])
    cv2.circle(overlay, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Original", img)
cv2.imshow("Edges", edges)
cv2.imshow("Trace overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
