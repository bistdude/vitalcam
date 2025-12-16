import cv2
import numpy as np

# ====== CONFIG ======
image_path = "/home/ji541/Shared/ekg1.png"       # <-- update to your full path
output_path = "/home/ji541/Shared/ekg_warped.png"
# ====================

img = cv2.imread(image_path)
if img is None:
    print(f"Error: could not load image: {image_path}")
    exit(1)

clone = img.copy()
points = []

window_name = "Click 4 corners: TL, TR, BR, BL"

def mouse_callback(event, x, y, flags, param):
    global points, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            # draw small circles to show selected points
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(window_name, clone)
            print(f"Point {len(points)}: ({x}, {y})")

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, clone)
cv2.setMouseCallback(window_name, mouse_callback)

print("Instructions:")
print("  1) Click EXACTLY 4 points in this order on the EKG strip:")
print("     - Top-left corner")
print("     - Top-right corner")
print("     - Bottom-right corner")
print("     - Bottom-left corner")
print("  2) Press 'w' when done to warp.")
print("  3) Press 'q' to quit without saving.")

while True:
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("Quitting without warp.")
        cv2.destroyAllWindows()
        exit(0)

    if key == ord('w'):
        if len(points) != 4:
            print(f"You selected {len(points)} points. Need 4 points to warp.")
            continue
        else:
            break

cv2.destroyAllWindows()

# Convert points to float32
pts = np.array(points, dtype=np.float32)

# Order: TL, TR, BR, BL
(tl, tr, br, bl) = pts

# Compute width of the new image
widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))

# Compute height of the new image
heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype=np.float32)

# Perspective transform matrix
M = cv2.getPerspectiveTransform(pts, dst)

# Warp the image
warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

# Save and show
cv2.imwrite(output_path, warped)
print(f"Warped EKG image saved to: {output_path}")

cv2.imshow("Warped EKG", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
