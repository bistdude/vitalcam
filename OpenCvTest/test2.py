import cv2

# 1. Load image (PNG or JPG)
img = cv2.imread("/home/ji541/Shared/baby.png")   # <-- put your filename here

if img is None:
    print("Error: image not found!")
    exit()

# 2. Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Gaussian blur (to reduce noise)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Edge detection (Canny)
edges = cv2.Canny(blur, 50, 150)

# 5. Display results
cv2.imshow("Original", img)
cv2.imshow("Grayscale", gray)
cv2.imshow("Blurred", blur)
cv2.imshow("Edges", edges)

# Wait for key press
cv2.waitKey(0)
cv2.destroyAllWindows()