import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
import os

# Global to hold the currently selected image path
current_image_path = None

def select_image():
    global current_image_path
    filetypes = [
        ("Image files", "*.png *.jpg *.jpeg *.bmp"),
        ("All files", "*.*"),
    ]
    filename = filedialog.askopenfilename(
        title="Select EKG image",
        filetypes=filetypes
    )
    if filename:
        current_image_path = filename
        status_var.set(f"Selected: {os.path.basename(filename)}")
    else:
        status_var.set("No image selected")

def warp_image():
    global current_image_path

    if current_image_path is None:
        messagebox.showwarning("No image", "Please select an image first.")
        return

    img = cv2.imread(current_image_path)
    if img is None:
        messagebox.showerror("Error", f"Could not load image:\n{current_image_path}")
        return

    clone = img.copy()
    points = []
    window_name = "Click 4 corners: TL, TR, BR, BL (press 'w' to warp, 'q' to quit)"

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(window_name, clone)
                print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("Instructions:")
    print("  1) Click EXACTLY 4 points on the EKG strip in this order:")
    print("     - Top-left corner")
    print("     - Top-right corner")
    print("     - Bottom-right corner")
    print("     - Bottom-left corner")
    print("  2) Press 'w' when done to warp.")
    print("  3) Press 'q' to cancel without saving.")

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Warping cancelled by user.")
            cv2.destroyAllWindows()
            status_var.set("Warp cancelled")
            return

        if key == ord('w'):
            if len(points) != 4:
                print(f"You selected {len(points)} points. Need 4 points to warp.")
                continue
            else:
                break

    cv2.destroyAllWindows()

    pts = np.array(points, dtype=np.float32)
    (tl, tr, br, bl) = pts

    # Compute dimensions of the warped output
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    # Build output path
    base, ext = os.path.splitext(current_image_path)
    output_path = base + "_warped.png"
    cv2.imwrite(output_path, warped)

    print(f"Warped image saved to: {output_path}")
    status_var.set(f"Warped image saved: {os.path.basename(output_path)}")

    cv2.imshow("Warped EKG", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------ Tkinter GUI ------------------

root = tk.Tk()
root.title("EKG Warping Tool (Pi)")

root.geometry("400x180")

frame = tk.Frame(root, padx=10, pady=10)
frame.pack(fill=tk.BOTH, expand=True)

select_btn = tk.Button(frame, text="Select EKG Image", command=select_image, width=25)
select_btn.pack(pady=5)

warp_btn = tk.Button(frame, text="Warp Selected Image", command=warp_image, width=25)
warp_btn.pack(pady=5)

status_var = tk.StringVar()
status_var.set("No image selected")

status_label = tk.Label(frame, textvariable=status_var, anchor="w", justify="left")
status_label.pack(fill=tk.X, pady=10)

quit_btn = tk.Button(frame, text="Quit", command=root.destroy, width=10)
quit_btn.pack(pady=5)

root.mainloop()
