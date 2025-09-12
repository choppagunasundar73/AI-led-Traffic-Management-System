import cv2
import os

# ---------------- CONFIG ----------------
image_folder_path = r'..\datasets\Hetra\IITM-HeTra_v2\Dataset-1\images'
labels_folder_path = r'..\datasets\Hetra\IITM-HeTra_v2\Dataset-1\labels_auto'
num_frames = 1417
min_contour_area = 500  # adjust for motorcycle size (in pixels)

# Create labels folder if it doesn't exist
os.makedirs(labels_folder_path, exist_ok=True)

# Create background subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

# First pass to build background model
for i in range(1, num_frames + 1):
    image_filename = f'frame_{i}.jpg'
    full_image_path = os.path.join(image_folder_path, image_filename)
    if not os.path.exists(full_image_path):
        continue
    frame = cv2.imread(full_image_path)
    fgbg.apply(frame)

print("✅ Background model built. Now extracting objects...")

# Second pass for detection + labeling
for i in range(1, num_frames + 1):
    image_filename = f'frame_{i}.jpg'
    full_image_path = os.path.join(image_folder_path, image_filename)
    if not os.path.exists(full_image_path):
        continue

    frame = cv2.imread(full_image_path)
    fgmask = fgbg.apply(frame)

    # Threshold and clean
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, (5,5))
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w, _ = frame.shape
    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_path = os.path.join(labels_folder_path, label_filename)

    with open(label_path, "w") as f:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)

            # Convert to YOLO format (class 0 for motorcycle)
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            width = bw / w
            height = bh / h

            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    # Optional: visualize detections
    vis_frame = frame.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_contour_area:
            x, y, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(vis_frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
    cv2.imshow("Detections", vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"✅ Auto labels saved in {labels_folder_path}")
