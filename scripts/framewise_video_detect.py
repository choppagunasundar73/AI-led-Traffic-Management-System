from ultralytics import YOLO
import cv2
import collections
import os
import time

# Load your YOLO model
model = YOLO('yolov8x.pt')
# Get the class names from the model
class_names = model.names

# --- Configuration for your image sequence ---
image_folder_path = r'..\datasets\Hetra\IITM-HeTra_v2\Dataset-1\images'
num_frames = 1417 # The total number of frames to process

# --- Get frame dimensions from the first image ---
first_image_path = os.path.join(image_folder_path, 'frame_1.jpg')
if not os.path.exists(first_image_path):
    print(f"Error: The first frame was not found at {first_image_path}")
    exit()

# --- Data structures for unique counting ---
# A dictionary to store the count for each class
class_counts = {}
# A single set to store the IDs of vehicles that have already been counted
counted_track_ids = set()

# --- Main loop to iterate through image files ---
for i in range(1, num_frames + 1):
    time.sleep(0.5)
    image_filename = f'frame_{i}.jpg'
    full_image_path = os.path.join(image_folder_path, image_filename)

    if not os.path.exists(full_image_path):
        print(f"Warning: Frame {image_filename} not found, skipping.")
        continue

    frame = cv2.imread(full_image_path)
    if frame is None:
        print(f"Warning: Could not read frame {image_filename}, skipping.")
        continue

    # Use the tracker on the entire frame
    results = model.track(frame, persist=True, conf=0.1, classes=[2, 3, 5, 7], device=0)

    # Plot the tracking results on the frame
    annotated_frame = results[0].plot(line_width=2)

    # Check if any objects were tracked
    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist()

        # Process each tracked object
        for track_id, cls in zip(track_ids, clss):
            # --- THE CORE LOGIC ---
            # If the current vehicle's ID has NOT been counted yet:
            if track_id not in counted_track_ids:
                # 1. Mark this ID as counted by adding it to the set.
                counted_track_ids.add(track_id)
                
                # 2. Increment the count for its specific class.
                class_name = class_names[cls]
                class_counts.setdefault(class_name, 0)
                class_counts[class_name] += 1

    # Display the total vehicle count on the frame
    total_vehicle_count = len(counted_track_ids)
    cv2.putText(annotated_frame, f'Unique Vehicle Count: {total_vehicle_count}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Unique Vehicle Counter", annotated_frame)

    # Allow exiting with the 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

# --- Print the final detailed tally to the console ---
print("\n" + "="*40)
print("--- Final Unique Vehicle Tally ---")
sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

if not counted_track_ids:
    print("No vehicles were detected.")
else:
    for class_name, count in sorted_classes:
        print(f"\n--- Class: {class_name.upper()} ---")
        print(f"  Unique Count: {count}")

print("\n" + "="*40)
print(f"Grand Total All Unique Vehicles: {len(counted_track_ids)}")
print(f"All Counted Track IDs: {sorted(list(counted_track_ids))}")
print("="*40)