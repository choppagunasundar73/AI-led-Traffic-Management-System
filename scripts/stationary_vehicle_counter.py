from ultralytics import YOLO
import cv2
import collections
import time

# Load your YOLO model
model = YOLO('yolo11x.pt') 
# Get the class names from the model
class_names = model.names

# Specify your video source
video_path = '../testfiles/videos/2.mp4'
cap = cv2.VideoCapture(video_path)

# --- Configuration for the counting lines ---
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
line_y = 0
x_bound = int(frame_width * 0.4)

# --- NEW: Data structures for per-class counting ---
# Store the track history
track_history = collections.defaultdict(lambda: [])
# Store counts and counted IDs for each class
class_counts = {}
counted_class_ids = {}

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Use the tracker
    # The 'truck' class (ID 7) is included, which covers Class 8 trucks
    results = model.track(frame, persist=True, conf=0.5, classes=[2, 3, 5, 7], device=0)
    # time.sleep(0.1)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.int().cpu().tolist() # Get class IDs

        annotated_frame = results[0].plot(line_width=2)
        
        # Draw boundary lines
        cv2.line(annotated_frame, (x_bound, 0), (x_bound, frame_height), (255, 255, 0), 2)
        cv2.line(annotated_frame, (x_bound, line_y), (frame_width, line_y), (0, 255, 0), 3)

        # Process each tracked object
        for box, track_id, cls in zip(boxes, track_ids, clss):
            x, y, w, h = box
            center_x = int(x)
            center_y = int(y)
            class_name = class_names[cls] # Get class name from ID

            track = track_history[track_id]
            track.append(center_y)
            if len(track) > 2:
                track.pop(0)

            # Check if this track_id has been counted for any class
            # This prevents a single object from being counted multiple times if its class flickers
            is_counted = any(track_id in ids for ids in counted_class_ids.values())

            # UPDATED: Counting logic with per-class tracking
            if len(track) == 2 and not is_counted:
                if track[1] >= line_y and center_x > x_bound:
                    # Initialize class in dictionaries if not present
                    class_counts.setdefault(class_name, 0)
                    counted_class_ids.setdefault(class_name, set())

                    # Increment count and add ID for the specific class
                    class_counts[class_name] += 1
                    counted_class_ids[class_name].add(track_id)
                    
                    cv2.line(annotated_frame, (x_bound, line_y), (frame_width, line_y), (0, 0, 255), 3)

    else:
        annotated_frame = frame
        cv2.line(annotated_frame, (x_bound, 0), (x_bound, frame_height), (255, 255, 0), 2)
        cv2.line(annotated_frame, (x_bound, line_y), (frame_width, line_y), (0, 255, 0), 3)

    # UPDATED: Display the total vehicle count on the frame
    total_vehicle_count = sum(class_counts.values())
    cv2.putText(annotated_frame, f'Total Vehicle Count: {total_vehicle_count}', (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow("Vehicle Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# --- NEW: Verbose printing at the end ---
print("\n" + "="*40)
print("--- Final Vehicle Tally ---")
print("="*40)

# Get the total count for sorting purposes, but handle cases where no vehicles were counted
total_vehicles = sum(class_counts.values())

if total_vehicles == 0:
    print("No vehicles were counted.")
else:
    # Sort classes by count in descending order
    sorted_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)

    for class_name, count in sorted_classes:
        print(f"\n--- Class: {class_name.upper()} ---")
        print(f"  Total Count: {count}")
        
        # Get the sorted list of track IDs for this class
        ids = sorted(list(counted_class_ids.get(class_name, set())))
        print(f"  Track IDs: {ids}")

print("\n" + "="*40)
print(f"Grand Total All Vehicles: {total_vehicles}")
print("="*40)