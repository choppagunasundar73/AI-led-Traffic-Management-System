import cv2
from ultralytics import YOLO
import numpy as np

# --- Configuration ---
# IMPORTANT: Set these paths correctly
VIDEO_PATH = '../datasets/INDRA/Videos/Videos/video1.MOV'  # The video you want to test on
MODEL_PATH = 'yolo11x.pt'             # Your trained YOLO model

# Confidence threshold: Detections with a score below this will be ignored
CONF_THRESHOLD = 0.5

# --- Main Application ---

def print_instructions():
    """Prints the control keys to the console."""
    print("\n" + "="*50)
    print("--- Manual Evaluation Tool ---")
    print("="*50)
    print("Controls:")
    print("  [SPACE]  -> Pause / Resume Video")
    print("  [D]      -> Next Frame (when paused)")
    print("  [A]      -> Previous Frame (when paused)")
    print("  [Q]      -> Quit")
    print("="*50 + "\n")

# 1. Load the YOLO model
print("Loading model...")
try:
    model = YOLO(MODEL_PATH)
    class_names = model.names
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Open the video file
print(f"Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"Error: Could not open video file at {VIDEO_PATH}")
    exit()

# Get video properties
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 3. Initialize variables for playback control
paused = False
frame_number = 0

print_instructions()

# 4. Main loop to process the video
while True:
    # Handle frame seeking when paused
    if paused:
        key = cv2.waitKey(0) & 0xFF
    else:
        # Move to the next frame automatically if not paused
        frame_number += 1
        if frame_number >= total_frames:
            break # End of video
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        key = cv2.waitKey(int(1000/fps)) & 0xFF # Wait for a duration based on FPS

    # Read the current frame
    success, frame = cap.read()
    if not success:
        print("End of video or cannot read frame.")
        break
    
    # --- Keyboard Controls ---
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord(' '):
        paused = not paused
        print("Paused" if paused else "Resumed")
        continue # Skip the rest of the loop to immediately reflect pause/resume
    elif key == ord('d') and paused:
        frame_number = min(frame_number + 1, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        print(f"Frame: {frame_number}/{total_frames}")
    elif key == ord('a') and paused:
        frame_number = max(frame_number - 1, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        print(f"Frame: {frame_number}/{total_frames}")

    # If the video is playing, we don't need to re-run prediction on the same frame
    if paused and key not in [ord('d'), ord('a')]:
        # We are paused but not advancing frame, so just show the current frame again
        cv2.imshow("Model Evaluation", annotated_frame)
        continue

    # 5. Run YOLO model on the current frame
    results = model.track(frame, persist = True, conf=CONF_THRESHOLD,classes = [2,3,5,7], verbose=False, device = 0)
    annotated_frame = results[0].plot() # .plot() conveniently draws boxes, labels, and conf

    # Add frame number and status text to the frame
    status_text = "PAUSED" if paused else "PLAYING"
    cv2.putText(annotated_frame, f"Frame: {frame_number}/{total_frames}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(annotated_frame, f"Status: {status_text}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


    # 6. Display the frame
    cv2.imshow("Model Evaluation", annotated_frame)


# 7. Cleanup
cap.release()
cv2.destroyAllWindows()
print("Evaluation tool closed.")
