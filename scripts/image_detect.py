from ultralytics import YOLO
import cv2

try:
    model = YOLO('../models/yolo11x.pt') 
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have an internet connection to download the model on the first run.")
    exit()

image_path = '../datasets/Hetra/IITM-HeTra_v2/Dataset-1/images/frame_1011.jpg' 

try:
    results = model(image_path, device=0) # device=0 GPU, or device='cpu'
except FileNotFoundError:
    print(f"Error:'{image_path}' was not found.")
    exit()
except Exception as e:
    print(f"An error occurred during detection: {e}")
    exit()


result = results[0]

annotated_frame = result.plot()

cv2.imshow('YOLOv8 Detection', annotated_frame)

print(f"Found {len(result.boxes)} objects in the image.")
for box in result.boxes:
    class_id = int(box.cls[0])
    class_name = model.names[class_id]
    
    confidence = float(box.conf[0])
    
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    print(f"- Detected '{class_name}' with {confidence:.2f} confidence at [{x1}, {y1}, {x2}, {y2}]")


print("\nPress any key to close the image window...")
cv2.waitKey(0)

cv2.destroyAllWindows()
