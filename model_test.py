import torch
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load the YOLO model
model = YOLO("best.pt")  # Replace 'best.pt' with the path to your trained YOLO model

# Function to display images using OpenCV
def show_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Load the car image
image_path = "car.jpg"  # Replace with the path to your input image
image = cv2.imread(image_path)
if image is None:
    print("Error: Could not load the image.")
    exit()

# Perform inference on the image
results = model(image)

# Process the results
for result in results:
    boxes = result.boxes  # Extract bounding boxes
    for box in boxes:
        # Extract coordinates and confidence
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Bounding box coordinates
        confidence = box.conf[0]  # Confidence score
        class_id = int(box.cls[0])  # Class ID
        
        # Draw the bounding box and label on the image
        label = f"Plate: {confidence:.2f}"
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Save or display the result
output_path = "output.jpg"
cv2.imwrite(output_path, image)
print(f"Processed image saved to: {output_path}")

# Display the result
show_image(image, "Detected Number Plates")
