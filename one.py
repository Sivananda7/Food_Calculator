import cv2
from ultralytics import YOLO
import os

# Load the model
model = YOLO('best.pt')

# Function to process an image
def process_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    
    # Run YOLOv8 inference on the image
    results = model(img)

    # Create a copy of the original image for annotation
    annotated_img = img.copy()

    # Iterate through the detections and draw them on the image
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get class and confidence
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Draw bounding box
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f'{model.names[cls]} {conf:.2f}'
            cv2.putText(annotated_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the annotated image
    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, annotated_img)

    # Generate text output
    text_output = []
    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls[0])]
            confidence = float(box.conf[0])
            text_output.append(f"{class_name}: {confidence:.2f}")

    return output_path, text_output

# Example usage
input_image_path = 'C:/Users/kaavi/OneDrive/Desktop/food/hash.jpeg'
output_image_path, predictions = process_image(input_image_path)

print(f"Annotated image saved as: {output_image_path}")
print("Predictions:")
for pred in predictions:
    print(pred)