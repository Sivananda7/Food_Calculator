from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import cv2
import numpy as np

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('best.pt')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Perform inference
            results = model(filepath)
            
            # Process results
            detections = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    detections.append({
                        'class': model.names[class_id],
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
            
            return jsonify({
                'filename': filename,
                'detections': detections
            })
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)