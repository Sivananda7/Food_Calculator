from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from ultralytics import YOLO
import numpy as np

app = Flask(__name__, static_url_path='/static')

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the YOLOv8 model
model = YOLO('best.pt')

# Macro nutrient dictionary (example values - replace with actual data)
MACRO_NUTRIENTS = {
    'AW cola': {'calories': 140, 'protein': 0, 'carbs': 39, 'fat': 0},
    'Beijing Beef': {'calories': 470, 'protein': 13, 'carbs': 51, 'fat': 26},
    'Chow Mein': {'calories': 510, 'protein': 13, 'carbs': 87, 'fat': 12},
    'Fried Rice': {'calories': 520, 'protein': 11, 'carbs': 85, 'fat': 16},
    'Hashbrown': {'calories': 150, 'protein': 2, 'carbs': 15, 'fat': 9},
    'Honey Walnut Shrimp': {'calories': 360, 'protein': 13, 'carbs': 20, 'fat': 26},
    'Kung Pao Chicken': {'calories': 290, 'protein': 16, 'carbs': 21, 'fat': 16},
    'String Bean Chicken Breast': {'calories': 190, 'protein': 14, 'carbs': 13, 'fat': 9},
    'Super Greens': {'calories': 90, 'protein': 6, 'carbs': 13, 'fat': 3},
    'The Original Orange Chicken': {'calories': 380, 'protein': 13, 'carbs': 47, 'fat': 18},
    'White Steamed Rice': {'calories': 380, 'protein': 7, 'carbs': 87, 'fat': 0},
    'black pepper rice bowl': {'calories': 500, 'protein': 20, 'carbs': 70, 'fat': 15},
    'burger': {'calories': 250, 'protein': 12, 'carbs': 30, 'fat': 10},
    'carrot_eggs': {'calories': 100, 'protein': 6, 'carbs': 5, 'fat': 7},
    'cheese burger': {'calories': 300, 'protein': 15, 'carbs': 30, 'fat': 15},
    'chicken waffle': {'calories': 400, 'protein': 20, 'carbs': 40, 'fat': 20},
    'chicken_nuggets': {'calories': 280, 'protein': 13, 'carbs': 18, 'fat': 17},
    'chinese_cabbage': {'calories': 20, 'protein': 1, 'carbs': 3, 'fat': 0},
    'chinese_sausage': {'calories': 200, 'protein': 9, 'carbs': 1, 'fat': 18},
    'crispy corn': {'calories': 150, 'protein': 3, 'carbs': 30, 'fat': 5},
    'curry': {'calories': 250, 'protein': 10, 'carbs': 20, 'fat': 15},
    'french fries': {'calories': 365, 'protein': 4, 'carbs': 48, 'fat': 17},
    'fried chicken': {'calories': 320, 'protein': 20, 'carbs': 12, 'fat': 22},
    'fried_chicken': {'calories': 320, 'protein': 20, 'carbs': 12, 'fat': 22},
    'fried_dumplings': {'calories': 250, 'protein': 8, 'carbs': 30, 'fat': 12},
    'fried_eggs': {'calories': 90, 'protein': 6, 'carbs': 1, 'fat': 7},
    'mango chicken pocket': {'calories': 300, 'protein': 15, 'carbs': 35, 'fat': 12},
    'mozza burger': {'calories': 520, 'protein': 28, 'carbs': 41, 'fat': 29},
    'mung_bean_sprouts': {'calories': 30, 'protein': 3, 'carbs': 6, 'fat': 0},
    'nugget': {'calories': 280, 'protein': 13, 'carbs': 18, 'fat': 17},
    'perkedel': {'calories': 100, 'protein': 3, 'carbs': 12, 'fat': 5},
    'rice': {'calories': 200, 'protein': 4, 'carbs': 44, 'fat': 0},
    'sprite': {'calories': 140, 'protein': 0, 'carbs': 37, 'fat': 0},
    'tostitos cheese dip sauce': {'calories': 50, 'protein': 2, 'carbs': 2, 'fat': 4},
    'triangle_hash_brown': {'calories': 150, 'protein': 2, 'carbs': 15, 'fat': 9},
    'water_spinach': {'calories': 20, 'protein': 2, 'carbs': 3, 'fat': 0}
}

@app.route('/favicon.ico')
def favicon():
    return app.send_static_file('favicon.ico')

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
            total_macros = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    class_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    food_name = model.names[class_id]
                    detections.append({
                        'class': food_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2]
                    })
                    
                    # Add macros for detected food
                    if food_name in MACRO_NUTRIENTS:
                        for macro, value in MACRO_NUTRIENTS[food_name].items():
                            total_macros[macro] += value
            
            return jsonify({
                'filename': filename,
                'detections': detections,
                'total_macros': total_macros
            })
    return render_template('index.html')

if __name__ == '__main__':
   app.run(debug=True)