# app.py
from flask import Flask, render_template, Response, request, jsonify
import cv2
from ultralytics import YOLO
import os
from werkzeug.utils import secure_filename
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
video_source = None
selected_models = []
cap = None
is_running = False

# Initialize models dictionary
MODELS_PATH = "."  # Current directory where models are located
models_dict = {}

def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {str(e)}")
        return None

@app.route('/')
def index():
    models = [
        'accident_detection.pt',
        'activity_detection.pt',
        'shoplift.pt',
        'yolov8n-pose.pt',
        'precrime.pt',
        'weapon_detection.pt'
    ]
    return render_template('index.html', models=models)

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global video_source, selected_models, cap, is_running
    
    try:
        # Get form data
        if request.content_type == 'application/json':
            data = request.get_json()
            selected_models = data.get('models', [])
        else:
            data = request.form
            selected_models = request.form.getlist('models')  # Use getlist for form data
        
        # Ensure selected_models is a list
        if isinstance(selected_models, str):
            selected_models = [selected_models]
        
        source_type = data.get('source_type')
        
        # Load selected models
        for model_name in selected_models:
            if model_name not in models_dict:
                model_path = os.path.join(MODELS_PATH, model_name)
                if os.path.exists(model_path):
                    models_dict[model_name] = load_model(model_path)
                else:
                    print(f"Model file not found: {model_path}")
                    return jsonify({'status': 'error', 'message': f'Model file not found: {model_name}'}), 404
        
        # Release existing capture if any
        if cap is not None:
            cap.release()
            time.sleep(0.1)  # Small delay to ensure proper release
            cap = None
        
        # Initialize video capture
        if source_type == 'webcam':
            print("Initializing webcam...")
            cap = cv2.VideoCapture(0)
            if not cap or not cap.isOpened():
                print("Failed to open webcam")
                return jsonify({'status': 'error', 'message': 'Failed to open webcam'}), 500
        else:
            video_file = request.files.get('video')
            if video_file:
                filename = secure_filename(video_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                video_file.save(filepath)
                print(f"Opening video file: {filepath}")
                cap = cv2.VideoCapture(filepath)
                if not cap or not cap.isOpened():
                    print("Failed to open video file")
                    return jsonify({'status': 'error', 'message': 'Failed to open video file'}), 500
            else:
                return jsonify({'status': 'error', 'message': 'No video file provided'}), 400
        
        print("Video capture initialized successfully")
        is_running = True
        return jsonify({'status': 'success'})
    
    except Exception as e:
        print(f"Error in start_detection: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

def generate_frames():
    global cap, is_running
    
    print("Starting frame generation...")
    
    if cap is None or not cap.isOpened():
        print("Video capture is not initialized")
        return
    
    try:
        while is_running:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame")
                break
            
            # Process frame with selected models
            for model_name in selected_models:
                model = models_dict.get(model_name)
                if model:
                    try:
                        results = model(frame)
                        frame = results[0].plot()
                    except Exception as e:
                        print(f"Error processing frame with model {model_name}: {str(e)}")
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Failed to encode frame")
                break
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Small delay to control frame rate
            time.sleep(0.01)
            
    except Exception as e:
        print(f"Error in generate_frames: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
            print("Video capture released")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_detection')
def stop_detection():
    global cap, is_running
    is_running = False
    if cap is not None:
        cap.release()
        cap = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)