from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import json
import gc
import time
import threading
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Configure TensorFlow for optimal performance[2][3]
try:
    # Enable memory growth for GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    # Optimize CPU usage
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(4)
except:
    print("GPU optimization not available, using CPU")

app = Flask(__name__)

# Thread lock for thread-safe predictions
prediction_lock = threading.Lock()

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'crop_disease_model.h5')
CLASS_NAMES_PATH = os.path.join(MODEL_DIR, 'class_names.json')

# Load the trained model (following search result [2] best practices)
print("Loading model...")
try:
    # Load model without compilation for faster loading[2]
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure 'crop_disease_model.h5' exists in the 'model' directory")
    exit(1)

# Load class names from JSON file
try:
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)
    print(f"✅ Loaded {len(class_names)} class names from JSON")
except Exception as e:
    print(f"❌ Error loading class names: {e}")
    print("Using fallback class names")
    # Comprehensive fallback class names for common crop diseases[1][4]
    class_names = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
        'Apple___healthy', 'Corn_(maize)___Cercospora_leaf_spot', 
        'Corn_(maize)___Common_rust', 'Corn_(maize)___Northern_Leaf_Blight',
        'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)',
        'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
        'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
        'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
        'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites', 
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
        'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
    ]

# Comprehensive disease solutions dictionary[1]
disease_solutions = {
    # Apple diseases
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "solution": "Apply fungicide spray containing captan or myclobutanil every 7-14 days during wet weather",
        "prevention": "Ensure good air circulation, avoid overhead watering, remove fallen leaves, plant resistant varieties"
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot", 
        "solution": "Remove infected fruits and branches, apply copper-based fungicide, improve air circulation",
        "prevention": "Proper pruning for air circulation, avoid wounding trees, remove mummified fruits"
    },
    "Apple___Cedar_apple_rust": {
        "name": "Cedar Apple Rust",
        "solution": "Apply preventive fungicide sprays containing propiconazole in early spring",
        "prevention": "Remove nearby cedar trees if possible, plant resistant apple varieties"
    },
    "Apple___healthy": {
        "name": "Healthy Apple",
        "solution": "No treatment needed - continue current care practices",
        "prevention": "Maintain proper spacing, regular inspection, balanced fertilization"
    },
    
    # Corn diseases
    "Corn_(maize)___Cercospora_leaf_spot": {
        "name": "Corn Cercospora Leaf Spot",
        "solution": "Apply strobilurin or triazole fungicides when symptoms first appear",
        "prevention": "Crop rotation with non-grass crops, deep tillage, remove crop debris"
    },
    "Corn_(maize)___Common_rust": {
        "name": "Corn Common Rust",
        "solution": "Apply fungicides containing propiconazole or tebuconazole if severe",
        "prevention": "Plant resistant hybrids, ensure proper plant spacing, monitor weather conditions"
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "name": "Northern Corn Leaf Blight",
        "solution": "Apply fungicide containing strobilurin or triazole active ingredients",
        "prevention": "Use resistant varieties, crop rotation, tillage to bury residue"
    },
    "Corn_(maize)___healthy": {
        "name": "Healthy Corn",
        "solution": "No treatment needed - maintain current management practices",
        "prevention": "Continue proper fertilization, irrigation, and pest monitoring"
    },
    
    # Grape diseases
    "Grape___Black_rot": {
        "name": "Grape Black Rot",
        "solution": "Apply fungicides containing mancozeb or copper compounds",
        "prevention": "Remove infected berries and leaves, ensure good air circulation"
    },
    "Grape___Esca_(Black_Measles)": {
        "name": "Grape Esca (Black Measles)",
        "solution": "Remove infected wood, apply wound protectants, consider trunk renewal",
        "prevention": "Avoid wounds during pruning, use clean pruning tools, proper canopy management"
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "name": "Grape Leaf Blight",
        "solution": "Apply copper-based fungicides or systemic fungicides",
        "prevention": "Improve air circulation, avoid overhead irrigation, remove fallen leaves"
    },
    "Grape___healthy": {
        "name": "Healthy Grape",
        "solution": "No treatment needed - continue monitoring",
        "prevention": "Maintain proper pruning, canopy management, and disease monitoring"
    },
    
    # Citrus diseases
    "Orange___Haunglongbing_(Citrus_greening)": {
        "name": "Citrus Huanglongbing (Greening)",
        "solution": "Remove infected trees immediately, control psyllid vectors with insecticides",
        "prevention": "Use certified disease-free nursery stock, control Asian citrus psyllid"
    },
    
    # Peach diseases
    "Peach___Bacterial_spot": {
        "name": "Peach Bacterial Spot",
        "solution": "Apply copper-based bactericides during dormant season and early growing season",
        "prevention": "Plant resistant varieties, avoid overhead irrigation, improve air circulation"
    },
    "Peach___healthy": {
        "name": "Healthy Peach",
        "solution": "No treatment needed - continue regular care",
        "prevention": "Proper pruning, balanced nutrition, regular monitoring"
    },
    
    # Pepper diseases
    "Pepper,_bell___Bacterial_spot": {
        "name": "Pepper Bacterial Spot",
        "solution": "Apply copper-based bactericides, remove infected plants",
        "prevention": "Use certified disease-free seeds, avoid overhead watering, crop rotation"
    },
    "Pepper,_bell___healthy": {
        "name": "Healthy Pepper",
        "solution": "No treatment needed",
        "prevention": "Continue current growing practices, monitor for pests and diseases"
    },
    
    # Potato diseases
    "Potato___Early_blight": {
        "name": "Potato Early Blight",
        "solution": "Apply fungicides containing chlorothalonil or azoxystrobin",
        "prevention": "Crop rotation, avoid overhead irrigation, remove infected debris"
    },
    "Potato___Late_blight": {
        "name": "Potato Late Blight",
        "solution": "Apply systemic fungicides immediately, destroy infected plants",
        "prevention": "Use certified seed potatoes, avoid wet conditions, apply preventive fungicides"
    },
    "Potato___healthy": {
        "name": "Healthy Potato",
        "solution": "No treatment needed",
        "prevention": "Continue proper cultural practices and monitoring"
    },
    
    # Strawberry diseases
    "Strawberry___Leaf_scorch": {
        "name": "Strawberry Leaf Scorch",
        "solution": "Apply fungicides containing myclobutanil or propiconazole",
        "prevention": "Plant resistant varieties, ensure good air circulation, avoid overhead watering"
    },
    "Strawberry___healthy": {
        "name": "Healthy Strawberry",
        "solution": "No treatment needed",
        "prevention": "Maintain proper spacing, remove old leaves, monitor regularly"
    },
    
    # Tomato diseases
    "Tomato___Bacterial_spot": {
        "name": "Tomato Bacterial Spot",
        "solution": "Apply copper-based bactericides, remove severely infected plants",
        "prevention": "Use pathogen-free seeds, avoid overhead watering, crop rotation"
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "solution": "Apply fungicides containing chlorothalonil or copper compounds weekly",
        "prevention": "Mulching, drip irrigation, crop rotation, remove lower leaves"
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight", 
        "solution": "Apply copper fungicides immediately, remove infected plants completely",
        "prevention": "Avoid overhead watering, ensure good ventilation, use resistant varieties"
    },
    "Tomato___Leaf_Mold": {
        "name": "Tomato Leaf Mold",
        "solution": "Improve ventilation, apply fungicides containing chlorothalonil",
        "prevention": "Reduce humidity, increase air circulation, avoid overhead watering"
    },
    "Tomato___Septoria_leaf_spot": {
        "name": "Tomato Septoria Leaf Spot",
        "solution": "Apply fungicides containing copper or chlorothalonil",
        "prevention": "Mulch around plants, avoid overhead watering, remove infected debris"
    },
    "Tomato___Spider_mites": {
        "name": "Tomato Spider Mites",
        "solution": "Apply miticides or insecticidal soaps, increase humidity around plants",
        "prevention": "Regular monitoring, avoid water stress, encourage beneficial insects"
    },
    "Tomato___Target_Spot": {
        "name": "Tomato Target Spot",
        "solution": "Apply fungicides containing azoxystrobin or chlorothalonil",
        "prevention": "Crop rotation, remove plant debris, avoid overhead irrigation"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name": "Tomato Yellow Leaf Curl Virus",
        "solution": "Remove infected plants, control whitefly vectors with insecticides",
        "prevention": "Use virus-free transplants, control whiteflies, use reflective mulches"
    },
    "Tomato___Tomato_mosaic_virus": {
        "name": "Tomato Mosaic Virus",
        "solution": "Remove infected plants immediately, disinfect tools between plants",
        "prevention": "Use certified disease-free seeds, avoid tobacco use near plants"
    },
    "Tomato___healthy": {
        "name": "Healthy Tomato",
        "solution": "No treatment needed - excellent plant health",
        "prevention": "Continue current practices: proper spacing, nutrition, and monitoring"
    }
}

def model_predict(image_array, model):
    """
    Enhanced prediction function based on search results[2]
    """
    try:
        # Use thread-safe prediction
        with prediction_lock:
            # Make prediction with optimized settings
            predictions = model.predict(image_array, batch_size=1, verbose=0)
            return predictions
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

@app.route('/')
def index():
    """Main page route"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Enhanced prediction endpoint with comprehensive error handling[2][3]
    """
    start_time = time.time()
    
    try:
        print("🔍 === PREDICTION REQUEST RECEIVED ===")
        
        # FIXED: Changed 'image' to 'file' to match HTML form name attribute
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']  # FIXED: Changed from 'image' to 'file'
        
        if file.filename == '':
            print("❌ No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"📁 Processing file: {file.filename}")
        
        # Read and validate image
        try:
            file_content = file.read()
            
            if len(file_content) == 0:
                return jsonify({'error': 'Empty file uploaded'}), 400
            
            # Create image stream
            image_stream = io.BytesIO(file_content)
            image = Image.open(image_stream)
            
            print(f"📊 Image loaded: {image.size}, Mode: {image.mode}")
            
        except Exception as e:
            print(f"❌ Image loading error: {e}")
            return jsonify({'error': f'Invalid image format: {str(e)}'}), 400
        
        # Image preprocessing (following training script requirements)
        processing_start = time.time()
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            print("🔄 Converted to RGB")
        
        # Resize to match training size (160x160 as per training script)
        IMG_SIZE = 160  # Match your training script
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
        print(f"📏 Resized to {IMG_SIZE}x{IMG_SIZE}")
        
        # Convert to numpy array and preprocess
        image_array = np.array(image, dtype=np.float32)
        image_array = np.expand_dims(image_array, axis=0)
        
        # Apply MobileNetV2 preprocessing (matching training)
        image_array = preprocess_input(image_array)
        
        print(f"⏱️ Preprocessing completed in {time.time() - processing_start:.3f}s")
        
        # Make prediction
        prediction_start = time.time()
        print("🤖 Making prediction...")
        
        predictions = model_predict(image_array, model)
        
        if predictions is None:
            return jsonify({'error': 'Prediction failed'}), 500
        
        print(f"⏱️ Prediction completed in {time.time() - prediction_start:.3f}s")
        
        # Process results
        predicted_class_index = np.argmax(predictions)
        confidence = float(np.max(predictions))
        
        print(f"📊 Prediction: Index {predicted_class_index}, Confidence: {confidence:.3f}")
        
        # Validate prediction index
        if predicted_class_index >= len(class_names):
            print(f"❌ Invalid prediction index: {predicted_class_index} >= {len(class_names)}")
            return jsonify({'error': 'Invalid prediction result'}), 500
        
        predicted_disease = class_names[predicted_class_index]
        print(f"🔬 Predicted disease: {predicted_disease}")
        
        # Get solution information
        solution_info = disease_solutions.get(predicted_disease, {
            "name": predicted_disease.replace('___', ' - ').replace('_', ' ').title(),
            "solution": "Consult an agricultural expert for specific treatment recommendations",
            "prevention": "Regular monitoring and proper crop management practices recommended"
        })
        
        # Prepare response
        response = {
            'disease': solution_info['name'],
            'confidence': f"{confidence*100:.1f}%",
            'solution': solution_info['solution'],
            'prevention': solution_info['prevention'],
            'processing_time': f"{time.time() - start_time:.2f}s"
        }
        
        print(f"✅ Response prepared: {solution_info['name']} ({confidence*100:.1f}%)")
        print(f"⏱️ Total processing time: {time.time() - start_time:.2f}s")
        
        # Memory cleanup
        gc.collect()
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f'Error processing image: {str(e)}'
        print(f"❌ {error_msg}")
        
        # Import traceback for detailed error info
        import traceback
        traceback.print_exc()
        
        # Memory cleanup on error
        gc.collect()
        
        return jsonify({'error': error_msg}), 500
    
    finally:
        print(f"⏱️ Total request time: {time.time() - start_time:.2f}s")
        print("🔍 === PREDICTION REQUEST COMPLETED ===\n")

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint with model validation[2]
    """
    try:
        # Test model prediction capability
        test_start = time.time()
        
        # Create test input matching model requirements
        test_input = np.random.random((1, 160, 160, 3)).astype('float32')
        test_input = preprocess_input(test_input)
        
        # Test prediction
        test_predictions = model.predict(test_input, verbose=0)
        test_time = time.time() - test_start
        
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'model_input_shape': str(model.input_shape),
            'model_output_shape': str(model.output_shape),
            'classes_count': len(class_names),
            'test_prediction_time': f"{test_time:.3f}s",
            'diseases_supported': len(disease_solutions)
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'model_loaded': False
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of supported disease classes"""
    return jsonify({
        'classes': class_names,
        'count': len(class_names),
        'supported_diseases': list(disease_solutions.keys())
    })

@app.route('/debug-static')
def debug_static():
    """Debug static file serving"""
    import os
    
    static_path = os.path.join(app.root_path, 'static')
    
    try:
        static_files = []
        if os.path.exists(static_path):
            static_files = os.listdir(static_path)
        
        return f"""
        <h1>Static File Debug</h1>
        <p><strong>App root path:</strong> {app.root_path}</p>
        <p><strong>Static folder path:</strong> {static_path}</p>
        <p><strong>Static folder exists:</strong> {os.path.exists(static_path)}</p>
        <p><strong>Files in static folder:</strong> {static_files}</p>
        <p><strong>CSS URL:</strong> <a href="{url_for('static', filename='style.css')}">{url_for('static', filename='style.css')}</a></p>
        <p><strong>JS URL:</strong> <a href="{url_for('static', filename='script.js')}">{url_for('static', filename='script.js')}</a></p>
        """
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    print("🚀 Starting Crop Disease Detection Flask App...")
    print(f"📂 Model path: {MODEL_PATH}")
    print(f"📋 Classes loaded: {len(class_names)}")
    print(f"💊 Disease solutions: {len(disease_solutions)}")
    print(f"🌐 Server starting at: http://localhost:5000")
    print("✅ Check http://127.0.0.1:5000/health for health status")
    
    # Run Flask app with optimized settings
    app.run(
        debug=False,  # Disable debug for production
        host='0.0.0.0', 
        port=5000, 
        threaded=True,  # Enable threading for concurrent requests
        use_reloader=False  # Prevent double loading in development
    )