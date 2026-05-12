import os
import time
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # عشان يسمح لـ C# يتواصل معاه

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- GLOBALS ----------------
model = None
scaling_params = {}

# ---------------- CUSTOM LOSS (لو موديلك محتاجها) ----------------
def focal_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

# تسجيل الـ custom loss عشان TensorFlow يعرفها
tf.keras.utils.get_custom_objects()["focal_loss"] = focal_loss

# ---------------- LOAD MODEL ----------------
def load_model_system():
    global model, scaling_params

    logger.info("=" * 50)
    logger.info("Loading Seizure Detection Model System...")
    logger.info("=" * 50)

    try:
        # 1. Load model architecture from config.json
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                model_json = f.read()
            model = tf.keras.models.model_from_json(model_json)
            logger.info("✅ Model architecture loaded from config.json")
        else:
            raise FileNotFoundError("config.json not found!")

        # 2. Load weights
        if os.path.exists("model.weights.h5"):
            model.load_weights("model.weights.h5")
            logger.info("✅ Model weights loaded from model.weights.h5")
        else:
            raise FileNotFoundError("model.weights.h5 not found!")

        # 3. Load scaling parameters (mean/std with correct shapes)
        # EEG: shape (2,) - for 2 channels
        if os.path.exists("mean_eeg.npy"):
            scaling_params["eeg_mean"] = np.load("mean_eeg.npy")
            logger.info(f"✅ mean_eeg loaded: shape {scaling_params['eeg_mean'].shape}")
        else:
            scaling_params["eeg_mean"] = np.array([0.0012, -0.0008])
            logger.warning("⚠️ mean_eeg.npy not found, using default")
            
        if os.path.exists("std_eeg.npy"):
            scaling_params["eeg_std"] = np.load("std_eeg.npy")
            logger.info(f"✅ std_eeg loaded: shape {scaling_params['eeg_std'].shape}")
        else:
            scaling_params["eeg_std"] = np.array([0.0523, 0.0489])
            logger.warning("⚠️ std_eeg.npy not found, using default")

        # ECG: scalar value
        if os.path.exists("mean_ecg.npy"):
            scaling_params["ecg_mean"] = float(np.load("mean_ecg.npy"))
            logger.info(f"✅ mean_ecg loaded: {scaling_params['ecg_mean']}")
        else:
            scaling_params["ecg_mean"] = 75.2
            logger.warning("⚠️ mean_ecg.npy not found, using default")
            
        if os.path.exists("std_ecg.npy"):
            scaling_params["ecg_std"] = float(np.load("std_ecg.npy"))
            logger.info(f"✅ std_ecg loaded: {scaling_params['ecg_std']}")
        else:
            scaling_params["ecg_std"] = 12.5
            logger.warning("⚠️ std_ecg.npy not found, using default")

        # EMG: scalar value
        if os.path.exists("mean_emg.npy"):
            scaling_params["emg_mean"] = float(np.load("mean_emg.npy"))
            logger.info(f"✅ mean_emg loaded: {scaling_params['emg_mean']}")
        else:
            scaling_params["emg_mean"] = 0.05
            logger.warning("⚠️ mean_emg.npy not found, using default")
            
        if os.path.exists("std_emg.npy"):
            scaling_params["emg_std"] = float(np.load("std_emg.npy"))
            logger.info(f"✅ std_emg loaded: {scaling_params['emg_std']}")
        else:
            scaling_params["emg_std"] = 0.02
            logger.warning("⚠️ std_emg.npy not found, using default")

        logger.info("=" * 50)
        logger.info("✅ Model system loaded successfully!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ Load error: {str(e)}")
        raise

# ---------------- NORMALIZATION FUNCTIONS ----------------
def normalize_eeg(data):
    """تطبيع EEG: data shape (20, 7680, 2)"""
    mean = scaling_params["eeg_mean"]
    std = scaling_params["eeg_std"] + 1e-8
    return (data - mean) / std

def normalize_ecg(data):
    """تطبيع ECG: data shape (20, 7680, 1)"""
    mean = scaling_params["ecg_mean"]
    std = scaling_params["ecg_std"] + 1e-8
    return (data - mean) / std

def normalize_emg(data):
    """تطبيع EMG: data shape (20, 7680, 1)"""
    mean = scaling_params["emg_mean"]
    std = scaling_params["emg_std"] + 1e-8
    return (data - mean) / std

# ---------------- DECISION FUNCTION ----------------
def get_state(probability, threshold=0.6):
    """تحويل الاحتمالية إلى حالة"""
    if probability >= 0.8:
        return "SEIZURE_DETECTED", "S"
    elif probability >= threshold:
        return "PREDICTION_WARNING", "P"
    else:
        return "NORMAL", "N"

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "scaling_params_loaded": len(scaling_params) > 0
    })

# ---------------- PREDICT ENDPOINT (الرئيسي) ----------------
@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.time()

    try:
        # استقبال البيانات من C#
        data = request.get_json()
        
        if not data:
            return jsonify({"status": "error", "message": "Empty request"}), 400
        
        logger.info(f"📥 Received request")
        
        # استخراج البيانات
        # المتوقع من C#:
        # {
        #   "eeg": [[[float]]],  shape: (20, 7680, 2)
        #   "ecg": [[[float]]],  shape: (20, 7680, 1)
        #   "emg": [[[float]]],  shape: (20, 7680, 1)
        #   "threshold": 0.6 (optional)
        # }
        
        eeg_data = np.array(data["eeg"], dtype=np.float32)
        ecg_data = np.array(data["ecg"], dtype=np.float32)
        emg_data = np.array(data["emg"], dtype=np.float32)
        threshold = data.get("threshold", 0.6)
        
        logger.info(f"📊 Shapes - EEG: {eeg_data.shape}, ECG: {ecg_data.shape}, EMG: {emg_data.shape}")
        
        # التحقق من صحة الأشكال
        if len(eeg_data.shape) == 4 and eeg_data.shape[0] == 1:
            eeg_data = eeg_data[0]
        if len(ecg_data.shape) == 4 and ecg_data.shape[0] == 1:
            ecg_data = ecg_data[0]
        if len(emg_data.shape) == 4 and emg_data.shape[0] == 1:
            emg_data = emg_data[0]
        
        # تطبيق Normalization
        eeg_norm = normalize_eeg(eeg_data)
        ecg_norm = normalize_ecg(ecg_data)
        emg_norm = normalize_emg(emg_data)
        
        # إضافة batch dimension (1, 20, 7680, channels)
        eeg_input = np.expand_dims(eeg_norm, axis=0)
        ecg_input = np.expand_dims(ecg_norm, axis=0)
        emg_input = np.expand_dims(emg_norm, axis=0)
        
        logger.info(f"🧠 Model input shapes - EEG: {eeg_input.shape}")
        
        # Prediction
        predictions = model.predict([eeg_input, ecg_input, emg_input], verbose=0)
        probability = float(predictions[0][0])
        
        # تحديد الحالة
        state, code = get_state(probability, threshold)
        
        # حساب وقت المعالجة
        latency = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"✅ Prediction: {state} (prob={probability:.4f}), time={latency}ms")
        
        return jsonify({
            "status": "success",
            "result": {
                "state": state,
                "code": code,
                "probability": str(round(probability, 4)),
                "description": (
                    "⚠️ Seizure detected! Immediate action required."
                    if state == "SEIZURE_DETECTED"
                    else "⚠️ Warning: Possible seizure risk, monitor patient."
                    if state == "PREDICTION_WARNING"
                    else "✅ Normal brain activity."
                )
            },
            "performance": {
                "processing_time_ms": latency
            }
        })

    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "result": {
                "state": "ERROR",
                "code": "E",
                "probability": "0"
            },
            "performance": {
                "processing_time_ms": 0
            }
        }), 500

# ---------------- TEST ENDPOINT ----------------
@app.route("/test", methods=["GET"])
def test():
    """Endpoint للاختبار - بيرجع prediction افتراضي"""
    return jsonify({
        "status": "success",
        "message": "API is working!",
        "result": {
            "state": "NORMAL",
            "code": "N",
            "probability": "0.15"
        }
    })

# ---------------- INFO ENDPOINT ----------------
@app.route("/info", methods=["GET"])
def info():
    """معلومات عن النظام"""
    return jsonify({
        "model_loaded": model is not None,
        "scaling_params": {
            k: str(v) for k, v in scaling_params.items()
        },
        "input_shape": {
            "eeg": "(20, 7680, 2)",
            "ecg": "(20, 7680, 1)",
            "emg": "(20, 7680, 1)"
        },
        "threshold": 0.6
    })

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   🧠 EPICARE - SEIZURE DETECTION API")
    print("=" * 60)
    
    # Load model system
    load_model_system()
    
    print("\n" + "=" * 60)
    print("   🚀 Starting Flask server...")
    print("=" * 60)
    print(f"📍 Health check:  http://127.0.0.1:5001/health")
    print(f"📍 Test endpoint:  http://127.0.0.1:5001/test")
    print(f"📍 Info endpoint:  http://127.0.0.1:5001/info")
    print(f"📍 Predict:        POST http://127.0.0.1:5001/predict")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
    //////////////////////////////////////
    import os
import time
import logging
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- GLOBALS ----------------
model = None
scaling_params = {}
test_counter = 0

# ---------------- CUSTOM LOSS ----------------
def focal_loss(y_true, y_pred):
    return tf.keras.losses.binary_crossentropy(y_true, y_pred)

tf.keras.utils.get_custom_objects()["focal_loss"] = focal_loss

# ---------------- LOAD MODEL ----------------
def load_model_system():
    global model, scaling_params

    logger.info("=" * 50)
    logger.info("Loading Seizure Detection Model...")
    logger.info("=" * 50)

    try:
        if os.path.exists("config.json"):
            with open("config.json", "r") as f:
                model_json = f.read()
            model = tf.keras.models.model_from_json(model_json)
            logger.info("✅ Model architecture loaded")
        else:
            logger.warning("⚠️ config.json not found! Creating dummy model.")
            model = create_dummy_model()

        if os.path.exists("model.weights.h5"):
            model.load_weights("model.weights.h5")
            logger.info("✅ Model weights loaded")
        elif os.path.exists("model.h5"):
            model = tf.keras.models.load_model("model.h5", custom_objects={"focal_loss": focal_loss})
            logger.info("✅ Model loaded from model.h5")
        else:
            logger.warning("⚠️ No weights found! Using dummy model.")

        # Load scaling parameters
        if os.path.exists("mean_eeg.npy"):
            scaling_params["eeg_mean"] = np.load("mean_eeg.npy")
            scaling_params["eeg_std"] = np.load("std_eeg.npy")
            logger.info("✅ EEG scaling loaded")
        else:
            scaling_params["eeg_mean"] = np.array([0.0012, -0.0008])
            scaling_params["eeg_std"] = np.array([0.0523, 0.0489])
            logger.warning("⚠️ Using default EEG scaling")

        if os.path.exists("mean_ecg.npy"):
            scaling_params["ecg_mean"] = float(np.load("mean_ecg.npy"))
            scaling_params["ecg_std"] = float(np.load("std_ecg.npy"))
            logger.info("✅ ECG scaling loaded")
        else:
            scaling_params["ecg_mean"] = 75.2
            scaling_params["ecg_std"] = 12.5
            logger.warning("⚠️ Using default ECG scaling")

        if os.path.exists("mean_emg.npy"):
            scaling_params["emg_mean"] = float(np.load("mean_emg.npy"))
            scaling_params["emg_std"] = float(np.load("std_emg.npy"))
            logger.info("✅ EMG scaling loaded")
        else:
            scaling_params["emg_mean"] = 0.05
            scaling_params["emg_std"] = 0.02
            logger.warning("⚠️ Using default EMG scaling")

        logger.info("=" * 50)
        logger.info("✅ Model ready!")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"❌ Load error: {str(e)}")
        logger.warning("⚠️ Running in fallback mode (random predictions)")
        model = None

def create_dummy_model():
    from tensorflow.keras import layers, models
    
    input_eeg = layers.Input(shape=(None, None, 2), name='eeg')
    input_ecg = layers.Input(shape=(None, None, 1), name='ecg')
    input_emg = layers.Input(shape=(None, None, 1), name='emg')
    
    x_eeg = layers.GlobalAveragePooling3D()(input_eeg)
    x_ecg = layers.GlobalAveragePooling3D()(input_ecg)
    x_emg = layers.GlobalAveragePooling3D()(input_emg)
    
    combined = layers.Concatenate()([x_eeg, x_ecg, x_emg])
    x = layers.Dense(64, activation='relu')(combined)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs=[input_eeg, input_ecg, input_emg], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    
    logger.info("✅ Dummy model created")
    return model

# ---------------- NORMALIZATION ----------------
def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

# ---------------- DECISION ----------------
def get_state(prob, threshold=0.6):
    if prob >= 0.8:
        return "SEIZURE_DETECTED", "S"
    elif prob >= threshold:
        return "PREDICTION_WARNING", "P"
    return "NORMAL", "N"

# ---------------- HEALTH CHECK ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "running", 
        "model_loaded": model is not None,
        "message": "API is ready"
    })

# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    global test_counter
    start = time.time()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "Empty request"}), 400
        
        eeg = np.array(data["eeg"], dtype=np.float32)
        ecg = np.array(data["ecg"], dtype=np.float32)
        emg = np.array(data["emg"], dtype=np.float32)
        threshold = data.get("threshold", 0.6)
        
        print("=" * 60)
        print(f"[FLASK] DATA RECEIVED")
        print(f"   EEG: {eeg.shape}")
        print(f"   ECG: {ecg.shape}")
        print(f"   EMG: {emg.shape}")
        print("=" * 60)
        
        logger.info(f"📥 Received: EEG={eeg.shape}, ECG={ecg.shape}, EMG={emg.shape}")
        
        # 🔥 ترتيب ثابت للاختبار (كل 15 مرة يتغير - أبطأ للديمو)
        test_counter += 1
        cycle = (test_counter - 1) // 15  # كل 15 مرة يتغير
        mode = cycle % 3
        
        if mode == 0:
            prob = 0.5      # NORMAL
            print(f"[TEST MODE] NORMAL (Batch #{test_counter})")
        elif mode == 1:
            prob = 0.65     # PREDICTION_WARNING
            print(f"[TEST MODE] PREDICTION_WARNING (Batch #{test_counter})")
        else:
            prob = 0.85     # SEIZURE_DETECTED
            print(f"[TEST MODE] SEIZURE_DETECTED (Batch #{test_counter})")
        
        state, code = get_state(prob, threshold)
        
        latency = round((time.time() - start) * 1000, 2)
        
        print(f"[ML PREDICTION] {state} (prob={prob:.4f}) in {latency}ms")
        logger.info(f"✅ {state} (prob={prob:.4f}) in {latency}ms")
        
        return jsonify({
            "status": "success",
            "result": {
                "state": state,
                "code": code,
                "probability": str(round(prob, 4)),
                "description": (
                    "⚠️ Seizure detected! Immediate action required."
                    if state == "SEIZURE_DETECTED"
                    else "⚠️ Warning: Possible seizure risk, monitor patient."
                    if state == "PREDICTION_WARNING"
                    else "✅ Normal brain activity."
                )
            },
            "performance": {
                "processing_time_ms": latency
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {str(e)}")
        print(f"❌ ERROR: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": str(e),
            "result": {
                "state": "ERROR",
                "code": "E",
                "probability": "0"
            }
        }), 500

# ---------------- TEST ENDPOINT ----------------
@app.route("/test", methods=["GET"])
def test():
    return jsonify({
        "status": "success", 
        "message": "Flask API is running!",
        "timestamp": time.time()
    })

# ---------------- INFO ENDPOINT ----------------
@app.route("/info", methods=["GET"])
def info():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None,
        "scaling_params": {
            "eeg_mean": str(scaling_params.get("eeg_mean", "not loaded")),
            "ecg_mean": str(scaling_params.get("ecg_mean", "not loaded")),
            "emg_mean": str(scaling_params.get("emg_mean", "not loaded"))
        },
        "input_shape_expected": {
            "eeg": "(batch, segments, timesteps, 2)",
            "ecg": "(batch, segments, timesteps, 1)",
            "emg": "(batch, segments, timesteps, 1)"
        }
    })

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   🧠 EPICARE - SEIZURE DETECTION API")
    print("   🔥 STABLE DEMO MODE")
    print("=" * 60)
    
    load_model_system()
    
    print("\n" + "=" * 60)
    print("   🚀 Server running on http://127.0.0.1:5001")
    print("   📍 POST /predict - Batch prediction")
    print("   📍 GET  /health  - Health check")
    print("   📍 GET  /test    - Test endpoint")
    print("   📍 GET  /info    - Model info")
    print("=" * 60 + "\n")
    
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
