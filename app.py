from flask import Flask, render_template, request, jsonify
from flask_cors import CORS 
import librosa
import numpy as np
import joblib
import os


app = Flask(__name__)
CORS(app, origins="http://127.0.0.1:5500")
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


meta_model = joblib.load('../meta_model.joblib')
scaler = joblib.load('../scaler.joblib')
svm_model = joblib.load('../svm_model.joblib')
rf_model = joblib.load('../rf_model.joblib')
xgb_model = joblib.load('../xgb_model.joblib')

emotions = ['Neutral', 'Calm', 'Happy', 'Sad', 'Angry', 'Fearful', 'Disgust', 'Surprised']


def extract_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        zcr = librosa.feature.zero_crossing_rate(y=y).mean()
        rmse = librosa.feature.rms(y=y).mean()

        features = np.hstack([mfccs.mean(axis=1), spectral_centroid, zcr, rmse])
        return features.reshape(1, -1)
    except Exception as e:
        return None


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("Receiving audio file...")
        file = request.files['file']
        if not file or file.filename == '':
            print("No file uploaded.")
            return jsonify({"error": "No file uploaded"}), 400
        
       
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        print(f"File saved at: {file_path}")

        
        features = extract_features(file_path)
        if features is None:
            raise ValueError("Failed to extract features from audio")
        print("Features extracted successfully.")

        
        features_scaled = scaler.transform(features)
        svm_probs = svm_model.predict_proba(features_scaled)
        rf_probs = rf_model.predict_proba(features_scaled)
        xgb_probs = xgb_model.predict_proba(features_scaled)
        print("Models predicted probabilities.")

        
        combined_features = np.hstack([svm_probs, rf_probs, xgb_probs])
        prediction = meta_model.predict(combined_features)
        emotion = emotions[prediction[0]]
        print(f"Predicted Emotion: {emotion}")

        
        os.remove(file_path)
        print(f"File {file.filename} removed.")
        return jsonify({"emotion": emotion})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500



@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
