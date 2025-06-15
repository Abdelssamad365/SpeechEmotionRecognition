import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_file
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
from sklearn.preprocessing import LabelEncoder
import random

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create directories if they don't exist
RECORDINGS_DIR = 'recordings'
UPLOADS_DIR = 'uploads'
for directory in [RECORDINGS_DIR, UPLOADS_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize LabelEncoder with the same emotions as training
emotions = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
le = LabelEncoder()
le.fit(emotions)

# Emotion descriptions
emotion_descriptions = {
    'neutral': 'Neutral speech',
    'happy': 'Joyful and cheerful voice',
    'sad': 'Melancholic or sorrowful tone',
    'angry': 'Frustrated or irritated voice',
    'fearful': 'Anxious or scared tone',
    'disgust': 'Displeased or repulsed voice',
    'surprised': 'Amazed or astonished tone'
}

# Load the model
try:
    model = tf.keras.models.load_model('emotion_model.h5', compile=False)
    # Recompile the model with the same optimizer and loss as training
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0008)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def validate_audio(audio_data, sample_rate):
    """Validate audio quality and characteristics."""
    # Check if audio is too quiet
    if np.max(np.abs(audio_data)) < 0.01:
        return False, "Audio is too quiet. Please speak louder."
    
    # Check if audio is too short
    if len(audio_data) < sample_rate * 2:  # Less than 2 seconds
        return False, "Audio is too short. Please speak longer."
    
    return True, "Audio validation passed"

def add_noise(y, noise_factor_range=(0.002, 0.008)):
    """Add random noise to audio signal."""
    noise_factor = random.uniform(noise_factor_range[0], noise_factor_range[1])
    noise = np.random.randn(len(y))
    return (y + noise_factor * noise).astype(type(y[0]))

def pitch_shift(y, sr, n_steps_range=(-3, 3)):
    """Apply pitch shifting to audio signal."""
    n_steps = random.randint(n_steps_range[0], n_steps_range[1])
    if n_steps == 0: return y
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)

def time_stretch(y, rate_range=(0.8, 1.2)):
    """Apply time stretching to audio signal."""
    rate = random.uniform(rate_range[0], rate_range[1])
    if rate == 1.0: return y
    return librosa.effects.time_stretch(y=y, rate=rate)

def extract_delta_features(y, sr, n_mels=128, n_mfcc=40, max_len=250):
    """Extract delta features from audio signal."""
    # Extract mel spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    log_mel = librosa.power_to_db(mel)
    
    # Extract MFCCs and deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack features
    features = np.vstack((log_mel, mfcc, mfcc_delta, mfcc_delta2))

    # Pad or truncate to max_len
    if features.shape[1] < max_len:
        features = np.pad(features, ((0, 0), (0, max_len - features.shape[1])), mode='constant')
    else:
        features = features[:, :max_len]

    # Standardize features
    if features.size > 0:
        features = (features - np.mean(features)) / (np.std(features) + 1e-6)

    return features

def spec_augment(spec, time_masking_para=40, freq_masking_para=30, num_time_masks=2, num_freq_masks=2):
    """Apply SpecAugment to the spectrogram."""
    spec_copy = spec.copy()
    n_mels, n_steps = spec_copy.shape
    
    # Frequency masking
    for _ in range(num_freq_masks):
        f = random.randrange(0, freq_masking_para)
        f0 = random.randrange(0, n_mels - f)
        if f > 0:  # Ensure f0+f does not exceed n_mels if f is small
            spec_copy[f0:f0+f, :] = 0
    
    # Time masking
    for _ in range(num_time_masks):
        t = random.randrange(0, time_masking_para)
        t0 = random.randrange(0, n_steps - t)
        if t > 0:  # Ensure t0+t does not exceed n_steps if t is small
            spec_copy[:, t0:t0+t] = 0
    
    return spec_copy

def process_audio(y, sr):
    """Process audio using the same feature extraction as training, but without augmentation."""
    try:
        if len(y) == 0:
            raise ValueError("Empty audio file")
            
        # Extract features without augmentation
        features = extract_delta_features(y, sr)
        
        # Convert to numpy array and add channel dimension
        features = np.array([features])[..., np.newaxis]
        
        return features
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def extract_features(audio_path):
    """Extract audio features for emotion recognition using the same process as test data in training."""
    try:
        # Load audio file with same parameters as training
        y, sr = librosa.load(audio_path, sr=16000)  # SR = 16000 as in training
        
        # Validate audio
        is_valid, message = validate_audio(y, sr)
        if not is_valid:
            raise ValueError(message)
        
        # Extract features exactly like test data in training
        features = extract_delta_features(y, sr)
        
        # Convert to numpy array and add channel dimension like in training
        features = np.array([features])[..., np.newaxis]
        
        return features
    except Exception as e:
        logger.error(f"Error extracting features: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/record', methods=['POST'])
def record_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400

        # Save the uploaded audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        filepath = os.path.join(RECORDINGS_DIR, filename)
        audio_file.save(filepath)
        
        # Process audio using the same feature extraction as test data in training
        features = extract_features(filepath)
        
        # Get prediction
        prediction = model.predict(features)
        
        # Get emotion prediction
        emotion_idx = np.argmax(prediction[0])
        emotion_name = le.inverse_transform([emotion_idx])[0]
        confidence = float(prediction[0][emotion_idx])
        
        # Log prediction details
        logger.info(f"Raw prediction: {prediction[0]}")
        logger.info(f"Predicted emotion index: {emotion_idx}")
        logger.info(f"Predicted emotion: {emotion_name}")
        logger.info(f"Confidence: {confidence}")
        
        return jsonify({
            'success': True,
            'emotion': emotion_name,
            'description': emotion_descriptions[emotion_name],
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'audio_file': filename
        })
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        return jsonify({
            'success': False,
            'error': "An error occurred while processing your audio. Please try again."
        }), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload a WAV file.'
            }), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_filename = f"upload_{timestamp}_{filename}"
        filepath = os.path.join(UPLOADS_DIR, saved_filename)
        file.save(filepath)

        # Process audio and get prediction
        features = extract_features(filepath)
        prediction = model.predict(features)
        emotion_idx = np.argmax(prediction[0])
        emotion_name = le.inverse_transform([emotion_idx])[0]
        confidence = float(prediction[0][emotion_idx])

        return jsonify({
            'success': True,
            'emotion': emotion_name,
            'description': emotion_descriptions[emotion_name],
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'audio_file': saved_filename
        })
    except ValueError as ve:
        logger.warning(f"Validation error: {str(ve)}")
        return jsonify({
            'success': False,
            'error': str(ve)
        }), 400
    except Exception as e:
        logger.error(f"Error processing uploaded file: {str(e)}")
        return jsonify({
            'success': False,
            'error': "An error occurred while processing your audio file. Please try again."
        }), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    """Serve the recorded audio file."""
    try:
        # Check both recordings and uploads directories
        for directory in [RECORDINGS_DIR, UPLOADS_DIR]:
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                return send_file(
                    filepath,
                    mimetype='audio/wav'
                )
        raise FileNotFoundError("Audio file not found")
    except Exception as e:
        logger.error(f"Error serving audio file: {str(e)}")
        return jsonify({
            'success': False,
            'error': "Audio file not found"
        }), 404

if __name__ == '__main__':
    app.run(debug=True) 