import os
from pathlib import Path


class Config:
    # Base directory
    BASE_DIR = Path(__file__).parent.parent

    # Directory paths
    MODEL_DIR = BASE_DIR / 'models'
    STATIC_DIR = BASE_DIR / 'static'
    UPLOAD_FOLDER = BASE_DIR / 'uploads'

    # Create directories if they don't exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # Model file paths
    MODEL_FILE = str(MODEL_DIR / 'best_model.joblib')
    SCALER_FILE = str(MODEL_DIR / 'scaler.joblib')
    ENCODER_FILE = str(MODEL_DIR / 'encoder.joblib')
    FEATURE_NAMES_FILE = str(MODEL_DIR / 'feature_names.pkl')
    PREPROCESSOR_FILE = str(MODEL_DIR / 'preprocessor.joblib')
    BACKGROUND_DATA_FILE = str(MODEL_DIR / 'background_data.joblib')  # Added this line

    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-123'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB upload limit

    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx'}