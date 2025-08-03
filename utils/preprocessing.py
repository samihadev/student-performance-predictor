import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from config import Config
import os

def save_preprocessor(preprocessor, feature_names):
    """Save preprocessor and feature names to disk"""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, Config.PREPROCESSOR_FILE)
    joblib.dump(feature_names, Config.FEATURE_NAMES_FILE)

def load_preprocessor():
    """Load preprocessor and feature names from disk"""
    try:
        preprocessor = joblib.load(Config.PREPROCESSOR_FILE)
        feature_names = joblib.load(Config.FEATURE_NAMES_FILE)
        return preprocessor, feature_names
    except Exception as e:
        print(f"Error loading preprocessor: {str(e)}")
        return None, None
def validate_data(df):
    """Validate the input DataFrame"""
    required_columns = ['G1', 'G2', 'G3', 'studytime', 'failures']
    validation = {'valid': True, 'message': ''}

    if not all(col in df.columns for col in required_columns):
        missing = [col for col in required_columns if col not in df.columns]
        validation['valid'] = False
        validation['message'] = f'Missing required columns: {missing}'
        return validation

    if df.isnull().sum().sum() > 0:
        validation['valid'] = False
        validation['message'] = 'Data contains missing values'
        return validation

    validation['target_column'] = 'G3'
    return validation


def preprocess_data(df, features, target_column):
    """
    Preprocess data and return DataFrames with feature names
    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    X = df[features].copy()
    y = df[target_column].copy()

    # Split data first to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Identify feature types
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = X_train.select_dtypes(include=['number']).columns.tolist()

    # Create transformers
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)  # Changed parameter

    # Create preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    # Fit and transform the data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Get feature names after transformation
    numeric_features = numeric_cols
    categorical_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
    feature_names = numeric_features + list(categorical_features)

    # Convert back to DataFrames with column names
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)

    return X_train, X_test, y_train, y_test, preprocessor


def preprocess_single_input(form_data, preprocessor, feature_names):
    """Enhanced preprocessing with comprehensive error handling"""
    try:
        print(f"Input form data: {form_data}")
        print(f"Expected features: {preprocessor.feature_names_in_}")

        # Create input dictionary with all expected features
        input_dict = {}

        for feature in preprocessor.feature_names_in_:
            if feature in form_data:
                input_dict[feature] = form_data[feature]
            else:
                # Set reasonable defaults for missing features
                if feature in ['age', 'studytime', 'failures', 'G1', 'G2', 'absences',
                               'Medu', 'Fedu', 'famrel', 'health', 'goout']:
                    input_dict[feature] = 0.0
                else:
                    input_dict[feature] = 'unknown'
                print(f"Warning: Missing feature {feature}, using default")

        # Create DataFrame
        input_df = pd.DataFrame([input_dict])
        print(f"Input DataFrame shape: {input_df.shape}")

        # Ensure column order matches training data
        input_df = input_df[preprocessor.feature_names_in_]

        # Transform the data
        transformed_data = preprocessor.transform(input_df)

        # Validate transformed data
        if transformed_data is None:
            raise ValueError("Preprocessor returned None")

        if np.isnan(transformed_data).any():
            print("Warning: NaN values detected, replacing with zeros")
            transformed_data = np.nan_to_num(transformed_data, nan=0.0)

        print(f"Transformed data shape: {transformed_data.shape}")
        return input_df, transformed_data

    except Exception as e:
        print(f"Enhanced preprocessing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def save_preprocessor(preprocessor, feature_names):
    """Save preprocessor and feature names to disk"""
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    joblib.dump(preprocessor, Config.PREPROCESSOR_FILE)
    joblib.dump(feature_names, Config.FEATURE_NAMES_FILE)


def load_preprocessor():
    """Load preprocessor and feature names from disk"""
    preprocessor = joblib.load(Config.PREPROCESSOR_FILE)
    feature_names = joblib.load(Config.FEATURE_NAMES_FILE)
    return preprocessor, feature_names
def validate_input(form_data, preprocessor):
    """Validate form input matches expected features"""
    errors = []
    for feature in preprocessor.feature_names_in_:
        if feature not in form_data:
            errors.append(f"Missing field: {feature}")
    return errors
def inspect_preprocessor(preprocessor):
    """Debug helper to show preprocessor configuration"""
    print("\n=== Preprocessor Inspection ===")
    print("Numeric features:", preprocessor.named_transformers_['num'].feature_names_in_)
    print("Categorical features:", preprocessor.named_transformers_['cat'].feature_names_in_)
    print("Numeric means:", preprocessor.named_transformers_['num'].mean_)
    print("Categorical categories:", [preprocessor.named_transformers_['cat'].categories_])


def validate_and_clean_form_data(form_data):
    """Validate and clean form input data for prediction"""
    cleaned_data = {}

    # Define expected fields and their validation rules
    field_rules = {
        'age': {'type': int, 'min': 15, 'max': 22, 'default': 18},
        'studytime': {'type': int, 'min': 1, 'max': 4, 'default': 2},
        'failures': {'type': int, 'min': 0, 'max': 4, 'default': 0},
        'G1': {'type': int, 'min': 0, 'max': 20, 'default': 10},
        'G2': {'type': int, 'min': 0, 'max': 20, 'default': 10},
        'absences': {'type': int, 'min': 0, 'max': 93, 'default': 0},
        'Medu': {'type': int, 'min': 0, 'max': 4, 'default': 2},
        'Fedu': {'type': int, 'min': 0, 'max': 4, 'default': 2},
        'famrel': {'type': int, 'min': 1, 'max': 5, 'default': 3},
        'health': {'type': int, 'min': 1, 'max': 5, 'default': 3},
        'goout': {'type': int, 'min': 1, 'max': 5, 'default': 3},
        'sex': {'type': str, 'options': ['M', 'F'], 'default': 'M'},
        'address': {'type': str, 'options': ['U', 'R'], 'default': 'U'},
        'schoolsup': {'type': str, 'options': ['yes', 'no'], 'default': 'no'}
    }

    for field, rules in field_rules.items():
        value = form_data.get(field, '')

        try:
            # Convert to correct type
            if rules['type'] == int:
                value = int(value) if str(value).strip() else rules['default']
            elif rules['type'] == str:
                value = str(value).strip().lower() if str(value).strip() else rules['default']
                if 'options' in rules and value not in rules['options']:
                    value = rules['default']

            # Validate ranges
            if rules['type'] == int:
                if 'min' in rules and value < rules['min']:
                    value = rules['min']
                if 'max' in rules and value > rules['max']:
                    value = rules['max']

            cleaned_data[field] = value

        except (ValueError, TypeError):
            cleaned_data[field] = rules['default']

    return cleaned_data