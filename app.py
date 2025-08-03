import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from config import Config
from utils.preprocessing import preprocess_data, validate_data, preprocess_single_input,validate_and_clean_form_data
from utils.modeling import train_and_evaluate_models
from utils.visualization import generate_visualizations
import joblib
from sklearn.ensemble import AdaBoostRegressor
from utils.explainability import (
    get_explainer,
    generate_shap_summary,
    explain_prediction,
    get_top_features,
    generate_actual_vs_predicted_plot,
    generate_feature_importance_plot,
    generate_enhanced_explanations,
    create_enhanced_shap_plot
)
import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
from utils.preprocessing import save_preprocessor, load_preprocessor
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Global variables to store data between requests
global_data = {
    'df': None,
    'results': None,
    'visualizations': None,
    'target_column': 'G3'  # Default target column
}


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def read_uploaded_file(file):
    if file.filename.endswith('.csv'):
        first_line = file.read(1024).decode('utf-8').split('\n')[0]
        file.seek(0)
        delimiter = ';' if ';' in first_line else ','
        try:
            return pd.read_csv(file, delimiter=delimiter, encoding='utf-8')
        except UnicodeDecodeError:
            file.seek(0)
            return pd.read_csv(file, delimiter=delimiter, encoding='latin1')
    return pd.read_excel(file)


@app.route('/')
def home():
    return redirect(url_for('index'))


@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                df = read_uploaded_file(file)
                validation = validate_data(df)
                if not validation['valid']:
                    flash(validation['message'])
                    return redirect(request.url)

                global_data['df'] = df
                if 'target_column' in validation:
                    global_data['target_column'] = validation['target_column']

                return redirect(url_for('analyze'))
            except Exception as e:
                flash(f'Error reading file: {str(e)}')
                return redirect(request.url)
    return render_template('index.html')


@app.route('/analyze')
def analyze():
    if global_data['df'] is None:
        return redirect(url_for('index'))

    visualizations = generate_visualizations(global_data['df'])
    global_data['visualizations'] = visualizations

    stats = {
        'rows': global_data['df'].shape[0],
        'columns': global_data['df'].shape[1],
        'columns_list': global_data['df'].columns.tolist(),
        'numeric_stats': global_data['df'].describe().to_html(classes='table table-striped')
    }
    return render_template('analyze.html', stats=stats, visualizations=visualizations)


@app.route('/train', methods=['GET', 'POST'])
def train():
    if global_data['df'] is None:
        return redirect(url_for('index'))

    if request.method == 'POST':
        selected_features = request.form.getlist('features')
        try:
            # Preprocess data
            X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
                global_data['df'], selected_features, global_data['target_column'])

            # Save preprocessing artifacts
            save_preprocessor(preprocessor, X_train.columns.tolist())
            joblib.dump(X_train.sample(100), Config.BACKGROUND_DATA_FILE)

            # Train models
            results = train_and_evaluate_models(X_train, X_test, y_train, y_test)
            global_data['results'] = results

            # Get best model
            best_model_name, best_model = min(results.items(), key=lambda x: x[1]['rmse'])

            # Generate visualizations
            visualizations = {
                'actual_vs_predicted': generate_actual_vs_predicted_plot(best_model['model'], X_test, y_test),
                'feature_importance': generate_feature_importance_plot(best_model['model']),
                'shap_summary': generate_shap_summary(best_model['model'], X_train.sample(100), best_model_name),
                'top_positive_feature': get_top_features(best_model['model'], X_train.sample(100))[0],
                'top_negative_feature': get_top_features(best_model['model'], X_train.sample(100))[1]
            }

            return render_template('model.html',
                                   results=results,
                                   visualizations=visualizations,
                                   best_model=(best_model_name, best_model))

        except Exception as e:
            flash(f'Training error: {str(e)}')
            return redirect(request.url)

    features = [col for col in global_data['df'].columns if col != global_data['target_column']]
    return render_template('train.html', features=features)
@app.route('/model-results')
def model_results():
    if global_data['results'] is None:
        return redirect(url_for('index'))

    # Get visualizations if they exist
    visualizations = global_data.get('visualizations', {})

    return render_template('model.html',
                           results=global_data['results'],
                           visualizations=visualizations)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Enhanced prediction route with better error handling and XAI"""

    # Check if required files exist
    required_files = [Config.MODEL_FILE, Config.PREPROCESSOR_FILE,
                      Config.FEATURE_NAMES_FILE, Config.BACKGROUND_DATA_FILE]

    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        flash(f'Missing required files: {missing_files}. Please train a model first.', 'danger')
        return redirect(url_for('train'))

    if request.method == 'POST':
        try:
            # 1. Load model and preprocessor with validation
            print("Loading model and preprocessor...")
            model = joblib.load(Config.MODEL_FILE)
            preprocessor, feature_names = load_preprocessor()
            background_data = joblib.load(Config.BACKGROUND_DATA_FILE)

            if model is None:
                raise ValueError("Failed to load model")
            if preprocessor is None or feature_names is None:
                raise ValueError("Failed to load preprocessor or feature names")

            print(f"Model loaded: {type(model)}")
            print(f"Feature names: {feature_names}")
            print(f"Background data shape: {background_data.shape}")

            # 2. Get and validate form data
            form_data = request.form.to_dict()
            print(f"Raw form data: {form_data}")

            # Validate required fields
            required_fields = ['age', 'studytime', 'failures', 'G1', 'G2']
            missing_fields = [field for field in required_fields
                              if field not in form_data or not str(form_data[field]).strip()]

            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            # 3. Clean and validate form data
            cleaned_data = validate_and_clean_form_data(form_data)
            print(f"Cleaned form data: {cleaned_data}")

            # 4. Preprocess input data
            print("Preprocessing input data...")
            X_df, X_array = preprocess_single_input(cleaned_data, preprocessor, feature_names)

            if X_array is None or np.isnan(X_array).any():
                raise ValueError("Failed to preprocess input data properly")

            print(f"Processed data shape: {X_array.shape}")
            print(f"First 5 values: {X_array[0][:5] if len(X_array[0]) >= 5 else X_array[0]}")

            # 5. Make prediction with validation
            print("Making prediction...")
            prediction = model.predict(X_array)

            if prediction is None or len(prediction) == 0:
                raise ValueError("Model returned invalid prediction")

            prediction_value = float(prediction[0])

            if np.isnan(prediction_value) or np.isinf(prediction_value):
                raise ValueError("Model returned NaN or infinite prediction")

            # Clamp prediction to valid range
            prediction_value = max(0, min(20, prediction_value))
            print(f"Final prediction: {prediction_value}")

            # 6. Generate comprehensive explanations
            explanation_data = generate_enhanced_explanations(
                model, X_array, X_df, background_data, feature_names, prediction_value
            )

            # 7. Return results with all explanation components
            return render_template('predict.html',
                                   prediction=round(prediction_value, 2),
                                   form_data=cleaned_data,
                                   shap_force=explanation_data.get('shap_force'),
                                   base_value=explanation_data.get('base_value'),
                                   top_positive_feature=explanation_data.get('top_positive_feature'),
                                   top_negative_feature=explanation_data.get('top_negative_feature'),
                                   confidence_score=explanation_data.get('confidence_score'),
                                   feature_impacts=explanation_data.get('feature_impacts'),
                                   prediction_range=explanation_data.get('prediction_range'))

        except ValueError as ve:
            print(f"Validation error: {str(ve)}")
            flash(f'Input error: {str(ve)}', 'danger')
            return render_template('predict.html', prediction=None, error=str(ve))

        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            flash(f'Prediction failed: {str(e)}', 'danger')
            return render_template('predict.html', prediction=None, error=str(e))

    # GET request - show empty form
    return render_template('predict.html', prediction=None)


@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.before_request
def validate_inputs():
    if request.method == 'POST' and request.endpoint == 'predict':
        required_fields = ['age', 'studytime', 'failures', 'G1', 'G2']
        for field in required_fields:
            if field not in request.form or not request.form[field].strip():
                flash(f'Missing required field: {field}', 'danger')
                return redirect(url_for('predict'))
if __name__ == '__main__':
    app.run(debug=True)