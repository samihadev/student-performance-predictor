import shap
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
import logging
from sklearn.inspection import permutation_importance
from config import Config
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_explainer(model, X_background=None):
    """Return appropriate SHAP explainer for the model type"""
    try:
        # Tree-based models
        if hasattr(model, 'tree_') or hasattr(model, 'estimators_'):
            logger.info("Using TreeExplainer")
            return shap.TreeExplainer(model)

        # Linear models
        elif hasattr(model, 'coef_'):
            logger.info("Using LinearExplainer")
            return shap.LinearExplainer(model, X_background)

        # Fallback for other models
        else:
            if X_background is None:
                raise ValueError("X_background required for KernelExplainer")
            logger.info("Using KernelExplainer")
            return shap.KernelExplainer(model.predict, X_background)

    except Exception as e:
        logger.error(f"Could not create explainer: {str(e)}")
        return None


def generate_shap_summary(model, X_train, model_name=None):
    """Generate SHAP summary plot for model interpretation"""
    try:
        explainer = get_explainer(model, X_train)
        if explainer is None:
            logger.error("Failed to create explainer")
            return None

        # Calculate SHAP values
        shap_values = explainer.shap_values(X_train)

        # Create plot
        plt.switch_backend('Agg')
        plt.figure(figsize=(10, 6))

        if isinstance(model, AdaBoostRegressor):
            shap.summary_plot(shap_values, X_train, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_train, show=False)

        if model_name:
            plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()

        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Failed to generate SHAP summary: {str(e)}")
        return None


def explain_prediction(model, X_background, sample, feature_names):
    """Generate individual prediction explanation"""
    try:
        explainer = get_explainer(model, X_background)
        if explainer is None:
            return None

        # Handle AdaBoost specially
        if isinstance(model, AdaBoostRegressor):
            shap_values = np.mean([e.shap_values(sample) for e in explainer.estimators_], axis=0)
        else:
            shap_values = explainer.shap_values(sample)

        # Create force plot
        plt.switch_backend('Agg')
        fig = plt.figure(figsize=(10, 3))
        shap.force_plot(
            explainer.expected_value,
            shap_values,
            sample,
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Prediction explanation failed: {str(e)}")
        return None


def get_top_features(model, X_sample, n_features=3):
    """Identify top positive and negative influence features"""
    try:
        explainer = get_explainer(model, X_sample)
        if explainer is None:
            return "Various features", "Various features"

        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)

        if len(shap_values.shape) > 2:  # Handle multi-output
            shap_values = shap_values[0]

        # Get mean absolute SHAP values
        mean_shap = pd.DataFrame(shap_values, columns=X_sample.columns).abs().mean()

        top_positive = mean_shap.nlargest(n_features).index.tolist()
        top_negative = mean_shap.nsmallest(n_features).index.tolist()

        return ", ".join(top_positive), ", ".join(top_negative)

    except Exception as e:
        print(f"Error getting top features: {str(e)}")
        return "Various features", "Various features"


def generate_actual_vs_predicted_plot(model, X_test, y_test):
    """Generate actual vs predicted values plot"""
    try:
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(8, 6))

        y_pred = model.predict(X_test)
        ax.scatter(y_test, y_pred, alpha=0.5)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'k--', lw=2)

        ax.set_xlabel('Actual Grades')
        ax.set_ylabel('Predicted Grades')
        ax.set_title('Actual vs Predicted Values')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Error generating actual vs predicted plot: {str(e)}")
        return None


def generate_feature_importance_plot(model):
    """Generate feature importance plot"""
    try:
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(8, 6))

        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            features = range(len(importances))
            ax.bar(features, importances)
            ax.set_title('Feature Importance Scores')
        else:
            ax.text(0.5, 0.5, 'Feature importance not available',
                    ha='center', va='center')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        logger.error(f"Error generating feature importance plot: {str(e)}")
        return None
    import shap
    import numpy as np

    def get_explainer(model, background_data):
        """More robust explainer creation"""
        try:
            if str(type(model)).endswith("sklearn.ensemble._forest.RandomForestRegressor'>"):
                return shap.TreeExplainer(model)
            return shap.Explainer(model, background_data)
        except:
            return None


def generate_enhanced_explanations(model, X_array, X_df, background_data, feature_names, prediction_value):
    """Generate comprehensive AI explanations for the prediction"""
    explanation_data = {
        'shap_force': None,
        'base_value': None,
        'top_positive_feature': "Various features",
        'top_negative_feature': "Various features",
        'confidence_score': 0.8,
        'feature_impacts': [],
        'prediction_range': "±1.0"
    }

    try:
        # Calculate base value (expected average)
        print("Calculating base value...")
        base_predictions = model.predict(background_data)
        base_value = float(np.mean(base_predictions))
        explanation_data['base_value'] = round(base_value, 2)
        print(f"Base value: {base_value}")

        # Generate SHAP explanations with fallback
        try:
            print("Generating SHAP explanations...")
            explainer = get_robust_explainer(model, background_data)

            if explainer is not None:
                shap_values = calculate_shap_values(explainer, X_array)

                if shap_values is not None:
                    # Process SHAP values
                    if len(shap_values.shape) > 1 and shap_values.shape[0] > 0:
                        shap_vals_1d = shap_values[0] if len(shap_values.shape) > 1 else shap_values

                        # Calculate feature impacts
                        feature_impacts = []
                        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_vals_1d)):
                            if abs(shap_val) > 0.01:  # Only significant impacts
                                feature_impacts.append({
                                    'feature': get_readable_feature_name(feature),
                                    'impact': round(float(shap_val), 3),
                                    'direction': 'positive' if shap_val > 0 else 'negative'
                                })

                        # Sort by absolute impact
                        feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
                        explanation_data['feature_impacts'] = feature_impacts[:8]  # Top 8 features

                        # Get top positive and negative features
                        positive_features = [f['feature'] for f in feature_impacts if f['direction'] == 'positive'][:3]
                        negative_features = [f['feature'] for f in feature_impacts if f['direction'] == 'negative'][:3]

                        explanation_data['top_positive_feature'] = ', '.join(
                            positive_features) if positive_features else "None significant"
                        explanation_data['top_negative_feature'] = ', '.join(
                            negative_features) if negative_features else "None significant"

                        # Generate SHAP force plot
                        explanation_data['shap_force'] = create_enhanced_shap_plot(
                            shap_vals_1d, feature_names, base_value, prediction_value
                        )

                        print("SHAP explanations generated successfully")

        except Exception as shap_error:
            print(f"SHAP explanation failed: {str(shap_error)}")
            # Fallback to feature importance
            explanation_data.update(get_fallback_explanations(model, feature_names))

        # Calculate prediction confidence
        confidence = calculate_prediction_confidence(prediction_value, base_value)
        explanation_data['confidence_score'] = confidence

        # Calculate prediction range
        prediction_std = np.std(base_predictions)
        range_value = min(2.0, prediction_std * 1.5)
        explanation_data['prediction_range'] = f"±{range_value:.1f}"

        print(f"Explanation generation completed")

    except Exception as e:
        print(f"Explanation generation failed: {str(e)}")
        # Return minimal explanation data
        explanation_data.update({
            'base_value': 11.5,
            'confidence_score': 0.7,
            'prediction_range': "±1.5"
        })

    return explanation_data


def get_robust_explainer(model, background_data):
    """Create SHAP explainer with multiple fallback options"""
    try:
        # Try TreeExplainer for tree-based models
        if hasattr(model, 'estimators_') or hasattr(model, 'tree_'):
            print("Using TreeExplainer")
            return shap.TreeExplainer(model)

        # Try LinearExplainer for linear models
        elif hasattr(model, 'coef_'):
            print("Using LinearExplainer")
            return shap.LinearExplainer(model, background_data)

        # Fallback to general Explainer
        else:
            print("Using general Explainer")
            return shap.Explainer(model, background_data)

    except Exception as e:
        print(f"Failed to create explainer: {str(e)}")
        return None


def calculate_shap_values(explainer, X_array):
    """Calculate SHAP values with error handling"""
    try:
        shap_values = explainer(X_array)

        # Handle different SHAP value formats
        if hasattr(shap_values, 'values'):
            return shap_values.values
        else:
            return shap_values

    except Exception as e:
        print(f"SHAP values calculation failed: {str(e)}")
        return None


def create_enhanced_shap_plot(shap_values, feature_names, base_value, prediction_value):
    """Create an enhanced SHAP visualization"""
    try:
        plt.switch_backend('Agg')
        fig, ax = plt.subplots(figsize=(12, 6))

        # Prepare data for plotting
        impacts = []
        labels = []
        colors = []

        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values)):
            if abs(shap_val) > 0.01:  # Only significant features
                impacts.append(shap_val)
                labels.append(get_readable_feature_name(feature))
                colors.append('#2E8B57' if shap_val > 0 else '#CD5C5C')

        # Sort by absolute impact
        sorted_data = sorted(zip(impacts, labels, colors), key=lambda x: abs(x[0]), reverse=True)
        impacts, labels, colors = zip(*sorted_data) if sorted_data else ([], [], [])

        # Create horizontal bar plot
        if impacts:
            y_pos = np.arange(len(labels))
            bars = ax.barh(y_pos, impacts, color=colors, alpha=0.8)

            # Add value labels on bars
            for i, (bar, impact) in enumerate(zip(bars, impacts)):
                width = bar.get_width()
                ax.text(width + (0.01 if width > 0 else -0.01), bar.get_y() + bar.get_height() / 2,
                        f'{impact:+.2f}', ha='left' if width > 0 else 'right', va='center', fontsize=10)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Feature Impact on Prediction')
            ax.set_title(f'AI Model Explanation\nBase Value: {base_value:.2f} → Prediction: {prediction_value:.2f}')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
        else:
            ax.text(0.5, 0.5, 'No significant feature impacts detected',
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)

        plt.tight_layout()

        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        plt.close()

        return base64.b64encode(buffer.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Enhanced SHAP plot creation failed: {str(e)}")
        return None


def get_fallback_explanations(model, feature_names):
    """Provide fallback explanations when SHAP fails"""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp_df = pd.DataFrame({
                'feature': feature_names[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)

            top_features = feature_imp_df.head(3)['feature'].tolist()
            return {
                'top_positive_feature': ', '.join([get_readable_feature_name(f) for f in top_features]),
                'top_negative_feature': "Based on feature importance ranking"
            }
    except Exception:
        pass

    return {
        'top_positive_feature': "Previous grades, study time",
        'top_negative_feature': "Failures, absences"
    }


def get_readable_feature_name(feature):
    """Convert technical feature names to readable format"""
    name_mapping = {
        'G1': 'First Period Grade',
        'G2': 'Second Period Grade',
        'studytime': 'Weekly Study Time',
        'failures': 'Past Failures',
        'absences': 'Number of Absences',
        'Medu': "Mother's Education",
        'Fedu': "Father's Education",
        'famrel': 'Family Relationship Quality',
        'health': 'Health Status',
        'goout': 'Social Activity Level',
        'age': 'Student Age',
        'sex': 'Gender',
        'address': 'Address Type',
        'schoolsup': 'Extra School Support'
    }

    return name_mapping.get(feature, feature.replace('_', ' ').title())


def calculate_prediction_confidence(prediction, base_value, min_grade=0, max_grade=20):
    """Calculate confidence score based on prediction characteristics"""
    try:
        # Confidence decreases as prediction moves away from reasonable range
        if base_value - 3 <= prediction <= base_value + 3:
            confidence = 0.9  # High confidence for predictions near average
        elif base_value - 5 <= prediction <= base_value + 5:
            confidence = 0.8  # Medium-high confidence
        elif 0 <= prediction <= 20:
            confidence = 0.7  # Medium confidence for valid range
        else:
            confidence = 0.6  # Lower confidence for extreme values

        return round(confidence, 2)

    except Exception:
        return 0.75  #