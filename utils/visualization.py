import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from config import Config


def generate_visualizations(df, results=None):
    """
    Generate all visualizations for the dataset and model results
    Returns dictionary of base64-encoded images
    """
    visualizations = {}
    plt.switch_backend('Agg')  # Set non-interactive backend

    try:
        # 1. Target Variable Distribution (G3)
        if 'G3' in df.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(df['G3'], bins=20, kde=True, color='#3498db')
            plt.title('Final Grade (G3) Distribution', fontsize=16, pad=20)
            plt.xlabel('Grade Points', fontsize=12)
            plt.ylabel('Number of Students', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.3)

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            visualizations['target_dist'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 2. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            corr = df[numeric_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                        cmap='coolwarm', center=0,
                        cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix', fontsize=16, pad=20)

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            visualizations['correlation'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 3. Model Results Visualizations
        if results:
            # Actual vs Predicted Plot
            plt.figure(figsize=(10, 6))
            for name, result in results.items():
                plt.scatter(result['y_test'], result['y_pred'],
                            alpha=0.5, label=name)
            plt.plot([df['G3'].min(), df['G3'].max()],
                     [df['G3'].min(), df['G3'].max()],
                     'k--', label='Perfect Prediction')
            plt.xlabel('Actual Grades', fontsize=12)
            plt.ylabel('Predicted Grades', fontsize=12)
            plt.title('Actual vs Predicted Grades', fontsize=16, pad=20)
            plt.legend()

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            visualizations['actual_vs_predicted'] = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Feature Importance Plot (if available)
            if hasattr(list(results.values())[0]['model'], 'feature_importances_'):
                model = list(results.values())[0]['model']
                features = df.columns.drop('G3')
                importances = model.feature_importances_

                plt.figure(figsize=(12, 6))
                sns.barplot(x=features, y=importances, palette='viridis')
                plt.xticks(rotation=45, ha='right')
                plt.xlabel('Features', fontsize=12)
                plt.ylabel('Importance Score', fontsize=12)
                plt.title('Feature Importance', fontsize=16, pad=20)

                buf = BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                plt.close()
                visualizations['feature_importance'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 4. Categorical Features Distribution
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:3]:  # Limit to first 3 categorical features
            plt.figure(figsize=(10, 6))
            sns.countplot(x=col, data=df, palette='pastel')
            plt.title(f'{col} Distribution', fontsize=16, pad=20)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Count', fontsize=12)

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            visualizations[f'cat_{col}'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # 5. Numeric Features vs Target
        numeric_cols = numeric_cols.drop('G3', errors='ignore')
        for col in numeric_cols[:3]:  # Limit to first 3 numeric features
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=col, y='G3', data=df, alpha=0.6)
            plt.title(f'{col} vs Final Grade', fontsize=16, pad=20)
            plt.xlabel(col, fontsize=12)
            plt.ylabel('Final Grade (G3)', fontsize=12)

            buf = BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            plt.close()
            visualizations[f'num_{col}'] = base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Visualization generation error: {str(e)}")

    return visualizations