# 🎓 Student Performance Predictor with Explainable AI (SHAP)

This web application predicts students' final grades based on personal, academic, and social information. It uses powerful ensemble machine learning models and integrates **Explainable AI (SHAP)** to help users understand what influences predictions.

---

## 🚀 Features

- 📁 Upload student data as a CSV file
- 📊 Visualize data (correlation matrix, feature importance, prediction accuracy)
- 🤖 Train multiple ensemble models:
  - Random Forest
  - Gradient Boosting
  - AdaBoost
- 🧠 View model evaluation metrics (R², MSE, RMSE)
- 💡 SHAP explainability: see how each feature affects prediction (positively or negatively)
- 🔮 Predict individual student final grade and explain the decision

---

## 🧪 Sample Inputs

The model uses features such as:

- Personal: `sex`, `age`, `address`, `health`, `absences`
- Family: `Medu`, `Fedu`, `famsize`, `Pstatus`, `famrel`
- School-related: `studytime`, `failures`, `schoolsup`, `paid`, `activities`
- Academic: `G1`, `G2` (first and second period grades)
- Target variable: `G3` (final grade)

> **Note:** The dataset must contain a `G3` column to train models.

---

## 📦 Installation

```bash
git clone https://github.com/samihadev/student-performance-predictor.git
cd student-performance-predictor
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
⚙️ Running the App
bash
Copier
Modifier
python app.py
Then open your browser at: http://localhost:5000

📈 Explainable AI with SHAP
🔴 Red bars: features that increase the predicted grade

🔵 Blue bars: features that decrease the predicted grade

📏 Bar length: impact magnitude

📶 Higher features: more important

SHAP helps users understand why the model made a specific prediction — improving transparency and trust.

📸 Screenshots (Optional)
You can include:

SHAP summary plot

Feature importance graph

Prediction vs Actual scatter plot

Individual explanation sample

📜 License
This project is open-source and available under the MIT License.

👤 Author
Developed by @samihadev
Feel free to contribute, suggest improvements, or ask questions.



