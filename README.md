# ❤️ CardioAI — Heart Disease Prediction App

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=flat-square&logo=streamlit)
![ML](https://img.shields.io/badge/Machine%20Learning-KNN-green?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

> An AI-powered cardiovascular risk assessment tool built with Python and Streamlit. Enter patient details and get an instant heart disease prediction using a K-Nearest Neighbours model trained on 918 patients.

🔗 **Live Demo:** [heartdisease007.streamlit.app](https://heartdisease007.streamlit.app)

---

## 📸 Screenshots

| Home | EDA | Model Comparison | Predict |
|------|-----|-----------------|---------|
| Overview & stats | Data visualizations | 6 model benchmarks | Patient risk assessment |

---

## ✨ Features

- 🏠 **Home** — Project overview with key stats (918 patients, 89.1% accuracy)
- 📊 **EDA** — 6 interactive charts including age distribution, sex vs disease, chest pain analysis, cholesterol boxplots, correlation heatmap, and key data insights
- 🤖 **Model Comparison** — Benchmarks 6 ML algorithms side by side with Accuracy, F1, Precision, Recall, ROC-AUC, ROC curves, confusion matrix, and feature importance
- 🔍 **Predict** — Real-time heart disease risk prediction with automatic risk flag detection
- 🌙 **Dark / Light Mode** — Toggle between themes

---

## 🤖 Models Compared

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| **KNN** ⭐ | **89.1%** | **90.3%** | **93.0%** |
| Logistic Regression | 89.1% | 90.3% | 93.3% |
| Naive Bayes | 87.5% | 88.4% | 94.3% |
| SVM | 85.9% | 87.5% | 94.2% |
| Random Forest | 85.3% | 86.6% | 92.9% |
| Decision Tree | 74.5% | 76.8% | 74.2% |

KNN was selected as the final model — non-parametric, no distribution assumptions, and highly interpretable for medical use cases.

---

## 📁 Project Structure

```
HeartDisease/
├── app.py              # Main Streamlit application
├── heart.csv           # Dataset (918 patients, 12 features)
├── KNN_heart.pkl       # Trained KNN model
├── scaler.pkl          # StandardScaler for feature normalisation
├── columns.pkl         # Feature column names
└── requirements.txt    # Python dependencies
```

---

## 🧠 Dataset

- **Source:** Heart Failure Prediction Dataset (Kaggle)
- **Size:** 918 patients
- **Features:** Age, Sex, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, Resting ECG, Max Heart Rate, Exercise Angina, Oldpeak, ST Slope
- **Target:** HeartDisease (0 = No, 1 = Yes)
- **Class balance:** 55.3% positive, 44.7% negative

### Data Preprocessing
- Zero values in Cholesterol and RestingBP replaced with column mean (medically invalid zeros)
- Categorical features one-hot encoded using `pd.get_dummies`
- Numerical features normalised with `StandardScaler`
- 80/20 stratified train-test split

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/yourusername/HeartDisease.git
cd HeartDisease

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 📦 Requirements

```
streamlit
pandas
numpy
joblib
scikit-learn
matplotlib
seaborn
```

---

## 📊 Key Findings

- **ST Slope** is the strongest predictor — flat slope strongly indicates disease (r = 0.55)
- **Exercise Angina** is a major red flag (r = 0.49)
- **Max Heart Rate** — lower values strongly correlate with disease (r = -0.40)
- **Sex disparity** — 63% of male patients have heart disease vs 26% of female patients
- Heart disease patients average **56 years old** vs 51 for healthy patients

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider.

---

## 👩‍💻 Author

Built by **RAJESH** as a machine learning portfolio project.

---

*If you found this useful, please ⭐ star the repo!*
