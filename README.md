# ❤️ Heart Disease Prediction
*by Arian Khoushabi*

A machine learning application for predicting the presence of heart disease using clinical features from the UCI Heart Disease dataset.

The project uses an **XGBoost classifier** combined with a preprocessing pipeline and a custom decision-threshold selection strategy designed to prioritize **recall**, which is particularly important in a heart disease screening context.

## 🚀 Live Demo

Try the deployed Streamlit application:

**[Heart Disease Prediction App](https://qlr4xappf9mppkq5bk8juq7.streamlit.app/)**

---

## 📌 Project Overview

Heart disease is one of the major health challenges worldwide. Machine learning can be used to identify patterns in clinical data and provide predictions that may assist with risk assessment.

This project develops an end-to-end machine learning pipeline that:

- Preprocesses categorical and numerical clinical features
- Uses **XGBoost** for classification
- Evaluates multiple classification metrics
- Optimizes the classification threshold with a focus on recall
- Saves the trained model for deployment
- Provides an interactive Streamlit web application

> **Important:** This project is an educational machine learning project and is **not a medical diagnostic tool**. The model should not be used for clinical decision-making.

---

## 📊 Dataset

The project uses a combined version of the **UCI Heart Disease dataset**.

The dataset contains clinical attributes commonly used in heart disease prediction, including:

- Age
- Sex
- Chest pain type
- Resting blood pressure
- Cholesterol
- Fasting blood sugar
- Resting ECG
- Maximum heart rate
- Exercise-induced angina
- ST depression
- ST slope
- Other clinical measurements

### Target

The target variable is:

`HeartDisease`

where the model predicts whether heart disease is present.

---

## 🧠 Machine Learning Approach

### Preprocessing

The preprocessing pipeline separates categorical and numerical features.

Categorical variables are encoded using:

- `OneHotEncoder`

Numerical features are transformed using:

- `PowerTransformer`

The preprocessing and model are combined into a **scikit-learn Pipeline**, helping ensure that the same transformations are applied consistently during training and prediction.

### Model

The classifier used in this project is:

**XGBoost (`XGBClassifier`)**

XGBoost was selected because of its strong performance on structured/tabular data and its ability to model nonlinear relationships between clinical features.

---

## 🎯 Decision Threshold Optimization

Instead of automatically using the default classification threshold of `0.5`, the project selects a threshold using validation data.

The threshold-selection strategy prioritizes **recall**, because missing a patient with potential heart disease can be more concerning than generating some additional false positives in a screening-oriented setting.

A minimum validation recall requirement of:

**0.85**

was used when selecting the decision threshold.

The selected threshold was approximately:

**0.4074**

### Why recall?

Recall is defined as:

```text
Recall = TP / (TP + FN)

A higher recall means the model identifies a larger proportion of positive cases.

For a screening-oriented application, reducing false negatives can be particularly important.

📈 Model Performance

The final model achieved the following performance on the test set:

Metric	Score
Accuracy	85.57%
Precision	83.43%
Recall	91.95%
F1 Score	87.48%
ROC-AUC	92.03%
PR-AUC	92.96%
Specificity	77.82%

The model's relatively high recall reflects the project's emphasis on identifying as many potential positive cases as possible.

🏗️ Project Structure
Heart/
│
├── app.py
├── model_preprocess.py
├── requirements.txt
├── README.md
│
├── data/
│   └── UCI_Heart_Disease_Dataset_Combined.csv
│
└── model/
    └── heart_model.pkl

The exact structure may vary depending on the current version of the repository.

🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Joblib
Streamlit
Main libraries
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
streamlit
⚙️ Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd Heart

Create a virtual environment:

python -m venv .venv

Activate it on Windows:

.venv\Scripts\activate

Install the dependencies:

pip install -r requirements.txt
▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will open in your browser.

🔬 Methodology

The general workflow of the project is:

Raw Dataset
     ↓
Data Cleaning
     ↓
Train / Validation / Test Split
     ↓
Feature Preprocessing
     ↓
XGBoost Classifier
     ↓
Validation Predictions
     ↓
Decision Threshold Selection
     ↓
Final Test Evaluation
     ↓
Saved Model
     ↓
Streamlit Deployment

The decision threshold is selected using validation data rather than the test set in order to avoid using test data for model-selection decisions.

⚠️ Limitations

Although the model demonstrates promising performance, several limitations should be considered.

Dataset limitations

The UCI Heart Disease dataset is a relatively small, structured dataset and does not represent the complexity and diversity of modern clinical datasets.

Real-world healthcare systems may contain:

Electronic Health Records (EHRs)
ECG signals
Medical imaging
Longitudinal patient histories
Laboratory measurements
Larger and more diverse patient populations

Therefore, the reported performance should not be interpreted as evidence of clinical effectiveness.

Generalization

The model has been evaluated on a held-out test set from the same dataset distribution. External validation using independent clinical datasets would be necessary before considering real-world clinical applications.

Clinical use

This application is intended for educational and research purposes only and should not replace professional medical assessment.

🔮 Future Improvements

Potential future improvements include:

External validation using independent datasets
Evaluation on larger clinical datasets
Hyperparameter optimization
Calibration of predicted probabilities
More extensive cross-validation
Explainable AI techniques such as SHAP
Feature importance analysis
Comparison with other machine learning models
Integration with real-world clinical datasets
Improved monitoring and model validation for deployment
📚 References
UCI Machine Learning Repository — Heart Disease Dataset
Chen, T. & Guestrin, C. — XGBoost: A Scalable Tree Boosting System
Scikit-learn documentation
XGBoost documentation
Streamlit documentation
👨‍💻 Author

Arian

This project was developed as a machine learning project focused on applying classification techniques to healthcare-related tabular data.

⭐ Disclaimer

This project is not intended for medical diagnosis, treatment, or clinical decision-making.

Predictions generated by the application should not be considered medical advice. Any health-related decision should be made in consultation with a qualified healthcare professional.