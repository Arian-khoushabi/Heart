Abstract
This project introduces an intelligent, end‑to‑end clinical decision‑support system designed to estimate the risk of heart disease by integrating classical Machine Learning (ML) models with a Large Language Model (LLM) for interpretability. A predictive model was trained on the widely used UCI Heart Disease Dataset to estimate the probability of cardiovascular disease based on structured clinical features. To enhance user understanding, an LLM is incorporated through an API to generate human‑readable clinical explanations of the model’s predictions. The system is implemented using FastAPI for backend computation and Streamlit for user interaction, allowing efficient data input, real‑time inference, and transparent interpretability.

1. Introduction
Heart disease remains one of the leading causes of mortality worldwide. Early identification of at‑risk individuals plays a critical role in reducing mortality and improving clinical outcomes. Traditional ML models have demonstrated strong predictive capability; however, their interpretability remains limited for non‑technical users.

To address this gap, this project proposes a hybrid framework that:

Predicts heart‑disease risk using an ML model trained on clinical attributes.
Uses a Large Language Model to produce an accessible clinical explanation based on the prediction.
Provides a fully interactive interface suitable for clinical decision‑support exploration.
2. Dataset
The system utilizes the UCI Heart Disease Dataset, one of the most established datasets in cardiovascular risk prediction research.

Features include:

Age
Sex
Chest pain type
Resting blood pressure
Serum cholesterol
Fasting blood sugar
Resting ECG results
Maximum heart rate achieved
Exercise-induced angina
ST depression (Oldpeak)
The dataset is preprocessed to ensure data consistency and optimal model performance.

3. Methodology
3.1 Machine Learning Model
A predictive classification model is developed using Python and Scikit‑Learn. Feature scaling, encoding, and preprocessing are implemented to prepare clinical variables for training.

The ML model outputs a probabilistic estimation of the likelihood of heart disease.

3.2 LLM‑Based Interpretation
To bridge the interpretability gap, an LLM is integrated using an external API.

Given the ML model’s output, the LLM generates:

A narrative clinical explanation
Insight into contributing variables
A user‑friendly summary of patient risk
This improves transparency and supports non‑technical stakeholders.

3.3 System Architecture
The system is composed of two main layers:

Backend (FastAPI):

Handles data preprocessing, prediction, and communication with the LLM API.

Frontend (Streamlit):

Provides a clean interface where users can input patient data, perform predictions, and view explanations.

Data flow is real‑time, enabling an interactive and educational user experience.

4. Implementation
4.1 Project Structure

Heart/
│── app.py                # FastAPI backend
│── main.py               # Streamlit user interface
│── model_preprocess.py   # Data preprocessing functions
│── myModel.py            # Model training script
│── Heart_ai.pkl          # Trained ML model
│── heart.ipynb           # Notebook for analysis and experiments
│── UCI_Heart_Disease...  # Dataset file
│── requirements.txt      # Dependencies
│── __pycache__/          # Cache files (ignored)

4.2 Running the System
Install dependencies:

pip install -r requirements.txt
Start FastAPI backend:

uvicorn app:app --reload
Launch Streamlit interface:

streamlit run main.py

5. Results
The integrated system demonstrates that combining a predictive ML model with an LLM‑based explanation module can significantly enhance the interpretability of clinical risk assessments.

The ML component provides numerical probabilities, while the LLM translates these into a coherent medical narrative. This hybrid approach has potential value for early screening tools and educational clinical applications.

6. Discussion and Future Work
Potential future extensions include:

Incorporation of SHAP or LIME for quantitative feature attribution
Deployment on cloud platforms such as AWS, Azure, or Google Cloud
Expansion to additional cardiovascular datasets
Integration of multimodal data (e.g., ECG signals, images)
Fine‑tuning domain‑specific LLMs for improved medical reasoning

7. Conclusion
This project demonstrates an effective fusion of classical machine learning and large language models to provide both prediction accuracy and interpretability in heart‑disease risk estimation. The combination of FastAPI, Streamlit, and LLM‑powered explanations forms a practical and educational tool for early clinical decision support.

8. References
UCI Machine Learning Repository: Heart Disease Dataset
Scikit‑Learn Documentation
FastAPI Documentation
Streamlit Documentation
