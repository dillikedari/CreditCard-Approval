# Credit Card Approval Prediction System

An end-to-end Machine Learning system that predicts whether a credit card application should be **Approved** or **Rejected** based on applicant demographic, financial, and credit risk factors.

---

## Features
* **Exploratory Data Analysis (EDA):** Automatic generation of target distribution, income density, asset correlations, and box plots.
* **Pre-packaged Scikit-Learn Pipelines:** Clean numerical imputation, scaling, and categorical one-hot encoding.
* **4 ML Models Built:** Logistic Regression, Decision Tree, Random Forest, and XGBoost Classifier.
* **Flask Web UI:** Premium responsive glassmorphic design to manually score applicants.
* **Cloud Ready:** Detailed IBM Watson Machine Learning online scoring deployment guide.

---

## Technologies Used
* **Backend Logic & Modeling:** Python, Pandas, Numpy, Scikit-Learn, XGBoost, Joblib, Openpyxl
* **Data Visualization:** Matplotlib, Seaborn
* **Web Server:** Flask, Jinja2, HTML5, Vanilla CSS3

---

## Folder Structure
```
CreditCardApproval/
│
├── app.py                      # Flask Server
├── generate_dataset.py         # Synthetic Excel Data Generator
├── build_project.py            # EDA & Model Training Execution Pipeline
├── model.pkl                   # Trained Best Performing Model Pipeline
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Setup Documentation
├── Credit_Card_Approval_Notebook.ipynb # Jupyter Notebook Deliverable
├── Credit_Card_Approval_Report.md      # Academic Project Report
├── watson_ml_deployment_guide.md       # IBM Watson ML Guide
│
├── static/                     # Web app stylesheet & generated EDA plots
│   ├── style.css
│   ├── approval_dist.png
│   ├── income_dist.png
│   ├── credit_history_dist.png
│   ├── employment_analysis.png
│   ├── correlation_heatmap.png
│   └── model_comparison.png
│
└── templates/                  # Web app HTML template
    └── index.html
```

---

## Installation & Running the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Data & Train Models
```bash
# Generate the Excel applicant dataset
python generate_dataset.py

# Run analysis, generate plots, evaluate models, and save the best pipeline
python build_project.py
```

### 3. Run Flask Web Application
```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## Results Summary
Our models yielded the following metrics:
* **Logistic Regression (Best Model):** F1-Score: 92.9%, Accuracy: 89.3%, ROC-AUC: 93.4%
* **Random Forest Classifier:** F1-Score: 92.8%, Accuracy: 89.3%, ROC-AUC: 91.3%
* **Decision Tree Classifier:** F1-Score: 92.0%, Accuracy: 88.0%, ROC-AUC: 89.4%
* **XGBoost Classifier:** F1-Score: 90.7%, Accuracy: 86.0%, ROC-AUC: 91.9%

The models show high stability, driven strongly by the credit history and payment record.

Dataset Notice :

The original dataset is not included in this repository due to size limitations.
The trained model (`model.pkl`) is included, and the Flask application can be used directly for predictions.
---

## Author
* **Academic Internship Project**
* Built with Python, Scikit-Learn, and Flask.
