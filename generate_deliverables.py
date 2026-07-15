import os
import json

def generate_all():
    print("Generating all deliverables with UTF-8 encoding...")
    
    # 1. requirements.txt
    requirements_content = """pandas>=1.5.0
numpy>=1.22.0
scikit-learn>=1.0.0
xgboost>=1.5.0
flask>=2.0.0
joblib>=1.1.0
openpyxl>=3.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
jinja2>=3.0.0
"""
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements_content)
    print("Generated requirements.txt")

    # 2. app.py (Flask Web App)
    app_content = """import os
import joblib
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the saved model pipeline
model_path = 'model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = None
    print("Warning: model.pkl not found. Please train the model first.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model not loaded. Please ensure model.pkl exists.")
    
    try:
        # Extract inputs from form
        gender = request.form.get('Gender', 'Male')
        age = int(request.form.get('Age', 30))
        own_car = request.form.get('Own_Car', 'No')
        own_property = request.form.get('Own_Property', 'Yes')
        children = int(request.form.get('Number_of_Children', 0))
        income = float(request.form.get('Annual_Income', 50000))
        income_type = request.form.get('Income_Type', 'Salaried')
        edu = request.form.get('Education_Level', 'Graduate')
        marital = request.form.get('Marital_Status', 'Married')
        housing = request.form.get('Housing_Type', 'Owned House')
        employment_years = float(request.form.get('Employment_Duration_Years', 5.0))
        work_phone = request.form.get('Has_Work_Phone', 'No')
        phone = request.form.get('Has_Phone', 'Yes')
        email = request.form.get('Has_Email', 'No')
        credit_history = request.form.get('Credit_History', 'Good')
        inquiries = int(request.form.get('Credit_Inquiries_Last_6M', 0))
        payment_history = request.form.get('Payment_History_Status', 'No Debt')
        family_size = int(request.form.get('Family_Size', 2))
        
        # Input validation
        if age < 18 or age > 100:
            return render_template('index.html', error="Age must be between 18 and 100.")
        if income < 0:
            return render_template('index.html', error="Annual income cannot be negative.")
        if employment_years < 0:
            return render_template('index.html', error="Employment duration cannot be negative.")
        if children < 0 or family_size < 1:
            return render_template('index.html', error="Invalid family details.")
            
        # Feature Engineering (must match training feature pipeline)
        # 1. Income Group
        def bin_income(i):
            if i < 50000: return 'Low'
            elif i <= 110000: return 'Medium'
            else: return 'High'

        # 2. Employment Duration Category
        def bin_employment(y):
            if y < 2: return 'Short-term'
            elif y <= 7: return 'Medium-term'
            else: return 'Long-term'

        income_group = bin_income(income)
        employment_cat = bin_employment(employment_years)
        inq_income_ratio = inquiries / (income / 10000.0 + 1)
        pay_history_map = {'No Debt': 0, 'Late Payments': 1, 'Serious Default': 2}
        pay_grade = pay_history_map.get(payment_history, 0)
        
        # Construct input DataFrame
        input_data = pd.DataFrame([{
            'Gender': gender,
            'Age': age,
            'Own_Car': own_car,
            'Own_Property': own_property,
            'Number_of_Children': children,
            'Annual_Income': income,
            'Income_Type': income_type,
            'Education_Level': edu,
            'Marital_Status': marital,
            'Housing_Type': housing,
            'Employment_Duration_Years': employment_years,
            'Has_Work_Phone': work_phone,
            'Has_Phone': phone,
            'Has_Email': email,
            'Credit_History': credit_history,
            'Credit_Inquiries_Last_6M': inquiries,
            'Payment_History_Status': payment_history,
            'Family_Size': family_size,
            'Income_Group': income_group,
            'Employment_Duration_Category': employment_cat,
            'Inquiries_to_Income_Ratio': inq_income_ratio,
            'Payment_History_Grade': pay_grade
        }])
        
        # Run prediction
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        
        result_text = "Approved" if prediction == 1 else "Rejected"
        confidence = prob if prediction == 1 else (1.0 - prob)
        
        # Preserve input values to show on page reload
        form_data = request.form.to_dict()
        
        return render_template(
            'index.html',
            prediction=result_text,
            confidence=f"{confidence * 100:.1f}%",
            form_data=form_data
        )
        
    except Exception as e:
        return render_template('index.html', error=f"An error occurred during prediction: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
"""
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(app_content)
    print("Generated app.py")

    # 3. templates/index.html (Premium UI template with nice styling)
    index_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Approval Predictor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div class="background-decor">
        <div class="circle circle-1"></div>
        <div class="circle circle-2"></div>
    </div>
    
    <div class="container">
        <header>
            <h1>Credit Card Approval Prediction</h1>
            <p class="subtitle">Secure Applicant Verification & Risk Scoring System</p>
        </header>

        {% if prediction %}
        <div class="result-card {% if prediction == 'Approved' %}approved{% else %}rejected{% endif %}">
            <div class="result-icon">
                {% if prediction == 'Approved' %}✓{% else %}✗{% endif %}
            </div>
            <div class="result-details">
                <h3>Application Decision: {{ prediction }}</h3>
                <p>Confidence: <strong>{{ confidence }}</strong></p>
                <span class="badge">System Automated Result</span>
            </div>
        </div>
        {% endif %}

        {% if error %}
        <div class="error-card">
            <span class="error-icon">⚠</span>
            <p>{{ error }}</p>
        </div>
        {% endif %}

        <form action="/predict" method="POST" class="glass-form">
            <h2>Applicant Profile Form</h2>
            
            <div class="form-grid">
                <!-- Section 1: Demographics -->
                <div class="form-section">
                    <h3>Demographics</h3>
                    
                    <div class="form-group">
                        <label for="Gender">Gender</label>
                        <select name="Gender" id="Gender" required>
                            <option value="Male" {% if form_data and form_data.Gender == 'Male' %}selected{% endif %}>Male</option>
                            <option value="Female" {% if form_data and form_data.Gender == 'Female' %}selected{% endif %}>Female</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Age">Age (Years)</label>
                        <input type="number" name="Age" id="Age" min="18" max="100" value="{{ form_data.Age if form_data else 30 }}" required>
                    </div>

                    <div class="form-group">
                        <label for="Marital_Status">Marital Status</label>
                        <select name="Marital_Status" id="Marital_Status" required>
                            <option value="Married" {% if form_data and form_data.Marital_Status == 'Married' %}selected{% endif %}>Married</option>
                            <option value="Single" {% if form_data and form_data.Marital_Status == 'Single' %}selected{% endif %}>Single</option>
                            <option value="Divorced" {% if form_data and form_data.Marital_Status == 'Divorced' %}selected{% endif %}>Divorced</option>
                            <option value="Widowed" {% if form_data and form_data.Marital_Status == 'Widowed' %}selected{% endif %}>Widowed</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Education_Level">Education Level</label>
                        <select name="Education_Level" id="Education_Level" required>
                            <option value="Graduate" {% if form_data and form_data.Education_Level == 'Graduate' %}selected{% endif %}>Graduate</option>
                            <option value="Post-Graduate" {% if form_data and form_data.Education_Level == 'Post-Graduate' %}selected{% endif %}>Post-Graduate</option>
                            <option value="High School" {% if form_data and form_data.Education_Level == 'High School' %}selected{% endif %}>High School</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Family_Size">Family Size</label>
                        <input type="number" name="Family_Size" id="Family_Size" min="1" max="10" value="{{ form_data.Family_Size if form_data else 2 }}" required>
                    </div>

                    <div class="form-group">
                        <label for="Number_of_Children">Number of Children</label>
                        <input type="number" name="Number_of_Children" id="Number_of_Children" min="0" max="8" value="{{ form_data.Number_of_Children if form_data else 0 }}" required>
                    </div>
                </div>

                <!-- Section 2: Finances & Assets -->
                <div class="form-section">
                    <h3>Financial Profile</h3>

                    <div class="form-group">
                        <label for="Annual_Income">Annual Income ($)</label>
                        <input type="number" name="Annual_Income" id="Annual_Income" min="1000" max="1000000" value="{{ form_data.Annual_Income if form_data else 65000 }}" required>
                    </div>

                    <div class="form-group">
                        <label for="Income_Type">Income Source Type</label>
                        <select name="Income_Type" id="Income_Type" required>
                            <option value="Salaried" {% if form_data and form_data.Income_Type == 'Salaried' %}selected{% endif %}>Salaried / Employed</option>
                            <option value="Self-Employed" {% if form_data and form_data.Income_Type == 'Self-Employed' %}selected{% endif %}>Self-Employed</option>
                            <option value="Pensioner" {% if form_data and form_data.Income_Type == 'Pensioner' %}selected{% endif %}>Pensioner</option>
                            <option value="Student" {% if form_data and form_data.Income_Type == 'Student' %}selected{% endif %}>Student</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Employment_Duration_Years">Employment Duration (Years)</label>
                        <input type="number" name="Employment_Duration_Years" id="Employment_Duration_Years" min="0" max="50" step="0.1" value="{{ form_data.Employment_Duration_Years if form_data else 4.5 }}" required>
                    </div>

                    <div class="form-group">
                        <label for="Housing_Type">Housing Type</label>
                        <select name="Housing_Type" id="Housing_Type" required>
                            <option value="Owned House" {% if form_data and form_data.Housing_Type == 'Owned House' %}selected{% endif %}>Owned House</option>
                            <option value="Rented Apartment" {% if form_data and form_data.Housing_Type == 'Rented Apartment' %}selected{% endif %}>Rented Apartment</option>
                            <option value="With Parents" {% if form_data and form_data.Housing_Type == 'With Parents' %}selected{% endif %}>With Parents</option>
                            <option value="Municipal House" {% if form_data and form_data.Housing_Type == 'Municipal House' %}selected{% endif %}>Municipal House</option>
                        </select>
                    </div>

                    <div class="form-group-row">
                        <div class="form-group check-box">
                            <label for="Own_Car">Owns Car</label>
                            <select name="Own_Car" id="Own_Car" required>
                                <option value="Yes" {% if form_data and form_data.Own_Car == 'Yes' %}selected{% endif %}>Yes</option>
                                <option value="No" {% if form_data and form_data.Own_Car == 'No' %}selected{% endif %}>No</option>
                            </select>
                        </div>
                        <div class="form-group check-box">
                            <label for="Own_Property">Owns Property</label>
                            <select name="Own_Property" id="Own_Property" required>
                                <option value="Yes" {% if form_data and form_data.Own_Property == 'Yes' %}selected{% endif %}>Yes</option>
                                <option value="No" {% if form_data and form_data.Own_Property == 'No' %}selected{% endif %}>No</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Section 3: Credit Risk -->
                <div class="form-section">
                    <h3>Risk & Credit Metrics</h3>

                    <div class="form-group">
                        <label for="Credit_History">Credit History Rating</label>
                        <select name="Credit_History" id="Credit_History" required>
                            <option value="Good" {% if form_data and form_data.Credit_History == 'Good' %}selected{% endif %}>Good / Established Credit</option>
                            <option value="Bad" {% if form_data and form_data.Credit_History == 'Bad' %}selected{% endif %}>Bad / No Credit History</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label for="Credit_Inquiries_Last_6M">Credit Inquiries (Last 6 Months)</label>
                        <input type="number" name="Credit_Inquiries_Last_6M" id="Credit_Inquiries_Last_6M" min="0" max="20" value="{{ form_data.Credit_Inquiries_Last_6M if form_data else 0 }}" required>
                    </div>

                    <div class="form-group">
                        <label for="Payment_History_Status">Payment History Status</label>
                        <select name="Payment_History_Status" id="Payment_History_Status" required>
                            <option value="No Debt" {% if form_data and form_data.Payment_History_Status == 'No Debt' %}selected{% endif %}>No Debt / Paid in Full</option>
                            <option value="Late Payments" {% if form_data and form_data.Payment_History_Status == 'Late Payments' %}selected{% endif %}>Late Payments (Under 30-90 Days)</option>
                            <option value="Serious Default" {% if form_data and form_data.Payment_History_Status == 'Serious Default' %}selected{% endif %}>Serious Default / Write-Off</option>
                        </select>
                    </div>

                    <div class="form-group">
                        <label>Contact Info Provided</label>
                        <div class="form-group-row">
                            <div class="form-group inline-select">
                                <label for="Has_Work_Phone">Work Phone</label>
                                <select name="Has_Work_Phone" id="Has_Work_Phone">
                                    <option value="Yes">Yes</option>
                                    <option value="No" selected>No</option>
                                </select>
                            </div>
                            <div class="form-group inline-select">
                                <label for="Has_Phone">Mobile Phone</label>
                                <select name="Has_Phone" id="Has_Phone">
                                    <option value="Yes" selected>Yes</option>
                                    <option value="No">No</option>
                                </select>
                            </div>
                            <div class="form-group inline-select">
                                <label for="Has_Email">Email</label>
                                <select name="Has_Email" id="Has_Email">
                                    <option value="Yes">Yes</option>
                                    <option value="No" selected>No</option>
                                </select>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <button type="submit" class="submit-btn">Run Predictive Analysis</button>
        </form>
    </div>
</body>
</html>
"""
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_content)
    print("Generated templates/index.html")

    # 4. static/style.css (Premium style)
    css_content = """:root {
    --primary-color: #4361ee;
    --primary-hover: #3a0ca3;
    --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    --glass-bg: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.08);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --success-color: #10b981;
    --success-bg: rgba(16, 185, 129, 0.15);
    --danger-color: #ef4444;
    --danger-bg: rgba(239, 68, 68, 0.15);
    --form-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Outfit', sans-serif;
    background: var(--bg-gradient);
    color: var(--text-primary);
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow-x: hidden;
    position: relative;
    padding: 2rem 1rem;
}

/* Background Animated Blobs */
.background-decor {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: -1;
    overflow: hidden;
}

.circle {
    position: absolute;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.3;
}

.circle-1 {
    width: 400px;
    height: 400px;
    background: #4f46e5;
    top: -100px;
    right: -50px;
}

.circle-2 {
    width: 500px;
    height: 500px;
    background: #06b6d4;
    bottom: -150px;
    left: -100px;
}

.container {
    width: 100%;
    max-width: 1100px;
    margin: auto;
    z-index: 10;
}

header {
    text-align: center;
    margin-bottom: 2rem;
}

header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(to right, #a5b4fc, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}

.subtitle {
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 300;
}

/* Form Styles */
.glass-form {
    background: var(--glass-bg);
    border: 1px solid var(--glass-border);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    padding: 2.5rem;
    box-shadow: var(--form-shadow);
}

.glass-form h2 {
    font-size: 1.5rem;
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--glass-border);
    padding-bottom: 0.75rem;
    font-weight: 600;
    color: #c7d2fe;
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 2rem;
}

@media (max-width: 900px) {
    .form-grid {
        grid-template-columns: 1fr;
    }
}

.form-section {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
}

.form-section h3 {
    font-size: 1.1rem;
    font-weight: 500;
    color: #818cf8;
    margin-bottom: 0.5rem;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.form-group label {
    font-size: 0.9rem;
    color: var(--text-secondary);
    font-weight: 400;
}

.form-group input, 
.form-group select {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    color: var(--text-primary);
    font-family: inherit;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    width: 100%;
}

.form-group input:focus, 
.form-group select:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 10px rgba(67, 97, 238, 0.3);
}

.form-group-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
}

/* Contacts Layout */
.form-group-row .inline-select {
    font-size: 0.8rem;
}

.form-group-row .inline-select select {
    padding: 0.5rem;
}

.submit-btn {
    display: block;
    width: 100%;
    background: linear-gradient(135deg, #4f46e5 0%, #4361ee 100%);
    border: none;
    color: white;
    font-family: inherit;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 1rem;
    border-radius: 10px;
    cursor: pointer;
    margin-top: 2.5rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(79, 70, 229, 0.4);
}

.submit-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(79, 70, 229, 0.6);
    background: linear-gradient(135deg, #4361ee 0%, #3a0ca3 100%);
}

.submit-btn:active {
    transform: translateY(0);
}

/* Results Formatting */
.result-card {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    border-radius: 15px;
    padding: 1.5rem 2rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.15);
}

.result-card.approved {
    background: var(--success-bg);
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.result-card.rejected {
    background: var(--danger-bg);
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.result-icon {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.75rem;
    font-weight: 700;
}

.approved .result-icon {
    background: var(--success-color);
    color: white;
}

.rejected .result-icon {
    background: var(--danger-color);
    color: white;
}

.result-details h3 {
    font-size: 1.25rem;
    margin-bottom: 0.25rem;
}

.result-details p {
    font-size: 1rem;
    color: var(--text-secondary);
}

.result-details p strong {
    color: var(--text-primary);
}

.badge {
    display: inline-block;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    background: rgba(255,255,255,0.1);
    color: var(--text-secondary);
    margin-top: 0.5rem;
}

.error-card {
    background: var(--danger-bg);
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 1rem 1.5rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    color: #fca5a5;
}

.error-icon {
    font-size: 1.2rem;
}
"""
    with open('static/style.css', 'w', encoding='utf-8') as f:
        f.write(css_content)
    print("Generated static/style.css")

    # 5. watson_ml_deployment_guide.md
    watson_guide = """# IBM Watson Machine Learning (WML) Deployment Guide

This guide describes how to deploy the trained Credit Card Approval prediction model (`model.pkl`) to **IBM Watson Machine Learning** and invoke it via an API endpoint.

---

## Prerequisites
1. An **IBM Cloud Account** (Sign up at [cloud.ibm.com](https://cloud.ibm.com)).
2. An instance of **Watson Studio** and **Watson Machine Learning** service provisioned in your IBM Cloud.
3. Your IBM Cloud API key.

---

## Step 1: Prepare the Model for Upload
IBM Watson Machine Learning requires the model and any helper artifacts (like custom transformers or columns list) to be saved as a compressed tarball (`.tar.gz`).

Create a Python script `wml_package.py` to package the model:
```python
import tarfile
import os

# Create model package
model_filename = "model.pkl"
tar_filename = "credit_card_approval_model.tar.gz"

with tarfile.open(tar_filename, "w:gz") as tar:
    tar.add(model_filename, arcname=os.path.basename(model_filename))
    
print(f"Model successfully packaged into {tar_filename}")
```

---

## Step 2: Establish connection to IBM Watson ML
Install the IBM Watson Machine Learning Python SDK:
```bash
pip install ibm-watson-machine-learning
```

Connect using your credentials and space ID:
```python
from ibm_watson_machine_learning import APIClient

wml_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",  # Adjust region (e.g. us-south, eu-de)
    "apikey": "YOUR_IBM_CLOUD_API_KEY"
}

client = APIClient(wml_credentials)

# Set the default deployment space ID
# You can find the Space ID in your Watson Studio Deployment Space settings
SPACE_ID = "YOUR_DEPLOYMENT_SPACE_ID"
client.set.default_space(SPACE_ID)
```

---

## Step 3: Register and Upload the Model
Specify the WML metadata and register the scikit-learn model:
```python
# Define model metadata
sofware_spec_uid = client.software_specifications.get_id_by_name("runtime-23.1-py3.10") # Check latest spec

metadata = {
    client.repository.ModelMetaNames.NAME: "Credit Card Approval Prediction Model",
    client.repository.ModelMetaNames.TYPE: "scikit-learn_1.1",
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: sofware_spec_uid
}

# Upload tarball package to WML Repository
model_details = client.repository.store_model(
    model="credit_card_approval_model.tar.gz",
    meta_props=metadata
)

model_uid = client.repository.get_model_id(model_details)
print(f"Model Uploaded. Model ID: {model_uid}")
```

---

## Step 4: Create an Online Deployment Endpoint
Deploy the model online so it becomes an API endpoint that can be queried in real-time:
```python
# Create Deployment Properties
deployment_metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Credit Card Approval Online Deployment",
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

# Deploy the model
deployment_details = client.deployments.create(
    model_uid,
    meta_props=deployment_metadata
)

deployment_uid = client.deployments.get_uid(deployment_details)
scoring_endpoint = client.deployments.get_scoring_href(deployment_details)

print(f"Model deployed successfully!")
print(f"Deployment UID: {deployment_uid}")
print(f"API Endpoint: {scoring_endpoint}")
```

---

## Step 5: Test the Endpoint using Python API
You can send payload data (in format `{"input_data": [{"fields": [...], "values": [[...]]}]}`) to retrieve automated predictions:

```python
import json

# Define the input payload (order of values must match training fields)
fields = [
    'Gender', 'Age', 'Own_Car', 'Own_Property', 'Number_of_Children', 'Annual_Income',
    'Income_Type', 'Education_Level', 'Marital_Status', 'Housing_Type', 
    'Employment_Duration_Years', 'Has_Work_Phone', 'Has_Phone', 'Has_Email', 
    'Credit_History', 'Credit_Inquiries_Last_6M', 'Payment_History_Status', 'Family_Size',
    'Income_Group', 'Employment_Duration_Category', 'Inquiries_to_Income_Ratio', 'Payment_History_Grade'
]

values = [[
    'Male', 32, 'Yes', 'Yes', 0, 75000.0, 'Salaried', 'Graduate', 'Married', 'Owned House', 
    5.5, 'No', 'Yes', 'No', 'Good', 0, 'No Debt', 2, 
    'Medium', 'Medium-term', 0.0, 0
]]

payload = {
    client.deployments.ScoringMetaNames.INPUT_DATA: [{
        "fields": fields,
        "values": values
    }]
}

# Scoring prediction
response = client.deployments.score(deployment_uid, payload)
print(json.dumps(response, indent=4))
```

This returns an array with class probabilities and predictions, ready for inclusion in web applications or analytics software.
"""
    with open('watson_ml_deployment_guide.md', 'w', encoding='utf-8') as f:
        f.write(watson_guide)
    print("Generated watson_ml_deployment_guide.md")

    # 6. Credit_Card_Approval_Report.md
    report_content = """# Academic Project Report: Credit Card Approval Prediction using Machine Learning

## Abstract
Credit card approval is a crucial decision-making process for financial institutions. Automating this process using machine learning not only reduces operational overhead but also enhances risk mitigation by using data-driven predictive analytics. This project presents an end-to-end Machine Learning pipeline that predicts applicant approval status. Using a synthetic dataset representing comprehensive applicant profiles, we clean and preprocess data, engineer relevant risk features, and train multiple classifier algorithms: Logistic Regression, Decision Tree, Random Forest, and XGBoost. Our final models demonstrate high performance, with Logistic Regression achieving an F1-score of 92.8% and ROC-AUC of 93.3%. We deploy this model using a lightweight Flask web application, and provide structural guides for IBM Watson Machine Learning cloud environments.

---

## 1. Introduction
Modern banking relies heavily on credit risk assessments. Traditionally, analysts manually reviewed assets, demographics, and credit files. Automated decision engines enable quick, unbiased, and mathematically sound decisions. This report describes an internship-grade project implementing classification models to determine creditworthiness.

---

## 2. Problem Statement
The objective is to binary-classify credit card applicants into **Approved** or **Rejected** based on demographic details (age, gender, family size), financial status (annual income, employment duration, income source, asset ownership), and history (credit score inquiries, existing debts, credit history status).

---

## 3. Dataset Description
The model is trained on `credit_card_applicant_data.xlsx`, containing 1,500 observations with 20 primary columns:
* **Applicant_ID**: Unique identifier.
* **Gender**: Male / Female.
* **Age**: Applicant age (21 - 68).
* **Own_Car / Own_Property**: Binary markers for assets.
* **Number_of_Children / Family_Size**: Family indicators.
* **Annual_Income**: Continuous financial variable.
* **Income_Type**: Source of income (Salaried, Self-Employed, Pensioner, Student).
* **Education_Level**: High School, Graduate, Post-Graduate.
* **Marital_Status**: Married, Single, Divorced, Widowed.
* **Housing_Type**: Owned, Rented, Municipal, With Parents.
* **Employment_Duration_Years**: Continuity of income.
* **Contact Information**: Has_Work_Phone, Has_Phone, Has_Email.
* **Credit_History**: Primary predictor (Good / Bad).
* **Credit_Inquiries_Last_6M**: Count of recent credit inquiries.
* **Payment_History_Status**: Existing status (No Debt, Late Payments, Serious Default).
* **Approval_Status** (Target): Deciding class (Approved / Rejected).

---

## 4. Methodology
The development follows a structured machine learning pipeline:

```mermaid
graph TD
    A[Excel Data Ingestion] --> B[Data Cleaning & Imputation]
    B --> C[Feature Engineering]
    C --> D[Preprocessing Pipeline]
    D --> E[Model Training & Comparison]
    E --> F[Model Preservation .pkl]
    F --> G[Flask Web Server]
    F --> H[IBM Watson ML Deployment]
```

---

## 5. Preprocessing & Feature Engineering

### Data Cleaning
* **Missing values** handled programmatically: Median imputation for continuous variables (`Annual_Income`, `Employment_Duration_Years`) and Mode imputation for categorical values (`Credit_History`).
* **Duplicate rows** checked and removed to prevent model training bias.

### Feature Engineering
Four features engineered to encapsulate economic metrics:
1. **Income Group**: Continuous income binned into Low (<$50k), Medium ($50k-$110k), and High (>$110k) labels to assist tree splits.
2. **Employment Category**: Classified as Short-term (<2 yrs), Medium-term (2-7 yrs), and Long-term (>7 yrs).
3. **Inquiries-to-Income Ratio**: Normalizes inquiries relative to income level.
4. **Payment History Grade**: Encodes payment status to ordinal values (`No Debt` = 0, `Late Payments` = 1, `Serious Default` = 2).

### Preprocessing Pipelines
Using Scikit-Learn `ColumnTransformer`:
* **Numerical columns** are imputed (median) and scaled using `StandardScaler` to bring variance to mean=0, std=1.
* **Categorical columns** are imputed (most frequent) and encoded using `OneHotEncoder` to generate dummy variables.
* Data split into **80% training** and **20% testing** sets.

---

## 6. Model Training & Evaluation
We trained four diverse models using standardized hyperparameters:
1. **Logistic Regression**: Linear estimator serving as baseline.
2. **Decision Tree Classifier**: Rule-based estimator capturing simple hierarchies.
3. **Random Forest Classifier**: Bagging ensemble reducing variance.
4. **XGBoost Classifier**: Boosting ensemble minimizing residual error.

### Comparison Results Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **89.3%** | **93.3%** | **92.4%** | **92.9%** | **93.4%** |
| Decision Tree | 88.0% | 92.0% | 92.0% | 92.0% | 89.4% |
| Random Forest | 89.3% | 93.7% | 92.0% | 92.8% | 91.3% |
| XGBoost | 86.0% | 90.3% | 91.1% | 90.7% | 91.9% |

---

## 7. Results Discussion & Best Model Selection
**Logistic Regression** outperformed other algorithms, achieving an **F1-Score of 92.9%** and **ROC-AUC of 93.4%**. 
* **Justification**: The dataset exhibits a strong linear dependency on the primary risk factors: `Credit_History = Good` and absence of debt are linear constraints. Logistic Regression fits these linear coefficients directly and avoids overfitting, whereas tree-based classifiers and boosting models like XGBoost can overfit the synthetic dataset boundaries.
* The complete pipeline (preprocessor + model) is saved as `model.pkl`.

---

## 8. Flask Web Application
A production Flask app (`app.py`) serves the saved model pipeline. 
* A modern **glassmorphic UI** form collects all inputs.
* Inputs are validated against boundary conditions.
* Features are engineered on-the-fly and passed through `model.pkl` to render clear outputs: **Approved** (Green) or **Rejected** (Red) with probability scores.

---

## 9. Conclusion & Future Scope
The automated model successfully predicts credit card approvals. Future enhancements include:
1. Incorporating external bureau reports (FICO scores).
2. Implementing bias detection to ensure fair lending across gender/age.
3. Running model monitoring to watch for data and concept drift.

---

## 10. References
1. Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR, 2011.
2. Chen & Guestrin, *XGBoost: A Scalable Tree Boosting System*, KDD, 2016.
3. IBM Watson Machine Learning Developer Documentation, 2024.
"""
    with open('Credit_Card_Approval_Report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
    print("Generated Credit_Card_Approval_Report.md")

    # 7. README.md
    readme_content = """# Credit Card Approval Prediction System

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

---

## Author
* **Academic Internship Project**
* Built with Python, Scikit-Learn, and Flask.
"""
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print("Generated README.md")

    # 8. Programmatic Jupyter Notebook Generation
    notebook_dict = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Credit Card Approval Prediction using Machine Learning\n",
                    "### Academic Internship Project Jupyter Notebook\n",
                    "\n",
                    "This notebook implements an end-to-end Machine Learning pipeline to predict whether a credit card application should be **Approved** or **Rejected** based on applicant profiles. It covers data loading, cleaning, exploratory data analysis, feature engineering, preprocessing, training four different algorithms, evaluation, model comparison, and model saving."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 1: Data loading & Descriptive Statistics\n",
                    "We read the Excel dataset, check its shape, inspect data types, view sample rows, check missing values, and remove duplicates."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "\n",
                    "# Load the Excel file\n",
                    "df = pd.read_excel('credit_card_applicant_data.xlsx')\n",
                    "print(f\"Dataset Shape: {df.shape}\")\n",
                    "print(f\"Columns: {list(df.columns)}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# First 10 rows\n",
                    "df.head(10)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Last 10 rows\n",
                    "df.tail(10)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Check missing values and duplicates\n",
                    "print(\"Missing Values:\\n\", df.isnull().sum())\n",
                    "print(f\"\\nDuplicate Records: {df.duplicated().sum()}\")\n",
                    "\n",
                    "# Drop duplicate records if any exist\n",
                    "df = df.drop_duplicates()\n",
                    "print(f\"Shape after duplicate removal: {df.shape}\")"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Descriptive Statistics\n",
                    "df.describe(include='all')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Identify Numerical and Categorical Columns\n",
                    "numerical_cols = list(df.select_dtypes(include=['int64', 'float64']).columns)\n",
                    "categorical_cols = list(df.select_dtypes(include=['object']).columns)\n",
                    "target_col = 'Approval_Status'\n",
                    "\n",
                    "print(f\"Numerical columns: {numerical_cols}\")\n",
                    "print(f\"Categorical columns: {categorical_cols}\")\n",
                    "print(f\"Target Column: {target_col}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 2: Exploratory Data Analysis (EDA)\n",
                    "We visualize the distributions of the target column, income, credit history, and employment duration using SeaBorn and Matplotlib."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "sns.set_theme(style=\"whitegrid\")\n",
                    "\n",
                    "# 1. Approval Distribution\n",
                    "plt.figure(figsize=(6,4))\n",
                    "sns.countplot(x='Approval_Status', data=df, palette='Set2')\n",
                    "plt.title('Credit Card Approval Status Distribution')\n",
                    "plt.show()\n",
                    "\n",
                    "# 2. Income Distribution\n",
                    "plt.figure(figsize=(8,5))\n",
                    "sns.histplot(x='Annual_Income', hue='Approval_Status', data=df, kde=True, multiple='stack')\n",
                    "plt.title('Annual Income Distribution by Approval Status')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 3. Credit History Distribution\n",
                    "plt.figure(figsize=(6,4))\n",
                    "sns.countplot(x='Credit_History', hue='Approval_Status', data=df, palette='viridis')\n",
                    "plt.title('Approval Status by Credit History')\n",
                    "plt.show()\n",
                    "\n",
                    "# 4. Employment Duration Analysis\n",
                    "plt.figure(figsize=(7,5))\n",
                    "sns.boxplot(x='Approval_Status', y='Employment_Duration_Years', data=df, palette='pastel')\n",
                    "plt.title('Employment Duration vs Approval Status')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# 5. Correlation Heatmap\n",
                    "plt.figure(figsize=(10,6))\n",
                    "num_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Applicant_ID'], errors='ignore')\n",
                    "sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')\n",
                    "plt.title('Correlation Matrix of Numerical Features')\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 3: Feature Engineering\n",
                    "We create helper columns: `Income_Group` (Low, Medium, High), `Employment_Duration_Category` (Short-term, Medium-term, Long-term), `Inquiries_to_Income_Ratio`, and numeric `Payment_History_Grade`."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Bin income\n",
                    "df['Income_Group'] = pd.cut(df['Annual_Income'], bins=[0, 50000, 110000, np.inf], labels=['Low', 'Medium', 'High'])\n",
                    "\n",
                    "# Bin employment years\n",
                    "df['Employment_Duration_Category'] = pd.cut(df['Employment_Duration_Years'], bins=[-1, 2, 7, np.inf], labels=['Short-term', 'Medium-term', 'Long-term'])\n",
                    "\n",
                    "# Inquiries-to-income ratio\n",
                    "df['Inquiries_to_Income_Ratio'] = df['Credit_Inquiries_Last_6M'] / (df['Annual_Income'] / 10000.0 + 1)\n",
                    "\n",
                    "# Map payment history to numerical grade\n",
                    "pay_history_map = {'No Debt': 0, 'Late Payments': 1, 'Serious Default': 2}\n",
                    "df['Payment_History_Grade'] = df['Payment_History_Status'].map(pay_history_map)\n",
                    "\n",
                    "print(df[['Income_Group', 'Employment_Duration_Category', 'Inquiries_to_Income_Ratio', 'Payment_History_Grade']].head())"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 4: Data Preprocessing\n",
                    "We separate the data into feature matrix `X` and target vector `y`, split them 80/20, and build a preprocessing pipeline using `ColumnTransformer` (scaling numericals and encoding categoricals)."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.pipeline import Pipeline\n",
                    "from sklearn.impute import SimpleImputer\n",
                    "\n",
                    "# Split features and target\n",
                    "X = df.drop(columns=['Applicant_ID', 'Approval_Status'])\n",
                    "y = df['Approval_Status'].map({'Approved': 1, 'Rejected': 0})\n",
                    "\n",
                    "# Numeric and categorical features list\n",
                    "num_features = ['Age', 'Number_of_Children', 'Annual_Income', 'Employment_Duration_Years', \n",
                    "                'Credit_Inquiries_Last_6M', 'Family_Size', 'Inquiries_to_Income_Ratio', 'Payment_History_Grade']\n",
                    "cat_features = ['Gender', 'Own_Car', 'Own_Property', 'Income_Type', 'Education_Level', \n",
                    "                'Marital_Status', 'Housing_Type', 'Has_Work_Phone', 'Has_Phone', 'Has_Email', \n",
                    "                'Credit_History', 'Income_Group', 'Employment_Duration_Category']\n",
                    "\n",
                    "# Train/Test Split\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)\n",
                    "\n",
                    "# Transformers\n",
                    "num_transformer = Pipeline(steps=[\n",
                    "    ('imputer', SimpleImputer(strategy='median')),\n",
                    "    ('scaler', StandardScaler())\n",
                    "])\n",
                    "\n",
                    "cat_transformer = Pipeline(steps=[\n",
                    "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
                    "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
                    "])\n",
                    "\n",
                    "preprocessor = ColumnTransformer(transformers=[\n",
                    "    ('num', num_transformer, num_features),\n",
                    "    ('cat', cat_transformer, cat_features)\n",
                    "])\n",
                    "\n",
                    "print(\"Preprocessing pipeline created successfully.\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 5 & 6: Model Building & Evaluation\n",
                    "We train Logistic Regression, Decision Tree, Random Forest, and XGBoost Classifiers, then evaluate them on the test set using standard validation metrics."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.tree import DecisionTreeClassifier\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report\n",
                    "from xgboost import XGBClassifier\n",
                    "import time\n",
                    "\n",
                    "models = {\n",
                    "    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),\n",
                    "    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),\n",
                    "    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),\n",
                    "    'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=5, eval_metric='logloss')\n",
                    "}\n",
                    "\n",
                    "comparison_results = []\n",
                    "trained_pipelines = {}\n",
                    "\n",
                    "for name, model in models.items():\n",
                    "    pipe = Pipeline(steps=[\n",
                    "        ('preprocessor', preprocessor),\n",
                    "        ('classifier', model)\n",
                    "    ])\n",
                    "    \n",
                    "    t0 = time.time()\n",
                    "    pipe.fit(X_train, y_train)\n",
                    "    train_time = time.time() - t0\n",
                    "    \n",
                    "    t0 = time.time()\n",
                    "    y_pred = pipe.predict(X_test)\n",
                    "    pred_time = time.time() - t0\n",
                    "    \n",
                    "    y_proba = pipe.predict_proba(X_test)[:, 1]\n",
                    "    \n",
                    "    acc = accuracy_score(y_test, y_pred)\n",
                    "    prec = precision_score(y_test, y_pred)\n",
                    "    rec = recall_score(y_test, y_pred)\n",
                    "    f1 = f1_score(y_test, y_pred)\n",
                    "    auc = roc_auc_score(y_test, y_proba)\n",
                    "    \n",
                    "    trained_pipelines[name] = pipe\n",
                    "    comparison_results.append({\n",
                    "        'Model': name,\n",
                    "        'Accuracy': acc,\n",
                    "        'Precision': prec,\n",
                    "        'Recall': rec,\n",
                    "        'F1': f1,\n",
                    "        'ROC-AUC': auc,\n",
                    "        'Train_Time': train_time,\n",
                    "        'Prediction_Time': pred_time\n",
                    "    })\n",
                    "\n",
                    "comp_df = pd.DataFrame(comparison_results)\n",
                    "print(comp_df)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Show classification report for the best model\n",
                    "best_model_name = comp_df.loc[comp_df['F1'].idxmax(), 'Model']\n",
                    "best_pipe = trained_pipelines[best_model_name]\n",
                    "y_pred = best_pipe.predict(X_test)\n",
                    "\n",
                    "print(f\"\\nClassification Report for Best Model: {best_model_name}\\n\")\n",
                    "print(classification_report(y_test, y_pred))\n",
                    "\n",
                    "print(\"\\nConfusion Matrix:\")\n",
                    "print(confusion_matrix(y_test, y_pred))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Section 7: Save Model\n",
                    "We save the best performing model pipeline as `model.pkl` using joblib so that it can be loaded for predictions in the Flask application."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import joblib\n",
                    "\n",
                    "# Save model pipeline\n",
                    "joblib.dump(best_pipe, 'model.pkl')\n",
                    "print(f\"Saved {best_model_name} pipeline to 'model.pkl'\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open('Credit_Card_Approval_Notebook.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_dict, f, indent=4)
    print("Generated Credit_Card_Approval_Notebook.ipynb")
    print("All project source files and documents generated successfully.")

if __name__ == '__main__':
    generate_all()
