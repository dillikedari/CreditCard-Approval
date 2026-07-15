import os
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
