import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Set number of samples
num_samples = 1500

# Generate columns
applicant_id = np.arange(10001, 10001 + num_samples)

gender = np.random.choice(['Male', 'Female'], size=num_samples, p=[0.48, 0.52])
own_car = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.35, 0.65])
own_property = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.68, 0.32])
children = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.60, 0.22, 0.12, 0.05, 0.01])

# Income & Age
annual_income = np.random.normal(loc=75000, scale=35000, size=num_samples).astype(int)
# Cap minimum income at 15000
annual_income = np.clip(annual_income, 15000, 350000)

age = np.random.randint(21, 68, size=num_samples)

# Income Type, Education, Marital Status, Housing
income_type = np.random.choice(
    ['Salaried', 'Self-Employed', 'Pensioner', 'Student'], 
    size=num_samples, 
    p=[0.65, 0.18, 0.12, 0.05]
)

education_level = np.random.choice(
    ['High School', 'Graduate', 'Post-Graduate'], 
    size=num_samples, 
    p=[0.35, 0.50, 0.15]
)

marital_status = np.random.choice(
    ['Married', 'Single', 'Divorced', 'Widowed'], 
    size=num_samples, 
    p=[0.60, 0.25, 0.10, 0.05]
)

housing_type = np.random.choice(
    ['Owned House', 'Rented Apartment', 'With Parents', 'Municipal House'], 
    size=num_samples, 
    p=[0.70, 0.18, 0.08, 0.04]
)

# Employment Duration: depends on age and income type
employment_years = np.zeros(num_samples)
for i in range(num_samples):
    if income_type[i] == 'Student':
        employment_years[i] = np.random.uniform(0, 2)
    elif income_type[i] == 'Pensioner':
        employment_years[i] = 0
    else:
        max_work_years = age[i] - 18
        if max_work_years > 0:
            employment_years[i] = np.random.uniform(0, max_work_years)
        else:
            employment_years[i] = 0

employment_years = np.round(employment_years, 1)

# Contact info
work_phone = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.25, 0.75])
phone = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.78, 0.22])
email = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.30, 0.70])

# Credit columns
credit_history = np.random.choice(['Good', 'Bad'], size=num_samples, p=[0.75, 0.25])
credit_inquiries = np.random.choice([0, 1, 2, 3, 4, 5], size=num_samples, p=[0.55, 0.22, 0.13, 0.06, 0.03, 0.01])
payment_history = np.random.choice(
    ['No Debt', 'Late Payments', 'Serious Default'], 
    size=num_samples, 
    p=[0.68, 0.22, 0.10]
)

# Family size (approximate correlation with children and marital status)
family_size = np.zeros(num_samples, dtype=int)
for i in range(num_samples):
    base = 2 if marital_status[i] == 'Married' else 1
    family_size[i] = base + children[i]

# Determine Approval target variable (Approved/Rejected) based on rules + noise
# Make it sound like a complex pattern
noise = np.random.normal(loc=0, scale=0.15, size=num_samples)

# Scored logic: credit history is key (weight 0.45)
# income (weight 0.20)
# employment years (weight 0.15)
# payment history (weight 0.15)
# credit inquiries (weight -0.15)
income_score = (annual_income - 15000) / (350000 - 15000)
employment_score = employment_years / 40.0
inquiries_score = credit_inquiries / 5.0

score = (
    0.45 * (credit_history == 'Good') +
    0.20 * income_score +
    0.15 * employment_score -
    0.15 * inquiries_score +
    0.15 * (payment_history == 'No Debt') +
    0.05 * (education_level != 'High School') +
    noise
)

# Set approval threshold
approval_status = np.where(score > 0.35, 'Approved', 'Rejected')

# Verify the approval distribution
print("Approval Distribution:")
print(pd.Series(approval_status).value_counts(normalize=True))

# Introduce some deliberate missing values to test preprocessing (around 2-5% missing)
# Missing values in annual_income, employment_years, credit_history
income_missing_idx = np.random.choice(num_samples, size=int(0.03 * num_samples), replace=False)
employment_missing_idx = np.random.choice(num_samples, size=int(0.04 * num_samples), replace=False)
credit_history_missing_idx = np.random.choice(num_samples, size=int(0.02 * num_samples), replace=False)

annual_income_float = annual_income.astype(float)
employment_years_float = employment_years.copy()
credit_history_obj = credit_history.astype(object)

annual_income_float[income_missing_idx] = np.nan
employment_years_float[employment_missing_idx] = np.nan
credit_history_obj[credit_history_missing_idx] = np.nan

# Construct DataFrame
df = pd.DataFrame({
    'Applicant_ID': applicant_id,
    'Gender': gender,
    'Age': age,
    'Own_Car': own_car,
    'Own_Property': own_property,
    'Number_of_Children': children,
    'Annual_Income': annual_income_float,
    'Income_Type': income_type,
    'Education_Level': education_level,
    'Marital_Status': marital_status,
    'Housing_Type': housing_type,
    'Employment_Duration_Years': employment_years_float,
    'Has_Work_Phone': work_phone,
    'Has_Phone': phone,
    'Has_Email': email,
    'Credit_History': credit_history_obj,
    'Credit_Inquiries_Last_6M': credit_inquiries,
    'Payment_History_Status': payment_history,
    'Family_Size': family_size,
    'Approval_Status': approval_status
})

# Save to Excel
output_path = 'credit_card_applicant_data.xlsx'
df.to_excel(output_path, index=False)
print(f"Dataset generated and saved successfully to: {output_path}")
print(f"Dataset shape: {df.shape}")
