import os
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def run_pipeline():
    print("--- STEP 1: LOADING & ANALYZING DATA ---")
    # Read Excel dataset
    df = pd.read_excel('credit_card_applicant_data.xlsx')
    
    # 1. Shape
    shape = df.shape
    print(f"Dataset Shape: {shape}")
    
    # 2. Columns
    columns = list(df.columns)
    print(f"Columns: {columns}")
    
    # 3. Data types
    dtypes = df.dtypes
    print("Data Types:\n", dtypes)
    
    # 4. Check missing values
    missing_vals = df.isnull().sum()
    print("Missing Values:\n", missing_vals)
    
    # 5. Check duplicates
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate records: {duplicate_count}")
    
    # 6. Remove duplicates
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print("Duplicates removed.")
        
    # 7. Identify target and feature columns
    # In our generated dataset, target is 'Approval_Status'
    target_col = 'Approval_Status'
    print(f"Target Column Identified: {target_col}")
    
    # Identify numerical and categorical columns
    # Exclude ID and target
    feature_cols = [c for c in df.columns if c not in ['Applicant_ID', target_col]]
    
    # --- STEP 2: EXPLORATORY DATA ANALYSIS (EDA) ---
    print("\n--- STEP 2: GENERATING EDA VISUALIZATIONS ---")
    os.makedirs('static', exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    # 1. Approval Distribution
    plt.figure()
    sns.countplot(x=target_col, data=df, palette='Set2')
    plt.title('Credit Card Approval Status Distribution')
    plt.xlabel('Approval Status')
    plt.ylabel('Count')
    plt.savefig('static/approval_dist.png', bbox_inches='tight')
    plt.close()
    
    # 2. Income Distribution
    plt.figure()
    sns.histplot(x='Annual_Income', hue=target_col, data=df, kde=True, multiple='stack', palette='muted')
    plt.title('Annual Income Distribution by Approval Status')
    plt.xlabel('Annual Income ($)')
    plt.savefig('static/income_dist.png', bbox_inches='tight')
    plt.close()
    
    # 3. Credit History Distribution
    plt.figure()
    sns.countplot(x='Credit_History', hue=target_col, data=df, palette='viridis')
    plt.title('Approval Status by Credit History')
    plt.xlabel('Credit History')
    plt.savefig('static/credit_history_dist.png', bbox_inches='tight')
    plt.close()
    
    # 4. Employment Analysis
    plt.figure()
    sns.boxplot(x=target_col, y='Employment_Duration_Years', data=df, palette='pastel')
    plt.title('Employment Duration vs Approval Status')
    plt.xlabel('Approval Status')
    plt.ylabel('Employment Duration (Years)')
    plt.savefig('static/employment_analysis.png', bbox_inches='tight')
    plt.close()
    
    # 5. Correlation Heatmap (only numerical features)
    plt.figure()
    numerical_df = df.select_dtypes(include=['int32', 'int64', 'float32', 'float64'])
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('static/correlation_heatmap.png', bbox_inches='tight')
    plt.close()
    
    # --- STEP 3: FEATURE ENGINEERING ---
    print("\n--- STEP 3: FEATURE ENGINEERING ---")
    # 1. Income Group
    def bin_income(income):
        if pd.isna(income):
            return np.nan
        if income < 50000:
            return 'Low'
        elif income <= 110000:
            return 'Medium'
        else:
            return 'High'
            
    df['Income_Group'] = df['Annual_Income'].apply(bin_income)
    
    # 2. Employment Duration Category
    def bin_employment(years):
        if pd.isna(years):
            return np.nan
        if years < 2:
            return 'Short-term'
        elif years <= 7:
            return 'Medium-term'
        else:
            return 'Long-term'
            
    df['Employment_Duration_Category'] = df['Employment_Duration_Years'].apply(bin_employment)
    
    # 3. Inquiries to Income Ratio
    df['Inquiries_to_Income_Ratio'] = df['Credit_Inquiries_Last_6M'] / (df['Annual_Income'] / 10000.0 + 1)
    
    # 4. Payment History Grade (numeric)
    pay_history_map = {'No Debt': 0, 'Late Payments': 1, 'Serious Default': 2}
    df['Payment_History_Grade'] = df['Payment_History_Status'].map(pay_history_map)
    
    print("Features engineered successfully.")
    
    # --- STEP 4: DATA PREPROCESSING ---
    print("\n--- STEP 4: DATA PREPROCESSING ---")
    
    # Define features and target
    X = df.drop(columns=['Applicant_ID', target_col])
    # Encode target variable: Approved = 1, Rejected = 0
    y = df[target_col].map({'Approved': 1, 'Rejected': 0})
    
    # Separate numeric and categorical features
    numeric_features = ['Age', 'Number_of_Children', 'Annual_Income', 'Employment_Duration_Years', 
                        'Credit_Inquiries_Last_6M', 'Family_Size', 'Inquiries_to_Income_Ratio', 'Payment_History_Grade']
    categorical_features = ['Gender', 'Own_Car', 'Own_Property', 'Income_Type', 'Education_Level', 
                            'Marital_Status', 'Housing_Type', 'Has_Work_Phone', 'Has_Phone', 'Has_Email', 
                            'Credit_History', 'Income_Group', 'Employment_Duration_Category']
    
    # Split into train and test sets (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")
    
    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # --- STEP 5 & 6: MODEL BUILDING & EVALUATION ---
    print("\n--- STEP 5 & 6: MODEL BUILDING & EVALUATION ---")
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=6),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
        'XGBoost': XGBClassifier(random_state=42, n_estimators=100, max_depth=5, eval_metric='logloss')
    }
    
    comparison_results = []
    trained_pipelines = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        # Measure training time
        start_train = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Measure prediction time
        start_pred = time.time()
        y_pred = pipeline.predict(X_test)
        pred_time = time.time() - start_pred
        
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        trained_pipelines[name] = pipeline
        comparison_results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'ROC-AUC': auc,
            'Train_Time_Sec': train_time,
            'Pred_Time_Sec': pred_time,
            'Confusion_Matrix': cm.tolist(),
            'Classification_Report': report
        })
        
    # --- STEP 7: MODEL COMPARISON ---
    print("\n--- STEP 7: MODEL COMPARISON ---")
    comp_df = pd.DataFrame(comparison_results)
    print(comp_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']])
    
    # Select best model based on F1 Score
    best_model_idx = comp_df['F1'].idxmax()
    best_model_name = comp_df.loc[best_model_idx, 'Model']
    best_model_pipeline = trained_pipelines[best_model_name]
    print(f"\nBest Model Selected: {best_model_name} with F1-Score: {comp_df.loc[best_model_idx, 'F1']:.4f}")
    
    # Save the best model
    print("\n--- STEP 8: SAVING MODEL ---")
    # Save model pipeline
    joblib.dump(best_model_pipeline, 'model.pkl')
    print("Model pipeline saved successfully to 'model.pkl'.")
    
    # Plot Comparison Graph
    plt.figure(figsize=(10, 6))
    comp_melt = comp_df.melt(id_vars='Model', value_vars=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    sns.barplot(x='Model', y='value', hue='variable', data=comp_melt, palette='Set1')
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Model')
    plt.ylabel('Score')
    plt.ylim(0.5, 1.0)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('static/model_comparison.png', bbox_inches='tight')
    plt.close()
    
    # Generate metadata for Flask application
    metadata = {
        'best_model': best_model_name,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features,
        'columns': list(X.columns)
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=4)
        
    return comp_df, best_model_name

if __name__ == '__main__':
    run_pipeline()
