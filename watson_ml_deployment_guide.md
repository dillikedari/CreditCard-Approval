# IBM Watson Machine Learning (WML) Deployment Guide

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
