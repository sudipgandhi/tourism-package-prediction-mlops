"""
STREAMLIT APPLICATION
---------------------
Loads trained model and threshold from Hugging Face,
accepts user input,
and predicts likelihood of package purchase.
"""

import json
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

# ============================
# LOAD MODEL & THRESHOLD
# ============================
MODEL_REPO_ID = "sudipgandhi/tourism-package-prediction-model"

model_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename="best_model.joblib"
)

threshold_path = hf_hub_download(
    repo_id=MODEL_REPO_ID,
    filename="best_threshold.json"
)

model = joblib.load(model_path)

with open(threshold_path) as f:
    THRESHOLD = json.load(f)["threshold"]

# ============================
# UI
# ============================
st.title("Tourism Package Purchase Prediction")

# ============================
# INPUTS
# ============================
Age = st.number_input("Age", 18, 100, 35)
CityTier = st.selectbox("City Tier", [1, 2, 3])
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
DurationOfPitch = st.number_input("Duration of Pitch", 1, 120, 35)
Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Persons Visiting", 1, 10, 2)
NumberOfFollowups = st.number_input("Followups", 0, 10, 4)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number of Trips", 0, 50, 3)
Passport = st.selectbox("Passport", [0, 1])
PitchSatisfactionScore = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
OwnCar = st.selectbox("Own Car", [0, 1])
NumberOfChildrenVisiting = st.number_input("Children Visiting", 0, 10, 1)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income", 10000, 500000, 350000)

# ============================
# INPUT DATAFRAME
# ============================
input_df = pd.DataFrame([{
    "Age": Age,
    "CityTier": CityTier,
    "TypeofContact": TypeofContact,
    "DurationOfPitch": DurationOfPitch,
    "Occupation": Occupation,
    "Gender": Gender,
    "NumberOfPersonVisiting": NumberOfPersonVisiting,
    "NumberOfFollowups": NumberOfFollowups,
    "ProductPitched": ProductPitched,
    "PreferredPropertyStar": PreferredPropertyStar,
    "MaritalStatus": MaritalStatus,
    "NumberOfTrips": NumberOfTrips,
    "Passport": Passport,
    "PitchSatisfactionScore": PitchSatisfactionScore,
    "OwnCar": OwnCar,
    "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
    "Designation": Designation,
    "MonthlyIncome": MonthlyIncome
}])

# ============================
# PREDICTION
# ============================
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]
    if prob >= THRESHOLD:
        st.success("Customer is LIKELY to purchase the Tourism Package.")
    else:
        st.error("Customer is UNLIKELY to purchase the Tourism Package.")
