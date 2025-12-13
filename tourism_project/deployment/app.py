import streamlit as st
import pandas as pd
import json
import joblib
from huggingface_hub import hf_hub_download

# ============================
# LOAD MODEL + THRESHOLD
# ============================
model_path = hf_hub_download(
    repo_id="sudipgandhi/tourism-package-prediction-model",
    filename="best_model.joblib"
)

threshold_path = hf_hub_download(
    repo_id="sudipgandhi/tourism-package-prediction-model",
    filename="best_threshold.json"
)

model = joblib.load(model_path)

with open(threshold_path) as f:
    threshold_data = json.load(f)

THRESHOLD = threshold_data["threshold"]

# ============================
# UI
# ============================
st.title("Tourism Package Purchase Prediction")

st.info(f"Decision Threshold: **{THRESHOLD:.2f}**")

# ============================
# INPUTS
# ============================
input_df = pd.DataFrame([{
    "Age": st.number_input("Age", 18, 100, 35),
    "TypeofContact": st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"]),
    "CityTier": st.selectbox("City Tier", [1, 2, 3]),
    "DurationOfPitch": st.number_input("Duration Of Pitch", 1, 120, 35),
    "Occupation": st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"]),
    "Gender": st.selectbox("Gender", ["Male", "Female"]),
    "NumberOfPersonVisiting": st.number_input("Persons Visiting", 1, 10, 2),
    "NumberOfFollowups": st.number_input("Follow-ups", 0, 10, 4),
    "ProductPitched": st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe"]),
    "PreferredPropertyStar": st.selectbox("Property Star", [3, 4, 5]),
    "MaritalStatus": st.selectbox("Marital Status", ["Single", "Married", "Divorced"]),
    "NumberOfTrips": st.number_input("Trips per Year", 0, 50, 3),
    "Passport": st.selectbox("Passport", [0, 1]),
    "PitchSatisfactionScore": st.selectbox("Pitch Satisfaction", [1, 2, 3, 4, 5]),
    "OwnCar": st.selectbox("Own Car", [0, 1]),
    "NumberOfChildrenVisiting": st.number_input("Children Visiting", 0, 5, 1),
    "Designation": st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]),
    "MonthlyIncome": st.number_input("Monthly Income", 10000, 500000, 350000),
}])

# ============================
# PREDICTION
# ============================
if st.button("Predict"):
    prob = model.predict_proba(input_df)[0][1]

    st.write(f"### Predicted Probability: **{prob:.4f}**")

    if prob >= THRESHOLD:
        st.success("Customer is **LIKELY** to purchase the Tourism Package.")
    else:
        st.error("Customer is **UNLIKELY** to purchase the Tourism Package.")
