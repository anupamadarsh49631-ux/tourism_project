import streamlit as st
import pandas as pd
import joblib
import os
from huggingface_hub import hf_hub_download

HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
MODEL_REPO = f"{HF_USERNAME}/tourism-package-model"


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.joblib",
        repo_type="model",
    )
    return joblib.load(model_path)


model = load_model()

st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package - Purchase Prediction")
st.write(
    "Enter a customer's details below to predict how likely they are to "
    "purchase the newly launched Wellness Tourism Package."
)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeofcontact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    citytier = st.selectbox("City Tier", [1, 2, 3])
    durationofpitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=60, value=10)
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    numpersons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    numfollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

with col2:
    preferredpropertystar = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    numtrips = st.number_input("Avg. Number of Trips per Year", min_value=0, max_value=20, value=2)
    passport = st.selectbox("Holds Passport", [0, 1])
    pitchsatisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])
    owncar = st.selectbox("Owns a Car", [0, 1])
    numchildren = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthlyincome = st.number_input("Monthly Income", min_value=1000, max_value=100000, value=20000)

if st.button("Predict"):
    input_df = pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": typeofcontact,
                "CityTier": citytier,
                "DurationOfPitch": durationofpitch,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": numpersons,
                "NumberOfFollowups": numfollowups,
                "ProductPitched": productpitched,
                "PreferredPropertyStar": preferredpropertystar,
                "MaritalStatus": maritalstatus,
                "NumberOfTrips": numtrips,
                "Passport": passport,
                "PitchSatisfactionScore": pitchsatisfaction,
                "OwnCar": owncar,
                "NumberOfChildrenVisiting": numchildren,
                "Designation": designation,
                "MonthlyIncome": monthlyincome,
            }
        ]
    )

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ This customer is **LIKELY** to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"❌ This customer is **UNLIKELY** to purchase the Wellness Package (probability: {probability:.1%})")
