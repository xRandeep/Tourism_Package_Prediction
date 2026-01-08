import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Configuration
HF_USERNAME = "iStillWaters"  # HuggingFace Profile
MODEL_REPO = f"{HF_USERNAME}/tourism-prediction-model"

st.set_page_config(page_title="Tourism Prediction AI", layout="centered")

@st.cache_resource
def load_assets():
    # Download model and encoders from Hugging Face
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="model.joblib")
    encoders_path = hf_hub_download(repo_id=MODEL_REPO, filename="encoders.joblib")
    
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    return model, encoders

st.title("Tourism Package Prediction")
st.markdown("Will a customer purchase the **Wellness Package**? Enter details below.")

try:
    model, encoders = load_assets()
    
    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 18, 90, 30)
            income = st.number_input("Monthly Income", 1000, 100000, 20000)
            pitch_dur = st.slider("Pitch Duration (min)", 5, 120, 15)
            gender = st.selectbox("Gender", encoders['Gender'].classes_)
            marital = st.selectbox("Marital Status", encoders['MaritalStatus'].classes_)
        
        with col2:
            contact = st.selectbox("Contact Type", encoders['TypeofContact'].classes_)
            occupation = st.selectbox("Occupation", encoders['Occupation'].classes_)
            product = st.selectbox("Product Pitched", encoders['ProductPitched'].classes_)
            designation = st.selectbox("Designation", encoders['Designation'].classes_)
            passport = st.selectbox("Has Passport?", [0, 1])

        submit = st.form_submit_button("Predict Probability")
        
if submit:
            # 1. Create DataFrame with user inputs (Order doesn't matter here yet)
            input_data = pd.DataFrame({
                'Age': [age], 
                'MonthlyIncome': [income], 
                'DurationOfPitch': [pitch_dur],
                'Gender': [gender], 
                'MaritalStatus': [marital], 
                'TypeofContact': [contact],
                'Occupation': [occupation], 
                'ProductPitched': [product], 
                'Designation': [designation],
                'Passport': [passport],
                # Defaults for columns not in form
                'CityTier': [1], 
                'NumberOfPersonVisiting': [3], 
                'NumberOfFollowups': [3],
                'PreferredPropertyStar': [3], 
                'NumberOfTrips': [3], 
                'PitchSatisfactionScore': [3],
                'OwnCar': [1], 
                'NumberOfChildrenVisiting': [1] 
            })
            
            # 2. Encode categorical columns
            for col, le in encoders.items():
                if col in input_data.columns:
                    input_data[col] = le.transform(input_data[col])
            
            # --- THE FIX: Reorder columns to match training data exactly ---
            expected_order = [
                'Age', 'TypeofContact', 'CityTier', 'DurationOfPitch', 'Occupation', 
                'Gender', 'NumberOfPersonVisiting', 'NumberOfFollowups', 'ProductPitched', 
                'PreferredPropertyStar', 'MaritalStatus', 'NumberOfTrips', 'Passport', 
                'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 
                'Designation', 'MonthlyIncome'
            ]
            input_data = input_data[expected_order]
            # ---------------------------------------------------------------
            
            # 3. Predict
            pred = model.predict(input_data)[0]
            prob = model.predict_proba(input_data)[0][1]
            
            if pred == 1:
                st.success(f"✅ Likely to Purchase! (Confidence: {prob:.1%})")
            else:
                st.error(f"❌ Unlikely to Purchase. (Confidence: {prob:.1%})")
				
except Exception as e:
    st.error(f"Error loading model: {e}")
