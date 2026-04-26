import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Load model
OUT = Path('output')
model_data = joblib.load(OUT / 'padew_thesis_model.joblib')
base_pipe = model_data['base_pipe']
rf_anom = model_data['rf_anom']
threshold = model_data['threshold']
feature_cols = model_data['feature_cols']

st.set_page_config(page_title="ICU Deterioration Predictor", page_icon="🏥")

st.title("🏥 ICU Patient Deterioration Prediction")
st.markdown("### PADEW Early Warning System")

# Input form
st.sidebar.header("Patient Features")

def get_user_input():
    return {
        'age': st.sidebar.slider('Age', 18, 95, 65),
        'sex_male': st.sidebar.radio('Sex', ['Female', 'Male']) == 'Male',
        'charlson_index': st.sidebar.slider('Charlson Index', 0, 15, 3),
        'heart_rate': st.sidebar.slider('Heart Rate (bpm)', 50, 180, 80),
        'systolic_bp': st.sidebar.slider('Systolic BP (mmHg)', 70, 200, 130),
        'diastolic_bp': st.sidebar.slider('Diastolic BP (mmHg)', 40, 120, 75),
        'map': st.sidebar.slider('Mean Arterial Pressure', 50, 140, 90),
        'spo2': st.sidebar.slider('SpO2 (%)', 80, 100, 97),
        'resp_rate': st.sidebar.slider('Respiratory Rate', 8, 40, 16),
        'temperature': st.sidebar.slider('Temperature (°C)', 35.0, 40.0, 37.0),
        'lactate': st.sidebar.slider('Lactate (mmol/L)', 0.5, 15.0, 1.5),
        'creatinine': st.sidebar.slider('Creatinine (mg/dL)', 0.4, 8.0, 0.9),
        'wbc': st.sidebar.slider('WBC (x10³/µL)', 2, 30, 8),
        'hemoglobin': st.sidebar.slider('Hemoglobin (g/dL)', 5, 18, 12),
        'potassium': st.sidebar.slider('Potassium (mEq/L)', 2.5, 6.5, 4.2),
        'sodium': st.sidebar.slider('Sodium (mEq/L)', 125, 155, 140),
        'ventilator': st.sidebar.checkbox('On Ventilator'),
        'vasopressor': st.sidebar.checkbox('On Vasopressor'),
        'fluid_balance_ml_24h': st.sidebar.slider('Fluid Balance (mL/24h)', -2000, 4000, 500),
        'trend_hr': st.sidebar.slider('HR Trend', -20, 30, 0),
        'trend_rr': st.sidebar.slider('RR Trend', -10, 20, 0),
        'trend_map': st.sidebar.slider('MAP Trend', -30, 20, 0),
        'trend_spo2': st.sidebar.slider('SpO2 Trend', -10, 5, 0),
        'missing_rate': st.sidebar.slider('Missing Data Rate', 0.0, 0.6, 0.1),
    }

# Get input
input_data = get_user_input()

# Create DataFrame
X = pd.DataFrame([input_data])[feature_cols]

# Make prediction
prob = base_pipe.predict_proba(X)[0, 1]
anom = rf_anom.predict_proba(X)[0, 1]
final_score = 0.6 * prob + 0.4 * anom
alert = final_score >= threshold

# Display results
st.divider()
st.subheader("Prediction Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Logistic Regression Prob", f"{prob:.1%}")

with col2:
    st.metric("Anomaly Score", f"{anom:.1%}")

with col3:
    st.metric("Final Score", f"{final_score:.1%}")

# Alert status
if alert:
    st.error(f"⚠️ ALERT: High risk of deterioration! (Threshold: {threshold:.2f})")
else:
    st.success(f"✅ No alert (Threshold: {threshold:.2f})")

# Explanation
st.divider()
st.subheader("Feature Contributions")

# Get coefficients
coef = base_pipe.named_steps['model'].coef_[0]
scaler = base_pipe.named_steps['scaler']
imputer = base_pipe.named_steps['imputer']

X_imp = imputer.transform(X)
X_scaled = scaler.transform(X_imp)
contributions = X_scaled[0] * coef

contrib_df = pd.DataFrame({
    'feature': feature_cols,
    'contribution': contributions
}).sort_values('contribution', key=abs, ascending=False)

# Show top 10
st.dataframe(
    contrib_df.head(10).style.format({'contribution': '{:.3f}'}),
    use_container_width=True
)

# Info
st.divider()
st.caption("Model: Logistic Regression + Random Forest ensemble | Threshold tuned for clinical utility")