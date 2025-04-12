import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

feature_names = [
    "Gender", "Tumor Size", "Wavelet-LHH_glrlm_LowGrayLevelRunEmphasis",
    "Wavelet-LLH_glcm_Correlation", "Wavelet-LHH_gldm_DependenceNonUniformityNormalized",
    "Log-sigma-4-0-mm-3D_firstorder_Kurtosis", "Log-sigma-3-0-mm-3D_firstorder_Maximum",
    "Wavelet-HHH_gldm_SmallDependenceLowGrayLevelEmphasis", "Wavelet-HLH_glrlm_RunEntropy",
    "Wavelet-HLH_glszm_ZoneEntropy", "Diagnostics_Mask-interpolated_VoxelNum",
    "Original_glrlm_LongRunLowGrayLevelEmphasis"
]

default_values = [
    # 0, 1.8, 0.0586096111126547, 0.152320654124126, 0.071395153856487, 3.585217627709164,
    # 61.57615661621094, 0.0094457048759033, 3.190494227725063, 3.918425987296181, 4233.0, 0.0516863372919313,

    0.0, 4.2, 82225.0, 0.0993999161932017,
    2.625075542712968, 53.39099884033203,
    0.0363411079070871, 0.0390479429623455, 0.0677708098186266,
    0.0043550137961862, 2.953906961344337, 2.499698143184412,
]

st.title("              ccRCC ISUP Grade Group Predictor               ")
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender (0=Female, 1=Male):", options=[0, 1])
    tumor_size = st.number_input("Tumor Size:", value=default_values[1])
    diagnostics_Mask_interpolated_VoxelNum = st.number_input("Diagnostics_Mask-interpolated_VoxelNum:", value=int(default_values[10]))
    original_glrlm_LongRunLowGrayLevelEmphasis = st.number_input("Original_glrlm_LongRunLowGrayLevelEmphasis:", value=default_values[11])
    log_sigma_4_0_mm_3D_firstorder_Kurtosis = st.number_input("Log-sigma-4-0-mm-3D_firstorder_Kurtosis:", value=default_values[5])
    log_sigma_3_0_mm_3D_firstorder_Maximum = st.number_input("Log-sigma-3-0-mm-3D_firstorder_Maximum:", value=default_values[6])
with col2:
    wavelet_LHH_glrlm_LowGrayLevelRunEmphasis = st.number_input("Wavelet-LHH_glrlm_LowGrayLevelRunEmphasis:", value=default_values[2])
    wavelet_LLH_glcm_Correlation = st.number_input("Wavelet-LLH_glcm_Correlation:", value=default_values[3])
    wavelet_LHH_gldm_DependenceNonUniformityNormalized = st.number_input("Wavelet-LHH_gldm_DependenceNonUniformityNormalized:", value=default_values[4])
    wavelet_HHH_gldm_SmallDependenceLowGrayLevelEmphasis = st.number_input("Wavelet-HHH_gldm_SmallDependenceLowGrayLevelEmphasis:", value=default_values[7])
    wavelet_HLH_glrlm_RunEntropy = st.number_input("Wavelet-HLH_glrlm_RunEntropy:", value=default_values[8])
    wavelet_HLH_glszm_ZoneEntropy = st.number_input("Wavelet-HLH_glszm_ZoneEntropy:", value=default_values[9])

feature_values = [
    gender, tumor_size,
    diagnostics_Mask_interpolated_VoxelNum,
    original_glrlm_LongRunLowGrayLevelEmphasis,
    log_sigma_4_0_mm_3D_firstorder_Kurtosis,
    log_sigma_3_0_mm_3D_firstorder_Maximum,

    wavelet_LHH_glrlm_LowGrayLevelRunEmphasis,
    wavelet_LLH_glcm_Correlation,
    wavelet_LHH_gldm_DependenceNonUniformityNormalized,
    wavelet_HHH_gldm_SmallDependenceLowGrayLevelEmphasis,
    wavelet_HLH_glrlm_RunEntropy,
    wavelet_HLH_glszm_ZoneEntropy,
]
features = np.array([feature_values])

# feature_values = default_values
# features = np.array([feature_values])


if st.button("Predict"):
    model = joblib.load('#lgb_op.pkl')
    # model = joblib.load('/mnt/c/Users/cc/PycharmProjects/test/#lgb_op.pkl')

    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]
    print(predicted_proba)

    st.write(f"**Predicted Class:** {predicted_class}")
    # st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = f"**The model predicts probability is {probability:.1f}%. The prediction is for +  >>  isupGG 3-4**"
    else:
        advice = f"**The model predicts probability is {probability:.1f}%. The prediction is for -  >>  isupGG 1-2**"
    st.write(advice)

    explainer = shap.TreeExplainer(model)
    # plt.figure(figsize=(15, 5), dpi=1200)
    # shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    # shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))
    shap.plots.waterfall(shap_values[-1], max_display=12, show=True)
    plt.savefig("shap.png", bbox_inches='tight', format='png')
    st.image("shap.png")
