import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义特征名称
feature_names = [
    "Gender", "Tumor Size", "Wavelet-LHH_glrlm_LowGrayLevelRunEmphasis",
    "Wavelet-LLH_glcm_Correlation", "Wavelet-LHH_gldm_DependenceNonUniformityNormalized",
    "Log-sigma-4-0-mm-3D_firstorder_Kurtosis", "Log-sigma-3-0-mm-3D_firstorder_Maximum",
    "Wavelet-HHH_gldm_SmallDependenceLowGrayLevelEmphasis", "Wavelet-HLH_glrlm_RunEntropy",
    "Wavelet-HLH_glszm_ZoneEntropy", "Diagnostics_Mask-interpolated_VoxelNum",
    "Original_glrlm_LongRunLowGrayLevelEmphasis"
]

# 特征默认值
default_values = [
    0, 1.8, 0.0586096111126547, 0.152320654124126, 0.071395153856487,
    3.585217627709164, 61.57615661621094, 0.0094457048759033, 3.190494227725063,
    3.918425987296181, 4233.0, 0.0516863372919313
]

st.title("Heart Disease Predictor")
gender = st.selectbox("Gender (0=Female, 1=Male):", options=[0, 1])
tumor_size = st.number_input("Tumor Size:", value=default_values[1])
wavelet_LHH_glrlm_LowGrayLevelRunEmphasis = st.number_input("Wavelet-LHH_glrlm_LowGrayLevelRunEmphasis:", value=default_values[2])
wavelet_LLH_glcm_Correlation = st.number_input("Wavelet-LLH_glcm_Correlation:", value=default_values[3])
wavelet_LHH_gldm_DependenceNonUniformityNormalized = st.number_input("Wavelet-LHH_gldm_DependenceNonUniformityNormalized:", value=default_values[4])
log_sigma_4_0_mm_3D_firstorder_Kurtosis = st.number_input("Log-sigma-4-0-mm-3D_firstorder_Kurtosis:", value=default_values[5])
log_sigma_3_0_mm_3D_firstorder_Maximum = st.number_input("Log-sigma-3-0-mm-3D_firstorder_Maximum:", value=default_values[6])
wavelet_HHH_gldm_SmallDependenceLowGrayLevelEmphasis = st.number_input("Wavelet-HHH_gldm_SmallDependenceLowGrayLevelEmphasis:", value=default_values[7])
wavelet_HLH_glrlm_RunEntropy = st.number_input("Wavelet-HLH_glrlm_RunEntropy:", value=default_values[8])
wavelet_HLH_glszm_ZoneEntropy = st.number_input("Wavelet-HLH_glszm_ZoneEntropy:", value=default_values[9])
diagnostics_Mask_interpolated_VoxelNum = st.number_input("Diagnostics_Mask-interpolated_VoxelNum:", value=int(default_values[10]))
original_glrlm_LongRunLowGrayLevelEmphasis = st.number_input("Original_glrlm_LongRunLowGrayLevelEmphasis:", value=default_values[11])

# 处理输入并进行预测
feature_values = [
    gender, tumor_size, wavelet_LHH_glrlm_LowGrayLevelRunEmphasis, wavelet_LLH_glcm_Correlation,
    wavelet_LHH_gldm_DependenceNonUniformityNormalized, log_sigma_4_0_mm_3D_firstorder_Kurtosis,
    log_sigma_3_0_mm_3D_firstorder_Maximum, wavelet_HHH_gldm_SmallDependenceLowGrayLevelEmphasis,
    wavelet_HLH_glrlm_RunEntropy, wavelet_HLH_glszm_ZoneEntropy, diagnostics_Mask_interpolated_VoxelNum,
    original_glrlm_LongRunLowGrayLevelEmphasis
]

features = np.array([feature_values])

if st.button("Predict"):
    # 加载模型
    model = joblib.load('#lgb_op.pkl')

    # 进行预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = f"The model predicts probability is {probability:.1f}%. The prediction is for heart disease."
    else:
        advice = f"The model predicts probability is {probability:.1f}%. The prediction is for no heart disease."

    st.write(advice)

    # 使用 SHAP 解释预测
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(pd.DataFrame([feature_values], columns=feature_names))

    # 显示 SHAP 力导向图
    shap.plots.force(shap_values[0, ...], matplotlib=True, show=True, )
    # plt.close()
    # shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    # plt.close()
    st.image("shap_force_plot.png")
