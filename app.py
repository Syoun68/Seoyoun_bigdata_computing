
# ==================================================
# 다중 특성 회귀 모델 웹 서비스
# ==================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ==================================================
# 저장된 모델 불러오기
# ==================================================

linear_model = joblib.load("linear.pkl")
poly_model = joblib.load("poly.pkl")
ridge_model = joblib.load("ridge.pkl")

# ==================================================
# 웹 페이지 제목
# ==================================================

st.set_page_config(
    page_title="Life Expectancy Prediction",
    layout="wide"
)

st.title("🌎 기대수명 예측 웹 서비스")

st.write(
    "Adult mortality, BMI, GDP, Alcohol 정보를 이용하여 기대수명을 예측합니다."
)

# ==================================================
# 사이드바 입력
# ==================================================

st.sidebar.header("입력값 설정")

adult = st.sidebar.slider(
    "Adult mortality",
    0,
    800,
    150
)

bmi = st.sidebar.slider(
    "BMI",
    0.0,
    80.0,
    25.0
)

gdp = st.sidebar.slider(
    "GDP",
    0,
    120000,
    10000
)

alcohol = st.sidebar.slider(
    "Alcohol",
    0.0,
    20.0,
    5.0
)

# ==================================================
# 모델 선택
# ==================================================

model_name = st.selectbox(
    "모델 선택",
    ["Linear", "Poly", "Ridge"]
)

if model_name == "Linear":
    model = linear_model

elif model_name == "Poly":
    model = poly_model

else:
    model = ridge_model

# ==================================================
# 입력 데이터 생성
# ==================================================

input_data = pd.DataFrame({
    "Adult mortality": [adult],
    "BMI": [bmi],
    "GDP": [gdp],
    "Alcohol": [alcohol]
})

# ==================================================
# 예측 수행
# ==================================================

prediction = model.predict(input_data)

# ==================================================
# 예측 결과 출력
# ==================================================

st.header("예측 결과")

st.metric(
    label="예상 기대수명",
    value=f"{prediction[0]:.2f} 세"
)

result_df = pd.read_csv("result_df.csv")

# ==================================================
# 성능 비교 테이블
# ==================================================

st.subheader("모델 성능 비교")

st.dataframe(result_df)

# ==================================================
# Test R2 그래프
# ==================================================

fig, ax = plt.subplots()

ax.bar(
    result_df["Model"],
    result_df["Test R2"]
)

ax.set_ylabel("Test R2")

st.pyplot(fig)
