# 필요 라이브러리 import
import pandas as pd
# import matplotlib.pyplot as plt
import streamlit as st


# streamlit 앱 제목
st.title("항공사 고객 만족도 Machine Learning")


# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

st.dataframe(airline)
