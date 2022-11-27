import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import prepro
import light

# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")
airline2.dropna(inplace=True)

airline.dropna(inplace=True)

st.title("항공사 고객 만족도 LightGBM")
st.write("메모리 문제로 인해 LightGBM만 따로 진행했습니다.")

st.write("")
st.write("")

X, y, airline_test_X, airline_test = prepro.preprocess(airline, airline2)

light.light_(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

st.markdown('**<center><span style="color: MidnightBlue; font-size:250%">Thank You!</span></center>**', unsafe_allow_html=True)
