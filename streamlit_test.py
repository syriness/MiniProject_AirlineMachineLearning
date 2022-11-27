# 필요 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import kmeans
import prepro
import logic
import tree
import forest
import xg
import light
import knn

# streamlit 앱 제목
st.title("항공사 고객 만족도 Machine Learning")

# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

st.header("데이터 확인")
st.table(airline.head(10))
st.write("원 데이터셋에 약 10만개의 데이터가 있으며, 훈련셋에는 약 26,000개의 데이터가 있다.")

st.write("")
st.write("")
st.write("")

airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")
airline2.dropna(inplace=True)

airline.dropna(inplace=True)
    
kmeans.kmeans_clustering(airline, airline2)   
    
X, y, airline_test_X, airline_test = prepro.preprocess(airline, airline2)
    
st.write("")
st.write("")
st.write("")

logic.logic_reg(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

tree.decision_(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

forest.random_(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

xg.xg_ensemble(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

light.light_(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

knn.neighbors(X, y, airline_test_X, airline_test)

st.write("")
st.write("")
st.write("")

st.markdown('<span style="color: MidnightBlue; font-size:250%"><center>**Thank You!**</center></span>', unsafe_allow_html=True)
