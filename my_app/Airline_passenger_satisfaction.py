# Contents of ~/my_app/Airline_passenger_satisfaction.py
import streamlit as st

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
import knn
import joblib
from PIL import Image
import xgboost as xgb
import shap

image = 'https://image.cnbcfm.com/api/v1/image/107124573-1664221269888-gettyimages-463523885-1[…]d06fa636af6.jpeg?v=1668095693&w=740&h=416&ffmt=webp&vtcrop=y'
st.image(image)

# streamlit 앱 제목
st.markdown('**<center><span style="color: #000000; font-size:250%; ">Airline passenger satisfaction  Machine Learning Data view</span></center>**', unsafe_allow_html=True)
st.markdown('---')
# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")
st.markdown('<center><span style="color: #666666; font-size:100%;">raw data: 원 데이터셋에 약 10만개의 데이터가 있으며, 훈련셋에는 약 26,000개의 데이터가 있다.</span></center>', unsafe_allow_html=True)
#st.table(airline.head(10))
st.dataframe(data=airline.head(10), width=None, height=None,  use_container_width=False)


# st.write("")
# st.write("")
# st.write("")

# airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")
# airline2.dropna(inplace=True)

# airline.dropna(inplace=True)
    
# kmeans.kmeans_clustering(airline, airline2)   
    
# X, y, airline_test_X, airline_test = prepro.preprocess(airline, airline2)
    
# st.write("")
# st.write("")
# st.write("")

# logic.logic_reg(X, y, airline_test_X, airline_test)

# st.write("")
# st.write("")
# st.write("")

# tree.decision_(X, y, airline_test_X, airline_test)

# st.write("")
# st.write("")
# st.write("")

# forest.random_(X, y, airline_test_X, airline_test)

# st.write("")
# st.write("")
# st.write("")

# xg.xg_ensemble(X, y, airline_test_X, airline_test)

# st.write("")
# st.write("")
# st.write("")

# st.header("LightGBM 학습 결과")

# evaluation_list = joblib.load("lightgbm_evaluation.pkl")

# st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[0]), unsafe_allow_html=True)
# st.markdown('<span style="color: PaleVioletRed; font-size:120%">**정밀도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[1]), unsafe_allow_html=True)
# st.markdown('<span style="color: LightBlue; font-size:120%">**재현도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[2]), unsafe_allow_html=True)
# st.markdown('<span style="color: PaleTurquoise; font-size:120%">**f1 score:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[3]), unsafe_allow_html=True)
# st.markdown('<span style="color: DeepSkyBlue; font-size:120%">**roc_auc_score:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[4]), unsafe_allow_html=True)

# image1 = Image.open("lightgbm_confusion_matrix.png")
# image2 = Image.open("lightgbm_shap.png")
# image3 = Image.open("lightgbm_featureimportance.png")

# st.image(image1)

# st.write("")
# st.subheader("LightGBM Shap Value")
# st.image(image2)

# st.write("")
# st.subheader("LightGBM의 feature importance")
# st.image(image3)

# st.write("")
# st.write("")
# st.write("")

# knn.neighbors(X, y, airline_test_X, airline_test)

# st.write("")
# st.write("")
# st.write("")

# #st.markdown('**<center><span style="color: MidnightBlue; font-size:250%">Thank You!</span></center>**', unsafe_allow_html=True)
# #page_names_to_funcs = {
# #    "Practice": Practice,
# #    "APP": APP,
# #}

# #selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# #page_names_to_funcs[selected_page]()
# # Contents of ~/my_app/main_page.py


# #def Practice():
# #   st.markdown("# Practice")
# #   st.sidebar.markdown("# Page 2 ❄️")

# #page_names_to_funcs = {
# #    "Practice": Practice,
# #}

# #selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
# #page_names_to_funcs[selected_page]()
