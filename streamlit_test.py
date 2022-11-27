# 필요 라이브러리 import
import pandas as pd
import matplotilb.pyplot as plt
import streamlit as st
import kmeans
import prepro
import eval


# streamlit 앱 제목
st.title("항공사 고객 만족도 Machine Learning")


# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

st.header("데이터 확인")
st.table(airline.head(10))

st.write("")
st.write("")
st.write("")

airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")
airline2.dropna(inplace=True)

airline.dropna(inplace=True)
    
kmeans.kmeans_clustering(airline, airline2)   
    
X, y, airline_test, airline_test_X = prepro.preprocess(airline, airline2)
    
# 결정트리
from sklearn.tree import DecisionTreeClassifier

st.header("결정트리 학습 결과")

dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=100, random_state=100)
dt.fit(X, y)
pred_dt = dt.predict(airline_test_X)
eval.evaluation(airline_test, pred_dt)
# accuracy: 0.9324, precision: 0.9373, recall: 0.9066, fl score: 0.9217, roc_auc_score: 0.9296

st.write("")
st.write("")

st.subheader("결정트리 학습 시 중요한 변수들")

dt.imp = pd.DataFrame({"satisfaction":X.columns, "values":dt.feature_importances_})

fig_dt = plt.figure(figsize=(20, 10))
sns.barplot(x="values", y= "satisfaction", data=dt.imp.sort_values(by="values", ascending=False))
st.pyplot(fig_dt)
