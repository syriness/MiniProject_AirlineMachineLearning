# 필요 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# streamlit 앱 제목
st.title("항공사 고객 만족도 Machine Learning")


# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")

st.dataframe(airline)


# 데이터 정보 확인
airline.info()


# null값 확인
airline.isnull().sum()


# 데이터 수에 비해 null값이 작기 때문에 그냥 제거함
airline.dropna(inplace=True)
    
    
# satisfaction 컬럼 인코딩
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder() 
encoder.fit(airline["satisfaction"])
airline["satisfaction"] = encoder.transform(airline["satisfaction"])


# 시험셋도 미리 불러오고 인코딩
airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")

airline2.isnull().sum()
airline2.dropna(inplace=True)

airline_test = airline2["satisfaction"]
airline_test_X = airline2.iloc[:, 8:24]

encoder2 = LabelEncoder()
encoder2.fit(airline_test)
airline_test = encoder2.transform(airline_test)

airline_test_X.info()
airline_test_X.astype(int)


# 고객 평가지표 데이터프레임
airline_score = airline.iloc[:, 8:24]


# 학습 전 데이터 전처리
airline_score.info()
airline_score.astype(int) # 컬럼 하나가 실수형이라 정수형으로 바꿔줌
airline_score["satisfaction"] = airline["satisfaction"]


# X, y 정의
X = airline_score.drop(["satisfaction"], axis=1)
y = airline_score["satisfaction"]


# 평가지표 함수 정의
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns

def evaluation(airline_test, pred):
    acc = accuracy_score(airline_test, pred)
    pre = precision_score(airline_test, pred)
    rec = recall_score(airline_test, pred)
    f1 = f1_score(airline_test, pred)
    roc = roc_auc_score(airline_test, pred)
    cf_matrix = confusion_matrix(airline_test, pred)
    print("정확도: {0:.4f}".format(acc))
    print("정밀도: {0:.4f}".format(pre))
    print("재현율: {0:.4f}".format(rec))
    print("f1 score: {0:.4f}".format(f1))
    print("roc_auc_score: {0:.4f}".format(roc))
    group_names = ['TN','FP','FN','TP']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='coolwarm')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()
    
    
# 결정트리
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=100, random_state=100)
dt.fit(X, y)
pred_dt = dt.predict(airline_test_X)
evaluation(airline_test, pred_dt)
# accuracy: 0.9324, precision: 0.9373, recall: 0.9066, fl score: 0.9217, roc_auc_score: 0.9296

dt.imp = pd.DataFrame({"satisfaction":X.columns, "values":dt.feature_importances_})

plt.figure(figsize=(20, 10))
sns.barplot(x="values", y= "satisfaction", data=dt.imp.sort_values(by="values", ascending=False))
