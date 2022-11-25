# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 16:14:49 2022

@author: ROY
"""

# 필요 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt


# 데이터 읽어오기
airline = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/train.csv")


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
    
    
# 그리드 함수 정의 - Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def lr_tuning(train_set, test_set, parameters):
    model = LogisticRegression(random_state=100)
    grid = GridSearchCV(model, parameters, scoring="roc_auc", cv=5)
    grid.fit(train_set, test_set)
    return grid.best_params_, grid.best_score_

LogisticRegression
# 그리드 함수 정의 - Decision Tree
from sklearn.tree import DecisionTreeClassifier

def dt_tuning(train_set, test_set, parameters):
    model = DecisionTreeClassifier(random_state=100)
    grid = GridSearchCV(model, parameters, scoring="roc_auc", cv=5)
    grid.fit(train_set, test_set)
    return grid.best_params_, grid.best_score_


# 그리드 함수 정의 - KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

def knn_tuning(train_set, test_set, parameters):
    model = KNeighborsClassifier()
    scaler = StandardScaler()
    grid = GridSearchCV(model, parameters, scoring="roc_auc", cv=5)
    grid.fit(scaler.fit_transform(train_set), test_set)
    return grid.best_params_, grid.best_score_


# 로지스틱 회귀 그리드 서치
lr_param1 = {"C":[0.01, 0.1, 1, 5, 10], "max_iter":[5000, 10000]}

lr_tuning(X, y, lr_param1)

# C=0.01, max_iter=5000 일 때 최적 (한 번 더 돌리기)
lr_param2 = {"C":[0.005, 0.01, 0.015, 0.05], "max_iter":[3000, 4000, 5000, 7000]}

lr_tuning(X, y, lr_param2)

# C=0.005, max_iter=3000일 때 최선이기에 한 번 더 돌려보자
lr_param3 = {"C":[0.001, 0.003, 0.005, 0.007], "max_iter":[1000, 2000, 3000, 4000]}

lr_tuning(X, y, lr_param3)

# C=0.003, max_iter=1000 일 때 최적 값 (한 번 더)
lr_param4 = {"C":[0.003], "max_iter":[300, 500, 1000]}

lr_tuning(X, y, lr_param4)
# max_iter가 300이하 일 때 수가 너무 작아서 warning, 300보다 크면 수렴하여 똑같은 roc_auc score를 보임

# 최적의 하이퍼 파라미터를 통해 로지스틱 회귀 학습 및 평가
lr = LogisticRegression(C=0.003, max_iter=300, random_state=100)
lr.fit(X, y)
pred_lr = lr.predict(airline_test_X)
evaluation(airline_test, pred_lr)
# 학습셋으로 진행할 때보다 줄어든 모습
# 기본 모델로 했을 때보다 정확도, 정밀도에서 약간 상승하고 재현율은 약간 떨어짐. f1 score와 roc_auc score는 거의 그대로


# 결정트리 그리드 서치
dt_param = {"max_depth":[5, 20, 100, 500, 1000], "min_samples_split":[5, 10, 50, 100], 
            "min_samples_leaf":[5, 10, 50, 100]}

dt_tuning(X, y, dt_param)

# max_depth=20, min_samples_leaf=5, min_samples_split=100 (한 번 더 돌리기)
dt_param2 = {"max_depth":[10, 20, 30, 50], "min_samples_split":[80, 100, 300, 500], 
            "min_samples_leaf":[3, 5, 7]}

dt_tuning(X, y, dt_param2)
# 한 번 더 돌렸음에도 max_depth=20, min_samples_leaf=5, min_samples_split=100


# 최적의 하이퍼 파라미터를 통한 결정 트리 학습 및 평가
dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=100, random_state=100)
dt.fit(X, y)
pred_dt = dt.predict(airline_test_X)
evaluation(airline_test, pred_dt)
# 기본 모델로 학습 했을 때보다 재현율을 제외하고 모든 지표에서 0.5% ~ 1% 정도 높은 모습을 보임.
# 물론 학습셋으로만 했을 때보다는 당연히 줄어듦


# KNN 그리드 서치
knn_param = {"n_neighbors":[5, 10, 20, 50], "p":[1, 2]}

knn_tuning(X, y, knn_param)

# 이웃 수 20, 맨해튼 거리 계산일 때 최적 (한 번 더 돌리기)
knn_param2 = {"n_neighbors":[15, 20, 25, 40], "p":[1]}

knn_tuning(X, y, knn_param2)

# 이웃 수 25일 때 최적(마지막 한 번)
knn_param3 = {"n_neighbors":[25, 30, 35], "p":[1]}

knn_tuning(X, y, knn_param3)
# 이웃 수 25, 거리 계산 법=맨해튼일 때 최적


# 최적의 하이퍼 파라미터를 통한 KNN 학습 및 평가
scaler = StandardScaler()

knn = KNeighborsClassifier(n_neighbors=25, p=1)
knn.fit(scaler.fit_transform(X), y)
pred_knn = knn.predict(scaler.fit_transform(airline_test_X))
evaluation(airline_test, pred_knn)
# 마찬가지로 학습셋으로만 그리드 서치를 했을 때 보다는 당연히 줄어든 성능을 보임.
# 기본 모델에 비해 정밀도를 제외한 모든 평가 지표에서 0.3% ~ 1% 정도 높은 모습을 보임.