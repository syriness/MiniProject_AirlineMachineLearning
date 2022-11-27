# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:10:33 2022

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


# Personal Data로 데이터프레임 구성 (개인 정보로 군집학습, 범주형 데이터를 구분하기 위해서)
airline_personal = airline[["satisfaction", "Gender", "Customer Type", 
                          "Age", "Type of Travel", "Class", "Flight Distance"]]


# Personal Data 레이블 인코딩 (범주형 데이터를 연속형 변수로 바꿈)
from sklearn.preprocessing import LabelEncoder

temp_list = ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]
encoder = LabelEncoder()
for i in temp_list:
    encoder.fit(airline_personal[i])
    airline_personal[i] = encoder.transform(airline_personal[i])
    

# 시험셋도 미리 불러오고 인코딩
airline2 = pd.read_csv("https://raw.githubusercontent.com/syriness/MiniProject_AirlineMachineLearning/main/test.csv")

airline2.isnull().sum()
airline2.dropna(inplace=True)

airline_test = airline2["satisfaction"]
airline_test_X = airline2.iloc[:, 8:24]

airline_personal_test = airline2[["satisfaction", "Gender", "Customer Type", 
                          "Age", "Type of Travel", "Class", "Flight Distance"]]

temp_list2 = ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]
encoder2 = LabelEncoder()
for i in temp_list2:
    encoder2.fit(airline_personal_test[i])
    airline_personal_test[i] = encoder2.transform(airline_personal_test[i])

encoder3 = LabelEncoder()
encoder3.fit(airline_test)
airline_test = encoder3.transform(airline_test)

airline_test_X.info()
airline_test_X.astype(int)


# Personal Data 정규화 스케일링 (KMeans가 거리 기반 모델이기 때문에 필요함)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler2 = StandardScaler()
airline_personal_scaling = scaler.fit_transform(airline_personal)
airline_personal_test_scaling = scaler2.fit_transform(airline_personal_test)


# 군집을 몇 개로 해야 하는지 파악하기 위해 엘보우 기법 사용
from sklearn.cluster import KMeans

sse=[]
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=100, max_iter=1000).fit(airline_personal_scaling)
    sse.append(kmeans.inertia_)

f, ax = plt.subplots(figsize=(18, 12))
ax.plot(range(1, 11), sse, marker="o") # 아주 분명하지는 않지만 3에서 꺾이는 모습


# 실루엣 스코어
from sklearn.metrics import silhouette_score

def no_clusters_silhouette(cluster_list, X_features):
    for n_cluster in cluster_list:
        clusters = KMeans(n_clusters=n_cluster, init="k-means++", random_state=100, max_iter=1000)
        cluster_labels = clusters.fit_predict(X_features)
        sil_avg = silhouette_score(X_features, cluster_labels)
        print("Number of Cluster: " + str(n_cluster) + " Silhouette Score: " + str(round(sil_avg, 4)))
no_clusters_silhouette([2, 3, 4, 5, 6, 7, 8, 9, 10], airline_personal_scaling)
# 3개일 때 가장 크기 때문에 K값을 3으로 결정


# K-Means
airline_personal_kmeans = airline_personal.copy()

kmeans = KMeans(n_clusters=3, init="k-means++", random_state=100, max_iter=1000)
kmeans.fit(airline_personal_scaling)
airline_personal_kmeans["KMeans"] = kmeans.predict(airline_personal_scaling)

kmeans_result = airline_personal_kmeans.groupby(["KMeans"]).mean().sort_values(by="satisfaction").reset_index()


# 그룹 당 비율
group_size = airline_personal_kmeans[["KMeans", "Gender"]].groupby(["KMeans"]).count().rename(columns={"Gender":"Group_Size"})
group_size["Group_Proportion"] = group_size["Group_Size"] / group_size["Group_Size"].sum()

f, ax = plt.subplots(figsize=(18, 12))
ax.pie(group_size["Group_Proportion"], 
       labels=["Dissatisfied Personal Traveler(Loyal)", "Dissatisfied Business Traveler(Disloyal)", 
               "Satisfied Business Traveler(Loyal)"], 
       autopct="%.2f%%", textprops={"size":"20", "color":"k"}, explode=(0.1, 0.1, 0.1), 
       colors=["lightcoral", "aquamarine", "royalblue"])
plt.title("Group Size Proportion", fontsize=26, fontweight="bold")
plt.legend(loc="best", prop={"size":16}, bbox_to_anchor=(1.5, -0.1, 0.2, 1.0), title="Group", title_fontsize=16)
plt.show()


# 시험셋에서 fit한 모델로 테스트셋 군집화 해보고 그래프 그리기
airline_personal_test_kmeans = airline_personal_test.copy()

airline_personal_test_kmeans["KMeans"] = kmeans.predict(airline_personal_test_scaling)

kmeans_result_test = airline_personal_test_kmeans.groupby(["KMeans"]).mean().sort_values(by="satisfaction").reset_index()

group_size_test = airline_personal_test_kmeans[["KMeans", 
                                                "Gender"]].groupby(["KMeans"]).count().rename(columns={"Gender":"Group_Size"})
group_size_test["Group_Proportion"] = group_size_test["Group_Size"] / group_size_test["Group_Size"].sum()

f, ax = plt.subplots(figsize=(18, 12))
ax.pie(group_size_test["Group_Proportion"], 
       labels=["Dissatisfied Personal Traveler(Loyal)", "Dissatisfied Business Traveler(Disloyal)", 
               "Satisfied Business Traveler(Loyal)"], 
       autopct="%.2f%%", textprops={"size":"20", "color":"k"}, explode=(0.1, 0.1, 0.1), 
       colors=["lightcoral", "aquamarine", "royalblue"])
plt.title("Group Size Proportion(test set)", fontsize=26, fontweight="bold")
plt.legend(loc="best", prop={"size":16}, bbox_to_anchor=(1.5, -0.1, 0.2, 1.0), title="Group", title_fontsize=16)
plt.show()
# 시험셋에 적용해도 거의 비슷한 비율을 가진다는 것을 확인할 수 있음


# 고객 평가지표 데이터프레임
airline_score = airline.iloc[:, 8:24]


# 학습 전 데이터 전처리
airline_score.info()
airline_score.astype(int) # 컬럼 하나가 실수형이라 정수형으로 바꿔줌
airline_score["satisfaction"] = airline_personal_kmeans["satisfaction"]


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
    
    
# 그리드 탐색을 통해 최적의 하이퍼 파라미터 조합을 찾음.
# 해당 조합으로 모델학습 및 평가와 중요한 컬럼들 확인.

# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

scaler4 = StandardScaler()
lr = LogisticRegression(C=0.003, max_iter=300, random_state=100)
lr.fit(scaler4.fit_transform(X), y)
pred_lr = lr.predict(scaler4.fit_transform(airline_test_X))
evaluation(airline_test, pred_lr)
# accuracy: 0.8143, precision: 0.8013, recall: 0.7670, fl score: 0.7838, roc_auc_score: 0.8091

lr_coef_list = lr.coef_
lr_imp = pd.DataFrame(lr_coef_list)
lr_imp.columns = X.columns
lr_imp_T = lr_imp.T
lr_imp_T.columns = ['satisfaction']
lr_imp_T["abs"] = lr_imp_T["satisfaction"].abs()
lr_imp_T.sort_values(by="abs", ascending=False, inplace=True)

plt.figure(figsize=(20, 10))
sns.lineplot(data=lr_imp_T["satisfaction"].head(8))
# 온라인 보딩, 레그룸, 이착륙 시 편안함, 기내 와이파이, 기내 영상, 기내 장비, 체크인 서비스가 중요 컬럼

plt.figure(figsize=(20, 10))
sns.lineplot(data=lr_imp_T["satisfaction"].tail(8))


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


# 랜덤 포레스트
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 350, max_depth = 15, min_samples_split = 10, min_samples_leaf= 5)
rf.fit(X, y)
pred_rf = rf.predict(airline_test_X)
evaluation(airline_test, pred_rf)
# accuracy: 0.9410, precision: 0.9542, recall: 0.9092, fl score: 0.9312, roc_auc_score: 0.9375

rf.imp = pd.DataFrame({"satisfaction":X.columns, "values":rf.feature_importances_})

plt.figure(figsize=(20,10))
sns.barplot(x="values", y= "satisfaction", data=rf.imp.sort_values(by="values", ascending=False))


# XGBoost
import xgboost as xgb

xg = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3, subsample=1)
xg.fit(X, y)
pred_xg = xg.predict(airline_test_X)
evaluation(airline_test, pred_xg)
# accuracy: 0.9479, precision: 0.9517, recall: 0.9286, fl score: 0.9400, roc_auc_score: 0.9458

import shap # shap value: 실제값과 예측값의 차이를 설명하여 각 컬럼과 결과 간의 인과관계를 추론

column_imp_xg = shap.TreeExplainer(xg)
shap_values_xg = column_imp_xg.shap_values(airline_test_X)

shap.summary_plot(shap_values_xg, airline_test_X, plot_type="bar")

xg.imp = pd.DataFrame({'features': X.columns, 'values': xg.feature_importances_})

plt.figure(figsize=(20, 10))
sns.barplot(x='values', y='features', data=xg.imp.sort_values(by='values', ascending=False))


# LightGBM
import lightgbm as lgb

gbm = lgb.LGBMClassifier(max_depth = 20, n_estimators = 1000, subsample = 0.2, learning_rate = 0.07)
gbm.fit(X, y)
pred_gbm = gbm.predict(airline_test_X)
evaluation(airline_test, pred_gbm)
# accuracy: 0.9498, precision: 0.9572, recall: 0.9271, fl score: 0.9419, roc_auc_score: 0.9473

column_imp_gbm = shap.TreeExplainer(gbm)
shap_values_gbm = column_imp_gbm.shap_values(airline_test_X)

shap.summary_plot(shap_values_gbm, airline_test_X, plot_type="bar")

# 영향을 많이 주는 변수
gbm.imp = pd.DataFrame({'features': X.columns, 'values': gbm.feature_importances_})

plt.figure(figsize=(20, 10))
sns.barplot(x='values', y='features', data=gbm.imp.sort_values(by='values', ascending=False))


# KNN
from sklearn.neighbors import KNeighborsClassifier

scaler5 = StandardScaler()

knn = KNeighborsClassifier(n_neighbors=25, p=1)
knn.fit(scaler5.fit_transform(X), y)
pred_knn = knn.predict(scaler5.fit_transform(airline_test_X))
evaluation(airline_test, pred_knn)
# accuracy: 0.9199, precision: 0.9260, recall: 0.8886, fl score: 0.9069, roc_auc_score: 0.9165