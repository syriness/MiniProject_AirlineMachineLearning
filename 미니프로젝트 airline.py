# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 18:21:29 2022

@author: ROY
"""

# 필요 라이브러리 import
import pandas as pd
import matplotlib.pyplot as plt


# 데이터 읽어오기
airline = pd.read_csv("train.csv")


# null값 확인 후 제거
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
airline2 = pd.read_csv("test.csv")

airline2.isnull().sum()
airline2.dropna(inplace=True)

airline_test = airline2["satisfaction"]
airline_test_X = airline2.iloc[:, 8:24]

encoder2 = LabelEncoder()
encoder2.fit(airline_test)
airline_test = encoder2.transform(airline_test)

airline_test_X.info()
airline_test_X.astype(int)


# Personal Data 정규화 스케일링 (KMeans가 거리 기반 모델이기 때문에 필요함)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
airline_personal_scaling = scaler.fit_transform(airline_personal)


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


# 고객 평가지표 데이터프레임
airline_score = airline.iloc[:, 8:24]
airline_score["Group"] = airline_personal_kmeans["KMeans"]
    

# 학습 전 데이터 전처리
airline_score.info()
airline_score.astype(int) # 컬럼 하나가 실수형이라 정수형으로 바꿔줌
airline_score["satisfaction"] = airline_personal_kmeans["satisfaction"]


# X, y 정의
X=airline_score.drop(["Group", "satisfaction"], axis=1)
y=airline_score["satisfaction"]


# 평가지표 함수 정의
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import roc_auc_score

def evaluation(airline_test, pred):
    acc = accuracy_score(airline_test, pred)
    pre = precision_score(airline_test, pred)
    rec = recall_score(airline_test, pred)
    roc = roc_auc_score(airline_test, pred)
    print("정확도: {0:.4f}".format(acc))
    print("정밀도: {0:.4f}".format(pre))
    print("재현율: {0:.4f}".format(rec))
    print("roc_auc_score: {0:.4f}".format(roc))

    
# 로지스틱 회귀
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1000)
lr.fit(X, y)
pred_lr = lr.predict(airline_test_X)
evaluation(airline_test, pred_lr)


# 결정트리
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(X, y)
pred_dt = dt.predict(airline_test_X)
evaluation(airline_test, pred_dt)


# 랜덤포레스트
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X, y)
pred_rf = rf.predict(airline_test_X)
evaluation(airline_test, pred_rf)


# XGBoost
import xgboost as xgb

xg = xgb.XGBClassifier()
xg.fit(X, y)
pred_xg = xg.predict(airline_test_X)
evaluation(airline_test, pred_xg)


# LightGBM
import lightgbm as lgb

gbm = lgb.LGBMClassifier()
gbm.fit(X, y)
pred_gbm = gbm.predict(airline_test_X)
evaluation(airline_test, pred_gbm)


# KNN
from sklearn.neighbors import KNeighborsClassifier

knn = knn = KNeighborsClassifier()
knn.fit(scaler.fit_transform(X), y)
pred_knn = knn.predict(scaler.fit_transform(airline_test_X))
evaluation(airline_test, pred_knn)


"""여기까지는 기본 모델이고, 향후 전처리 과정에서의 피쳐 엔지니어링, 모델 학습 과정에서의 그리드 서치나
    반복문 사용 등을 통해 더 정확하고 엄밀한 모델 평가를 해야할 필요성이 있음.
    또한, 점수 말고 다른 항목들이 만족도에 미치는 영향을 분석해봐도 좋을 것 같음.
    (미니프로젝트 이후) 좀 더 발전시켜서, 만족도에 큰 영향을 주는 서비스 항목 등을 분석하여 향후 개선 전략 등을
    수립하는 등의 포트폴리오를 만들 수 있겠음."""