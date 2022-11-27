import pandas as pd
import streamlit as st
import eval
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def neighbors(X, y, airline_test_X, airline_test):
    st.header("KNN 학습 결과")
    
    scaler5 = StandardScaler()
    
    knn = KNeighborsClassifier(n_neighbors=25, p=1)
    knn.fit(scaler5.fit_transform(X), y)
    pred_knn = knn.predict(scaler5.fit_transform(airline_test_X))
    eval.evaluation(airline_test, pred_knn)
