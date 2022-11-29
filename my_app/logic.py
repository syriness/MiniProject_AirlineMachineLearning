from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import eval
from sklearn.linear_model import LogisticRegression

def logic_reg(X, y, airline_test_X, airline_test):
    st.header("로지스틱 회귀 학습 결과")
    
    scaler4 = StandardScaler()
    lr = LogisticRegression(C=0.003, max_iter=300, random_state=100)
    lr.fit(scaler4.fit_transform(X), y)
    pred_lr = lr.predict(scaler4.fit_transform(airline_test_X))
    eval.evaluation(airline_test, pred_lr)
    
    lr_coef_list = lr.coef_
    lr_imp = pd.DataFrame(lr_coef_list)
    lr_imp.columns = X.columns
    lr_imp_T = lr_imp.T
    lr_imp_T.columns = ['satisfaction']
    lr_imp_T["abs"] = lr_imp_T["satisfaction"].abs()
    lr_imp_T.sort_values(by="abs", ascending=False, inplace=True)
    
    st.write("")
    st.subheader("상관관계 상위 8개 컬럼")
    
    fig1 = plt.figure(figsize=(20, 10))
    sns.lineplot(data=lr_imp_T["satisfaction"].head(8))
    st.pyplot(fig1)
    
    st.write("")
    st.subheader("상관관계 하위 8개 컬럼")
    
    fig2 = plt.figure(figsize=(20, 10))
    sns.lineplot(data=lr_imp_T["satisfaction"].tail(8))
    st.pyplot(fig2)
