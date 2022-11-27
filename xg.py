import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import eval
import xgboost as xgb
import shap

def xg_ensemble(X, y, airline_test_X, airline_test):
    st.header("XGBoost 학습결과")
    
    xg = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3, subsample=1)
    xg.fit(X, y)
    pred_xg = xg.predict(airline_test_X)
    eval.evaluation(airline_test, pred_xg)
    
    st.write("")
    st.subheader("XGBoost Shap Value")
    st.write("Shap Value란 실제값과 예측값의 차이를 설명하여 각 컬럼과 결과 간의 인과관계를 추론하는 것으로,")
    st.write("단순 feature importance는 너무 많은 가중치가 적용되기에 종속변수와 독립변수 간의 관계를 보는데는 shap value가 보다 적합하다.")
    
    column_imp_xg = shap.TreeExplainer(xg)
    shap_values_xg = column_imp_xg.shap_values(airline_test_X)

    fig = shap.summary_plot(shap_values_xg, airline_test_X, plot_type="bar")
    st.pyplot(fig)
    
    st.write("")
    st.subheader("XGBoost의 feature importance")
    
    xg.imp = pd.DataFrame({'features': X.columns, 'values': xg.feature_importances_})

    fig2 = plt.figure(figsize=(20, 10))
    sns.barplot(x='values', y='features', data=xg.imp.sort_values(by='values', ascending=False))
    st.pyplot(fig2)
