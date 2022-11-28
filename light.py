import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import eval
# import shap
import lightgbm as lgb

def light_(X, y, airline_test_X, airline_test):
    st.header("LightGBM 학습 결과")
    
    gbm = lgb.LGBMClassifier(max_depth = 20, n_estimators = 1000, subsample = 0.2, learning_rate = 0.07)
    gbm.fit(X, y)
    pred_gbm = gbm.predict(airline_test_X)
    eval.evaluation(airline_test, pred_gbm)
    
    # st.write("")
    # st.subheader("LightGBM Shap Value")
    
    # column_imp_gbm = shap.TreeExplainer(gbm)
    # shap_values_gbm = column_imp_gbm.shap_values(airline_test_X)
    
    # fig = plt.figure(figsize=(15, 10))
    # shap.summary_plot(shap_values_gbm, airline_test_X, plot_type="bar")
    # st.pyplot(fig)
    
    st.write("")
    st.subheader("LightGBM의 feature importance")
    
    xg.imp = pd.DataFrame({'features': X.columns, 'values': xg.feature_importances_})

    fig2 = plt.figure(figsize=(20, 10))
    sns.barplot(x='values', y='features', data=xg.imp.sort_values(by='values', ascending=False))
    st.pyplot(fig2)
