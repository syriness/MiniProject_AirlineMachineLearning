import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import eval
from sklearn.tree import DecisionTreeClassifier

def decision_(X, y, airline_test_X, airline_test):
    st.header("결정트리 학습 결과")
    
    dt = DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=100, random_state=100)
    dt.fit(X, y)
    pred_dt = dt.predict(airline_test_X)
    eval.evaluation(airline_test, pred_dt)
    
    dt.imp = pd.DataFrame({"satisfaction":X.columns, "values":dt.feature_importances_})
    
    st.write("")
    st.subheader("결정트리의 feature importance")

    fig = plt.figure(figsize=(20, 10))
    sns.barplot(x="values", y= "satisfaction", data=dt.imp.sort_values(by="values", ascending=False))
    st.pyplot(fig)
