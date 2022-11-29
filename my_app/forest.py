import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import eval
from sklearn.ensemble import RandomForestClassifier

def random_(X, y, airline_test_X, airline_test):
    st.header("랜덤 포레스트 학습 결과")
    
    rf = RandomForestClassifier(n_estimators = 350, max_depth = 15, min_samples_split = 10, min_samples_leaf= 5)
    rf.fit(X, y)
    pred_rf = rf.predict(airline_test_X)
    eval.evaluation(airline_test, pred_rf)
    
    rf.imp = pd.DataFrame({"satisfaction":X.columns, "values":rf.feature_importances_})
    
    st.write("")
    st.subheader("랜덤 포레스트의 feature importance")

    fig = plt.figure(figsize=(20,10))
    sns.barplot(x="values", y= "satisfaction", data=rf.imp.sort_values(by="values", ascending=False))
    st.pyplot(fig)
