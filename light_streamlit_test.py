import streamlit as st
import joblib
from PIL import Image

st.title("TEST")

evaluation_list = joblib.load("lightgbm_evaluation.pkl")

st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[0]), unsafe_allow_html=True)
st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[1]), unsafe_allow_html=True)
st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[2]), unsafe_allow_html=True)
st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[3]), unsafe_allow_html=True)
st.markdown('<span style="color: LightPink; font-size:120%">**정확도:**</span> <span style="color: SeaGreen; font-size:110%">{0:.4f}</span>'.format(evaluation_list[4]), unsafe_allow_html=True)

image1 = Image.open("lightgbm_confusion_matrix.png")
image2 = Image.open("lightgbm_shap.png")
image3 = Image.open("lightgbm_featureimportance.png")

st.image(image1)
st.image(image2)
st.image(image3)
