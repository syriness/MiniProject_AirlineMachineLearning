# Contents of ~/my_app/streamlit_app.py
import streamlit as st

#def main_page():
#    st.markdown("# Practice")
#    st.sidebar.markdown("# Practice")
import pandas as pd
import numpy as np
import streamlit as st
import joblib

image = 'https://www.sisajournal.com/news/photo/201907/188115_92491_371.jpg'
st.image(image)

st.markdown('**<center><span style="color: #000000; font-size:250%; line-height:1.0">항공기 만족도 예측 </span></center>**', unsafe_allow_html=True)
st.markdown('**<center><span style="color: #666666; font-size:150%; line-height:2.2">6가지 머신러닝 모델 활용으로 변수들을 바꾸어 예측해보기</span></center>**', unsafe_allow_html=True)
#st.markdown('**<center><span style="color: #000000; font-size:150%"> </span></center>**', unsafe_allow_html=True)

# 첫 번째 행
r1_col1, r1_col2, r1_col3 = st.columns(3)

Inflight_wifi_service = r1_col1.slider("Inflight wifi service", 1, 5)

Departure_Arrival_time_convenient = r1_col2.slider("Departure/Arrival time convenient", 1, 5)

Ease_of_Online_booking = r1_col3.slider("Ease of Online booking", 1,5)


# 두번째 행
r2_col1, r2_col2, r2_col3 = st.columns(3)

Gate_location = r2_col1.slider("Gate location",1,5 )
Food_and_drink = r2_col2.slider("Food and drink", 1,5)
Online_boarding = r2_col3.slider("Online boarding", 1,5)

# 세번째 행
r3_col1, r3_col2, r3_col3 = st.columns(3)
Leg_room_service=r3_col1.slider("Leg room service ",1,5)
Baggage_handling =r3_col2.slider("Baggage handling",1,5)
On_board_service =r3_col3.slider("On-board service",1,5)


# 네 번째 행
r4_col1, r4_col2, r4_col3 = st.columns(3)
Seat_comfort =r4_col1.slider("Seat comfort ",1,5)
Inflight_entertainment =r4_col2.slider("Inflight entertainment",1,5)
Check_in_service =r4_col3.slider("Check-in service",1,5)

# 다섯번째 행
r5_col1, r5_col2 = st.columns(2)
Inflight_service =r5_col1.slider("Inflight service ",1,5)
Cleanliness =r5_col2.slider("Cleanliness",1,5)

# 여섯번째 행
r6_col1, r6_col2 = st.columns(2)
Departure_Delay_in_Minutes=r6_col1.slider("Departure_Delay_in_Minutes ",0,30)
Arrival_Delay_in_Minutes=r6_col2.slider("Arrival_Delay_in_Minutes ",0 ,30)

# 예측 버튼
predict_button = st.button("만족도 예측")

st.write("---")

# 예측 결과
model_list = ['LR_model.pkl', 'KNN_model.pkl', 'DT_model.pkl', 'RandomForestClassifier_model.pkl', 'xgb_model.pkl', 'LightGBM_model.pkl']
result_list = ["로지스틱 회귀 결과", "KNN 결과", "결정트리 결과", "랜덤 포레스트 결과", "XGBoost 결과", "LightGBM 결과"]
for i in range(1, 7):
    if predict_button:
        model = joblib.load(model_list[i-1])
        pred = model.predict(np.array([[Inflight_wifi_service, Departure_Arrival_time_convenient,
        Ease_of_Online_booking, Gate_location, Food_and_drink, Online_boarding, Seat_comfort,
        Inflight_entertainment, On_board_service, Leg_room_service, Baggage_handling, Check_in_service, 
        Inflight_service, Cleanliness,Departure_Delay_in_Minutes,Arrival_Delay_in_Minutes]]))
        
        if pred == 1:
            st.metric(result_list[i-1], "Satisfied")
        else:
            st.metric(result_list[i-1], "Dissatisfied")
    
st.write("")                              

#def page2():
#    st.markdown("# Page 2 ❄️")
#    st.sidebar.markdown("# Page 2 ❄️")

       
#page_names_to_funcs = {
#    "Practice": Practice,
#    "APP": APP,
#}

#selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
#page_names_to_funcs[selected_page]()
## Contents of ~/my_app/main_page.py

