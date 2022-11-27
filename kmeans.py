import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import streamlit as st

def kmeans_clustering(airline, airline2):
  st.header("K-Means 군집화 결과")
  
  airline_personal = airline[["satisfaction", "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance"]]
  
  temp_list = ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]
  encoder = LabelEncoder()
  for i in temp_list:
      encoder.fit(airline_personal[i])
      airline_personal[i] = encoder.transform(airline_personal[i])
      
  airline_personal_test = airline2[["satisfaction", "Gender", "Customer Type", "Age", "Type of Travel", "Class", "Flight Distance"]]
  
  temp_list2 = ["satisfaction", "Gender", "Customer Type", "Type of Travel", "Class"]
  encoder2 = LabelEncoder()
  for i in temp_list2:
      encoder2.fit(airline_personal_test[i])
      airline_personal_test[i] = encoder2.transform(airline_personal_test[i])
  
  scaler = StandardScaler()
  scaler2 = StandardScaler()
  airline_personal_scaling = scaler.fit_transform(airline_personal)
  airline_personal_test_scaling = scaler2.fit_transform(airline_personal_test)
  
  st.subheader("엘보우 기법")
  
  sse=[]
  for i in range(1, 11):
      kmeans = KMeans(n_clusters=i, init="k-means++", random_state=100, max_iter=1000).fit(airline_personal_scaling)
      sse.append(kmeans.inertia_)
  
  f, ax = plt.subplots(figsize=(18, 12))
  ax.plot(range(1, 11), sse, marker="o")
  st.pyplot(f)
  
  st.subheader("실루엣 스코어")
  silhouette_list = [0.2531, 0.3307, 0.2750, 0.2595, 0.2833, 0.2697, 0.2782, 0.2842, 0.2842, 0.2865]
  for i in range(2, 11):
      if i == 3:
          st.markdown('<span style="color: SteelBlue; font-size:120%">**군집 3개: 0.3307**</span>', unsafe_allow_html=True)
      else:
          st.markdown('<span style="color: Azure"> 군집 %d개: %f</span>' % (i, silhouette_list[i-2]), unsafe_allow_html=True)
          
  airline_personal_kmeans = airline_personal.copy()
  kmeans = KMeans(n_clusters=3, init="k-means++", random_state=100, max_iter=1000)
  kmeans.fit(airline_personal_scaling)
  airline_personal_kmeans["KMeans"] = kmeans.predict(airline_personal_scaling)
  kmeans_result = airline_personal_kmeans.groupby(["KMeans"]).mean().sort_values(by="satisfaction").reset_index()
  
  group_size = airline_personal_kmeans[["KMeans", "Gender"]].groupby(["KMeans"]).count().rename(columns={"Gender":"Group_Size"})
  group_size["Group_Proportion"] = group_size["Group_Size"] / group_size["Group_Size"].sum()
  
  st.subheader("훈련셋 군집화 결과")
  
  f, ax = plt.subplots(figsize=(18, 12))
  ax.pie(group_size["Group_Proportion"], 
         labels=["Dissatisfied Personal Traveler(Loyal)", "Dissatisfied Business Traveler(Disloyal)", 
                 "Satisfied Business Traveler(Loyal)"], 
         autopct="%.2f%%", textprops={"size":"20", "color":"k"}, explode=(0.1, 0.1, 0.1), 
         colors=["lightcoral", "aquamarine", "royalblue"])
  plt.title("Group Size Proportion", fontsize=26, fontweight="bold")
  plt.legend(loc="best", prop={"size":16}, bbox_to_anchor=(1.5, -0.1, 0.2, 1.0), title="Group", title_fontsize=16)
  st.pyplot(f)
  
  airline_personal_test_kmeans = airline_personal_test.copy()

  airline_personal_test_kmeans["KMeans"] = kmeans.predict(airline_personal_test_scaling)

  kmeans_result_test = airline_personal_test_kmeans.groupby(["KMeans"]).mean().sort_values(by="satisfaction").reset_index()

  group_size_test = airline_personal_test_kmeans[["KMeans", 
                                                  "Gender"]].groupby(["KMeans"]).count().rename(columns={"Gender":"Group_Size"})
  group_size_test["Group_Proportion"] = group_size_test["Group_Size"] / group_size_test["Group_Size"].sum()
  
  st.subheader("시험셋 군집화 결과")
  
  g, ax = plt.subplots(figsize=(18, 12))
  ax.pie(group_size_test["Group_Proportion"], 
         labels=["Dissatisfied Personal Traveler(Loyal)", "Dissatisfied Business Traveler(Disloyal)", 
                 "Satisfied Business Traveler(Loyal)"], 
         autopct="%.2f%%", textprops={"size":"20", "color":"k"}, explode=(0.1, 0.1, 0.1), 
         colors=["lightcoral", "aquamarine", "royalblue"])
  plt.title("Group Size Proportion(test set)", fontsize=26, fontweight="bold")
  plt.legend(loc="best", prop={"size":16}, bbox_to_anchor=(1.5, -0.1, 0.2, 1.0), title="Group", title_fontsize=16)
  st.pyplot(g)
