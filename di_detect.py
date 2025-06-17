# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import math
#Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


#Load the dataset
# For static data, we can use a cache to avoid reloading it every time
@st.cache_data

def load_data():
    return pd.read_csv("diabetes.csv")

df=load_data()

#Split the dataset
X= df.drop("Outcome",axis=1)
y=df["Outcome"]
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2)

#Train the dataset
model = RandomForestClassifier(n_estimators=100)  # Using RandomForest for better performanc
model.fit(X_train,y_train)

#UI for Streamlit
st.title("Diabetes Prediction App")
st.write("Enter health details of the concerned patient")

#User inputs
Glucose= st.slider("Glucose Level", round(df["Glucose"].min()), round(df["Glucose"].max()), round(df["Glucose"].mean()))
BloodPressure = st.slider("Blood Pressure", round(df["BloodPressure"].min()), round(df["BloodPressure"].max()), round(df["BloodPressure"].mean()))
SkinThickness = st.slider("Skin Thickness", round(df["SkinThickness"].min()), round(df["SkinThickness"].max()), round(df["SkinThickness"].mean()))
Insulin = st.slider("Insulin Level", round(df["Insulin"].min()), round(df["Insulin"].max()), round(df["Insulin"].mean()))
BMI = st.slider("BMI", round(df["BMI"].min()), round(df["BMI"].max()), round(df["BMI"].mean()))
Age = st.slider("Age", round(df["Age"].min()), round(df["Age"].max()), round(df["Age"].mean()))
Pregnancies = st.slider("Pregnancies", round(df["Pregnancies"].min()), round(df["Pregnancies"].max()), round(df["Pregnancies"].mean())) 
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", round(df["DiabetesPedigreeFunction"].min(), 1), round(df["DiabetesPedigreeFunction"].max(), 1), round(df["DiabetesPedigreeFunction"].mean(), 1))


#Making the prediction
input_data= np.array([[Glucose, BloodPressure, SkinThickness, Insulin, BMI, Age, Pregnancies, DiabetesPedigreeFunction]])
prediction=model.predict(input_data)

# Add a button to trigger the prediction
if st.button("Predict"):
    st.write("Processing your input...")

    # Display the prediction result
    if prediction[0] == 1:
        st.error("The patient is likely to have diabetes.")
    else:
        st.success("The patient is likely not to have diabetes.")
