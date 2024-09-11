import streamlit as st
from utils import PrepProcesor, columns

import numpy as np
import pandas as pd
import joblib


scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load('my_model_titanic.joblib')

st.title('Did they survive? :ship:')

# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
passengerid = st.text_input("Input Passenger ID", '123456')
pclass = st.selectbox("Choose class", [1, 2, 3])
name = st.text_input("Input Passenger Name", 'John Smith')
sex = st.selectbox("Choose sex", ['male', 'female'])
age = st.slider("Choose age", 0, 100)
sibsp = st.slider("Choose siblings", 0, 10)
parch = st.slider("Choose parch", 0, 2)
ticket = st.text_input("Input Ticket Number", "12345")
fare = st.number_input("Input Fare Price", 0, 1000)
cabin = st.text_input("Input Cabin", "C52")
embarked = st.selectbox("Did they Embark?", ['S', 'C', 'Q'])

# passengerid = '123456'
# pclass = 1
# name = 'John Smith'
# sex = 'male'
# age = 0
# sibsp = 0
# parch = 0
# ticket = "12345"
# fare = 0
# cabin = "C52"
# embarked = 'S'

# def predict_result():
#     row = np.array([passengerid, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked])
#     X = pd.DataFrame([row], columns=columns)
#     prediction = model.predict(X)
#     if prediction[0] == 1:
#         st.success('Passenger Survived :thumbsup:')
#     else:
#         st.error('Passenger did not Survive :thumbsdown:')

def predict_result():
    # Create a DataFrame from user input
    data = {
        'PassengerId': [passengerid],
        'Pclass': [pclass],
        'Name': [name],
        'Sex': [sex],
        'Age': [age],
        'SibSp': [sibsp],
        'Parch': [parch],
        'Ticket': [ticket],
        'Fare': [fare],
        'Cabin': [cabin],
        'Embarked': [embarked]
    }

    df = pd.DataFrame(data, columns=columns)

    return df

    # # Make prediction
    # prediction = model.predict(df)

    # # Display the result
    # if prediction[0] == 1:
    #     st.success('Passenger Survived :thumbsup:')
    # else:
    #     st.error('Passenger did not Survive :thumbsdown:')


#trigger = st.button('Predict', on_click=predict_result)
#trigger = st.button('Predict', on_click=predict_result)

if st.button('Predict'):
    df = predict_result()
    st.write(df)
    st.success("Данные на базу! :sunglasses:")