import streamlit as st
import re
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

def transform_data(X):
    X['CabinClass'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^a-zA-Z]', '', x))
    X['CabinNumber'] = X['Cabin'].fillna('M').apply(lambda x: str(x).replace(" ", "")).apply(lambda x: re.sub(r'[^0-9]', '', x)).replace('', 0) 
    X['Embarked'] = X['Embarked'].fillna('M')
    X = X.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1)
    return X

def predict_result():
    # Create a DataFrame from user input

    columns = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
    numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'CabinNumber']
    categorical_features = ['Sex', 'Embarked', 'CabinClass']
    
    row = np.array([passengerid,pclass,name,sex,age,sibsp,parch,ticket,fare,cabin,embarked]) 
    X = pd.DataFrame([row], columns = columns)

    X = transform_data(X)
    X[numeric_features] = scaler.transform(X[numeric_features])
    X_encoded = encoder.transform(X[categorical_features]).toarray()
    X_encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(categorical_features))
    X = pd.concat([X.drop(categorical_features, axis=1), X_encoded_df], axis=1)

    # Make prediction
    prediction = model.predict(X)

    # Display the result
    if prediction[0] == 1:
        st.success('Passenger Survived :thumbsup:')
    else:
        st.error('Passenger did not Survive :thumbsdown:')

trigger = st.button('Predict', on_click=predict_result)