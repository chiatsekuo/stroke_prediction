import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st

st.write("""
# Stroke Detection
Detect if someon has stroke using machine learning!
""")

image = Image.open('./health.jpg')
st.image(image, caption='ML', use_column_width=True)

df = pd.read_csv('stroke.csv')

st.subheader('Data info:')
st.dataframe(df)
st.write(df.describe())
chart = st.bar_chart(df)

X = df.iloc[:, 0:10].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=66, shuffle =True)

def get_user_input():
    gender = st.sidebar.slider('gender', 0, 1, 0)
    ever_married = st.sidebar.slider('ever married', 0, 1, 0)
    work_type = st.sidebar.slider('work type', 0, 4, 2)
    residence_type = st.sidebar.slider('residence type', 0, 1, 1)
    smoking_status = st.sidebar.slider('smoking status', 0, 3, 1)
    age = st.sidebar.slider('age', 0, 100, 45)
    hypertension = st.sidebar.slider('hypertension', 0, 1, 0)
    heart_disease = st.sidebar.slider('heart desease', 0, 1, 0)
    avg_glucose_level = st.sidebar.slider('glucose', 0, 300, 100)
    bmi = st.sidebar.slider('bmi', 0, 100, 28)
    
    

    user_data = {
        'gender': gender,
        'ever_married': ever_married,
        'work_type': work_type,
        'Residence_type': residence_type,
        'smoking_status': smoking_status,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'avg_glucose_level': avg_glucose_level,
        'bmi': bmi
    }

    features = pd.DataFrame(user_data, index = [0])
    return features

user_input = get_user_input()


st.subheader('User Input:')
st.write(user_input)

RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, y_train)

st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(y_test, RandomForestClassifier.predict(X_test))*100)+'%')

prediction = RandomForestClassifier.predict(user_input)

st.subheader('Classification: ')
st.write(prediction)