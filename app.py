import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('heart.csv')

df.drop_duplicates(keep='first',inplace = True)

def restbps(data):
    if data <= 120:
        return 0
    if 121 <= data <= 129:
        return 1
    if 130 <= data <= 139:
        return 2
    if 140 <= data <= 179:
        return 4
    if data >= 180:
        return 5

df['cat_trest'] = df['trestbps'].apply(restbps)

def chol(x):
    if x < 200:
        return 0
    return 1

df['cat_chol'] = df['chol'].apply(chol)

def cat_thalach(x,y):
    if 220 - x > y:
        return 1
    else:
        return 0
df['cat_thalach'] = df[['age', 'thalach']].apply(lambda x: cat_thalach(x[0], x[1]), axis=1)

df.drop(['trestbps','chol','thalach'],axis=1,inplace=True)

X = df.drop(['target'],axis=1)
y = df['target']

X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0) 

rnd_clf = RandomForestClassifier(n_estimators=130)
rnd_clf.fit(X_train,y_train)
accuracy_train = rnd_clf.score(X_train, y_train)

y_pred = rnd_clf.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)

t = np.zeros(len(X_test.columns))

st.title("Heart Disease Prediction")

with st.form(key="form1"):
    age = st.number_input("Enter your age :", min_value=1, max_value=125, step=1)

    sex_i = st.radio("Select Sex : ",["Male","Female"])
    if sex_i == "Male":
        sex = 1
    else:
        sex = 0

    cp_i = st.radio("Select Chest Pain Type :",['Typical angina','Atypical angina','Non-anginal pain','Asymptomatic'])
    if cp_i == 'Typical angina':
        cp = 0 
    if cp_i == 'Atypical angina':
        cp = 1
    if cp_i == 'Non-anginal pain':
        cp = 2
    if cp_i == 'Asymptomatic':
        cp = 3

    trest_i = st.number_input("Enter Resting Blood Pressure (mmHg):", min_value=1, max_value=250, step=1)
    if trest_i <= 120:
        trest = 0
    if 121 <= trest_i <= 129:
        trest = 1
    if 130 <= trest_i <= 139:
        trest = 2
    if 140 <= trest_i <= 159:
        trest = 3
    if 160 <= trest_i <= 179:
        trest = 4
    if trest_i >= 180:
        trest = 5
    
    chol_i = st.number_input("Enter Serum Cholestoral (mg/dl):", min_value=1, max_value=800, step=1)
    if chol_i < 200:
        chl = 0
    else:
        chl = 1
    
    fbs_i = st.number_input("Enter Fasting Blood Sugar (mg/dl)", min_value=1, max_value=500, step=1)
    if fbs_i > 120:
        fbs = 1
    else:
        fbs = 0

    restecg_i = st.radio("Your Resting Electrocardiographic Report?",['Normal','Wave Abnormality','Possible or definite left Ventricular Hypertrophy'])
    if restecg_i == 'Normal':
        restecg = 0
    if restecg_i == 'Wave Abnormality':
        restecg = 0
    if restecg_i == 'Possible or definite left Ventricular Hypertrophy':
        restecg = 0
    


    thalach_i = st.number_input("Enter Maximum Heart Rate Achieved:",min_value=1, max_value=300, step=1)
    if 220 - age > thalach_i:
        thalach = 1
    else:
        thalach = 0 

    exang_i = st.radio("Do you experience chest pain during exercise?",['Yes','No'])
    if exang_i == 'Yes':
        exang = 1
    else:
        exang = 0
    
    oldpeak = st.number_input("Amount of ST depression induced by exercise:",min_value=0.0, max_value=10.0, step=0.01)

    slope_i = st.radio("How is the slope of the peak exercise ST segment?",['Upsloping','Flatsloping','Downslopins'] )
    if slope_i == 'Upsloping':
        slope = 0
    if slope_i == 'Flatsloping':
        slope = 1
    if slope_i == 'Downslopins':
        slope = 2
    
    ca_i = st.radio("Your Flourosopy report?",['Normal Blood Flow','Minimal Slowing of Blood Flow','Evident Slowing of Blood Flow','Partial Clotting','Complete Stasis'])
    if ca_i == 'Normal Blood Flow':
        ca = 0
    if ca_i == 'Minimal Slowing of Blood Flow':
        ca = 1
    if ca_i == 'Evident Slowing of Blood Flow':
        ca = 2
    if ca_i == 'Partial Clotting':
        ca = 3
    if ca_i == 'Complete Stasis':
        ca = 4
    
    thal_i = st.radio("Your Thalium Stress Report?",['Noraml','Narrowing of 1 or more arteries supplying the heart','Used to be defect but Ok now','No blood movement while exercise'])
    if thal_i == 'Noraml':
        thal = 0
    if thal_i == 'Narrowing of 1 or more arteries supplying the heart':
        thal = 1
    if thal_i == 'Used to be defect but Ok now':
        thal = 2
    if thal_i == 'No blood movement while exercise':
        thal = 3

    submit = st.form_submit_button("Submit")

if submit:
    t = np.array([age,sex,cp,fbs,restecg,exang,oldpeak,slope,ca,thal,trest,chl,thalach])
    t=t.reshape(1,-1)
    y_pred = rnd_clf.predict(t)
    if y_pred[0] == 1:
        st.subheader("You may be suffering from a Heart Condition.")
        st.subheader("Kindly consult a Cardiologist.")
    else:
        st.subheader("Your heart health seems to be fine.")



