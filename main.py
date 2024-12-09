import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

st.write("""
# Indiana iLearn Prediction App

This app predicts whether a school corporation with user selected demographics will perform above or below the State Average for the ILEARN exam.
""")

st.sidebar.subheader('User Input Parameters')

def user_input_features():
    type_school=st.sidebar.selectbox("Choose Type of Corporation", ["Public (Non-Charter)", 
        "Charter", "Other"])
    diverse_population=st.sidebar.slider('Percentage of Non-White Students',0,100, 50)
    
    

    paid_lunches=st.sidebar.slider('Percentage of Students Receiving Free or Reduced Lunch Aid', 
                                  0,100,50)
    
    non_enl=st.sidebar.slider('Percentage of Non-Native English Speakers',
                             0,100,50)

    df_display={
            'WhitePopulation': diverse_population,
            'Non_ENL': non_enl,
            'Paid_Lunch': paid_lunches,
            'Corp Type': type_school}
    display=pd.DataFrame(df_display,index=["Input Selection"])    


    non_enl=1-(non_enl/100)
    diverse_population=1-(diverse_population/100)
    paid_lunches=1-(paid_lunches/100)
    if type_school=="Other":
        type_school=2
    elif type_school=="Charter":
        type_school=1
    else:
        type_school=0
    data = {
            'WhitePopulation': diverse_population,
            'Non_ENL': non_enl,
            'Paid_Lunch': paid_lunches,
            'Corp Type': type_school,}
    features = pd.DataFrame(data, index=[0])
    return features,display

df,df_display = user_input_features()
df_display=df_display.rename(columns={
            'WhitePopulation': ' Percentage of Non-White Students',
            'Non_ENL': "Percentage of Non-Native English Speakers",
            'Paid_Lunch': 'Percentage of Students Receiving Free or Reduced Lunch Aid',
            'Corp Type': 'Type of School Corporation',
            }
)

st.subheader('User Input parameters')

st.dataframe(df_display.T)


df_model=pd.read_csv('df_model.csv',index_col=0)
X=df_model.drop('AboveState',axis=1)
y=df_model['AboveState']

lr=LogisticRegression(C=.1,penalty='l2')
lr.fit(X,y)



prediction=lr.predict(df)
if prediction ==0:
    prediction="Corporation Predicted to Have Score **Below** State Average"
else:
    prediciton="Corporation Predicted to Have Score **Above** State Average"
prediction_proba=pd.DataFrame(lr.predict_proba(df))
prediction_proba=prediction_proba.rename(columns={0:"Below State Average",1:"Above State Average"})


st.subheader('Prediction')
st.markdown(prediction)


st.subheader('Prediction Probability')
st.write(prediction_proba.set_index(prediction_proba.columns[0]))



