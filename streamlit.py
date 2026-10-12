import pandas as pd 
import numpy as np  
import pickle as p   
from sklearn.preprocessing  import  StandardScaler,OneHotEncoder,LabelEncoder   
import tensorflow as tf    
print('libraries imported ')     
import streamlit as st


model=tf.keras.models.load_model('model.h5')       


# loading encoders and Scalers 
with open('scaler.pkl','rb')as f: 
    scaler=p.load(f)  

with open('label_gender.pkl','rb') as f: 
    label_encoder_gender=p.load(f)  

with open("one_hot_geo.pkl",'rb') as f : 
    one_hot_encoder_geo=p.load(f)  

print(" encoder and Scalers are loaded ")




st.title("Customer Chun Prediction  App")   


# user input 

geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox("Gender",label_encoder_gender.classes_)
age=st.slider("Age",18,100)
balance=st.number_input("Balance")  
# 'CreditScore', 'Geography',
#        'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
#        'IsActiveMember', 'EstimatedSalary',

estimated_salary=st.number_input("EstimatedSalary") 
credit_score=st.number_input("CReditScore")  
tenure=st.slider("Tenure",0,10) 
numof_products=st.slider("Number of products ",0,10)
hascards=st.selectbox('hascard',[0,1]) 
Isactive_member=st.selectbox('isactive member',[0,1])   

input_data = pd.DataFrame({
'CreditScore': [credit_score],
'Gender': [label_encoder_gender.transform([gender])[0]],
'Age': [age],
'Tenure': [tenure],
'Balance': [balance],
'NumOfProducts': [numof_products],
'HasCrCard': [hascards],
'IsActiveMember': [Isactive_member],
'EstimatedSalary': [estimated_salary]


}) 



#  One hot encoding for Geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()     
geo_encode_df= pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])   )




# Concatenate the one-hot encoded geography with the input data     
input_data=pd.concat([input_data.reset_index(drop=True), geo_encode_df.reset_index(drop=True)], axis=1  )



#  Scale the input Data  
input_data_scaled = scaler.transform(input_data)  


# Prediction  Churn 


prediction = model.predict(input_data_scaled)
prediction_proba=prediction[0][0]



st.write(f'prediction for probabilty : {prediction_proba}')  
if prediction_proba  >.5 : 
    st.write('customer will stay')  

else : 
    st.write("customer will leave ")