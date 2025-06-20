import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.cluster import KMeans
import joblib 

#load the saved model 
Kmeans=joblib.load("Model.pkl")
df=pd.read_csv("Mall_Customers.csv")
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
X_array=X.values

#streamlit application 
st.set_page_config(page_title="Customer Cluster Prediction",layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the Customer Annual Income and spending score to predict the cluster")

#input 
annual_income= st.number_input("annual income of the customer",min_value=0,max_value=400,value=50)
spending_score=st.slider("spending score 1-100",1,100,20)

#predict the cluster 
if st.button("Predict cluster"):
    input_data = pd.DataFrame([[annual_income, spending_score]],columns=["Annual Income (k$)", "Spending Score (1-100)"])
    cluster = Kmeans.predict(input_data)[0]
    st.success(f"Predicted cluster is:{cluster}")