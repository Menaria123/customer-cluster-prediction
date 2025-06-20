#prepare a cluster of customers to predict the purchase power based on their income and spending score 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans
import joblib 

#loading the dataset into Dataframe 
df=pd.read_csv('Mall_Customers.csv')

df.info()
print(df.isnull().sum())

X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

wcss_list=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=1)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)

#visualize the results 
# plt.plot(range(1,11),wcss_list)
# plt.title("Elbow method")
# plt.xlabel("number of clusters")
# plt.ylabel("Wcss")
# plt.show()

model=KMeans(n_clusters=6,init="k-means++",random_state=1)
y_predict=model.fit_predict(X)
print(y_predict)
#converting the dataframe X into a numpy array 
X_array=X.values
#ploting the gragh of clusters 
plt.scatter(X_array[y_predict==0,0],X_array[y_predict==0,1],s=100,color="Green")
plt.scatter(X_array[y_predict==1,0],X_array[y_predict==1,1],s=100,color="Red")
plt.scatter(X_array[y_predict==2,0],X_array[y_predict==2,1],s=100,color="Yellow")
plt.scatter(X_array[y_predict==3,0],X_array[y_predict==3,1],s=100,color="Blue")
plt.scatter(X_array[y_predict==4,0],X_array[y_predict==4,1],s=100,color="Pink")
plt.scatter(X_array[y_predict==5,0],X_array[y_predict==5,1],s=100,color="black")

plt.title("Customer segmentation group")
plt.xlabel("spending score")
plt.ylabel("annual income")
plt.show()

joblib.dump(model,"Model.pkl")
print("Model has been saved!")