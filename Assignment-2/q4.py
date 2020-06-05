#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# In[38]:


class Weather:
    def __init__(self):
        self.coefficient=None
    def read_data(self,mypath):
        df=pd.read_csv('./Datasets/Question-4/weather.csv')
        #X_train,X_test = train_test_split(np.array(df), test_size=0.2,random_state=40)
        return np.array(df)
    def normalize_data(self,X_tr):
        scaler = MinMaxScaler()
        X_train=scaler.fit_transform(X_tr)  #normalizing data
        #X_test=scaler.fit_transform(X_ts)
        return X_train
    def getPredict(self,coeffs,row):
        yPred=coeffs[0]
        for i in range(1,len(coeffs)):
            yPred=yPred+(coeffs[i]*row[i-1])
        return yPred
    
    def fit(self,X,Y,initial_value,iterations,l_rate):
        coeffs = []
        #l_rate = 0.001
        for i in range(0,len(X[0])+1):
            coeffs.append(initial_value)
        for i in range(iterations):
            #print(i,end=" ")
            for j in range(len(X)):
                yPred=self.getPredict(coeffs,X[j])
                error=yPred-Y[j]
                coeffs[0]-=l_rate*error
                for k in range(len(X[j])):
                    coeffs[k+1]-=l_rate*error*X[j][k]
        return coeffs
    def train(self,mypath):
        train_data=self.read_data(mypath)
        train_data_normalize=self.normalize_data(train_data[:,3:8])
        label_data=train_data[:,4:5]
        self.coefficient=self.fit(train_data_normalize,label_data,0.0,50,0.001)
        #print(self.coefficient)
        
    def predict(self,mypath):
        test_data=self.read_data(mypath)
        test_data_normalize=self.normalize_data(test_data[:,3:8])
        predictions=[]
        for k in range(len(test_data_normalize)):
            predictions.append(self.getPredict(self.coefficient,test_data_normalize[k]))
        return predictions
            


# In[39]:


#mypath='./Datasets/Question-4/weather.csv'
#ws=Weather()
#train_data=ws.read_data(mypath)
#train_data_normalize=ws.normalize_data(train_data[:,3:8])
#label_data=train_data[:,4:5]
#ws.train(mypath)
#ws.predict(mypath)
#print(label_data.shape)
#print(train_data_normalize.shape)

#y_train=train_data[:,4:5]
#y_test=test_data[:,4:5]
#coefficient=ws.fit(train_data,Y_tr,0.0,50,0.001)


# In[ ]:





# In[ ]:





# In[ ]:




