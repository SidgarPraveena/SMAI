#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[53]:


class Airfoil:
    def __init__(self):
        self.coefficient=None
    def read_data(self,mypath):
        df=pd.read_csv(mypath,header=None)
        #print(df.shape)
        df=np.array(df)
        return df[:,0:5],df[:,5:]
    def getPredict(self,coeffs,row):
        yPred=coeffs[0]
        for i in range(1,len(coeffs)):
            yPred=yPred+(coeffs[i]*row[i-1])
        return yPred
    def normalize_data(self,df):
        scaler = MinMaxScaler()
        return scaler.fit_transform(df)  #normalizing data
    def fit(self,X,Y,initial_value,iterations,l_rate):
        coeffs = []
        for i in range(0,len(X[0])+1):
            coeffs.append(initial_value)
        for i in range(iterations):
            for j in range(len(X)):
                yPred=self.getPredict(coeffs,X[j])
                error=yPred-Y[j]
                coeffs[0]-=l_rate*error
                for k in range(len(X[j])):
                    coeffs[k+1]-=l_rate*error*X[j][k]
        self.coefficient=coeffs
    def train(self,mypath):
        data_train,data_label=self.read_data(mypath)
        data_train_normalize=self.normalize_data(data_train)
        self.fit(data_train_normalize,data_label,0.0,400,0.001)
    def predict(self,mypath):
        df_test=pd.read_csv(mypath,header=None)
        df_test=np.array(df_test)
        df_test_normalize=self.normalize_data(df_test)
        predictions=[]
        for k in range(len(df_test_normalize)):
            predictions.append(self.getPredict(self.coefficient,df_test_normalize[k]))
        return predictions
        
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




