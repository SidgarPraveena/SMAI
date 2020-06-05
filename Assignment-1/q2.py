#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report as cr
from sklearn.metrics import accuracy_score


# In[78]:


class KNNClassifier: 
    def __init__(self):
        self.num_neighbors=3
    def fill_missing_values(self,df):
        for column in df.columns:
            mode = df.mode()[column][0]
            df[column]=df[column].replace({'?': mode})
        return df

    def encodeData(self,df):
        encode=list()
        for i in range(0,len(df)):
            res=list()
            for j in range(0,len(df.columns)):
                        #print(i,j,df[i][j])
                r=ord(df[j][i])
                res.append(r-97)
            encode.append(res)
        return encode
    
    def train(self, filename):
        self.filename=filename
        df=pd.read_csv(self.filename, header=None)
        df=self.fill_missing_values(df)
        self.encode=self.encodeData(df)
        
    
    def get_neighbors(self,train, test_row):
        distances = list()
        for train_row in train:
            dist=norm(train_row[1:]-test_row[0:])
            distances.append((train_row[0], dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = list()
        for i in range(self.num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def predict_classification(self,train, test_row):
        neighbors = self.get_neighbors(train, test_row)
        output_values=[]
        for i in range(len(neighbors)):
            output_values.append(neighbors[i])
        prediction=max(set(output_values), key=output_values.count)
        return prediction
    
    
    def knn(self,df_train,df_test):
        predict=[]
        for i in range(0,len(df_test)):
            pre=self.predict_classification(df_train, df_test[i])
            predict.append(pre);
        return predict
    
    def predict(self, test_filename):
        df_train=self.encode
        df=pd.read_csv(test_filename, header=None)
        df=self.fill_missing_values(df)
        df_test=self.encodeData(df)
        
        predictions = self.knn(np.array(df_train), np.array(df_test))
        for i in range(0, len(predictions)): #decode
            predictions[i]=chr(predictions[i]+97)
        return predictions


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




