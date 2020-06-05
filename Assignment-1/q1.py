#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from math import sqrt
from numpy import array
from numpy.linalg import norm
from sklearn.metrics import accuracy_score
from sklearn import metrics
from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier
import csv
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report as cr


# In[3]:


class KNNClassifier: 
    def __init__(self):
        self.num_neighbors=3
    
    def train(self, filename):
        self.filename=filename
        self.df=pd.read_csv(self.filename, header=None)
        
    
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
        df_train=self.df
        df_test=pd.read_csv(test_filename, header=None)
        
        predictions = self.knn(np.array(df_train), np.array(df_test))

        return predictions

