#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report as cr


# In[10]:


class AuthorClassifier:
    def __init__(self):
        self.SVM=None
        self.v=None
        
    def read_data(self,mypath):
        df=pd.read_csv(mypath)
        df=df.drop(df.columns[0],axis=1)
        #X_train,X_test,Y_train,Y_test=train_test_split(df['text'],df['author'], test_size=0.2,random_state=40)
        return df
    def vectorize_data(self,df):  
        vectorizer = TfidfVectorizer(use_idf=True)
        #vectorizer.fit(df['text'])
        df_tfidf = vectorizer.fit_transform(df['text'])
        self.v=vectorizer.vocabulary_
        return df_tfidf
    def vectorize_test_data(self,df):
        #vocab=self.vocab
        vectorizer = TfidfVectorizer(use_idf=True, vocabulary=self.v)
        df_tfidf = vectorizer.fit_transform(df['text'])
        return df_tfidf
    def encode_label(self,df):
        Encoder = LabelEncoder()
        label = Encoder.fit_transform(df['author'])
        return label
    def train(self,mypath):
        data=self.read_data(mypath)
        data_tf_idf=self.vectorize_data(data)
        data_label=self.encode_label(data)
        self.SVM = svm.SVC(C=2.0, kernel='linear', gamma='auto')
        self.SVM.fit(data_tf_idf,data_label)
    def predict(self,mypath):
        df_test=pd.read_csv(mypath)
        df_test_tfidf=self.vectorize_test_data(df_test)
        predictions=self.SVM.predict(df_test_tfidf)
        #print(df_test_tfidf.shape)
        return predictions
        


# In[11]:


#auth_classifier = AuthorClassifier()
#auth_classifier.train('./Datasets/Question-5/train(1).csv') # Path to the train.csv will be provided
#predictions = auth_classifier.predict('./Datasets/Question-5/train(1).csv') # Path to the test.csv will be provided
#print(predictions)


# In[ ]:





# In[ ]:




