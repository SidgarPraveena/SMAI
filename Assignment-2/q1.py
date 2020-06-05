#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score


# In[3]:


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


# In[4]:


dict1=unpickle('./Datasets/Question-1/data_batch_1')
dict2=unpickle('./Datasets/Question-1/data_batch_2')
dict3=unpickle('./Datasets/Question-1/data_batch_3')
dict4=unpickle('./Datasets/Question-1/data_batch_4')
dict5=unpickle('./Datasets/Question-1/data_batch_5')


# In[5]:


x1=dict1[b'data']
print(x1.shape)
x1=np.append(x1,dict2[b'data'],axis=0)
print(x1.shape)
x1=np.append(x1,dict3[b'data'],axis=0)
print(x1.shape)
x1=np.append(x1,dict4[b'data'],axis=0)
print(x1.shape)
x1=np.append(x1,dict5[b'data'],axis=0)
print(x1.shape)


# In[6]:


y1=dict1[b'labels']
#print(y1.shape)
y1=np.append(y1,dict2[b'labels'],axis=0)
print(y1.shape)
y1=np.append(y1,dict3[b'labels'],axis=0)
print(y1.shape)
y1=np.append(y1,dict4[b'labels'],axis=0)
print(y1.shape)
y1=np.append(y1,dict5[b'labels'],axis=0)
print(y1.shape)


# In[7]:


print(x1.shape)
print(y1.shape)


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(x1,y1, test_size=0.2,random_state=109)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[9]:


scaler = MinMaxScaler()
X_train=scaler.fit_transform(X_train)  #normalizing data
X_test=scaler.fit_transform(X_test)
print(X_train.shape)
print(X_test.shape)


# In[10]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true-y_pred))


# In[11]:


C=[0.001,0.003,0.01,0.03]
acc_scr=[]
m_sq_error=[]
m_abs_error=[]
m_abs_per=[]
for i in range (len(C)):
    clf=LinearSVC(random_state=0,tol=1e-9,max_iter=500,C=C[i]) 
    clf.fit(X_train, y_train)
    y_pred=clf.predict(X_test)
    print(accuracy_score(y_test,y_pred),end=" ")
    print(mean_squared_error(y_test,y_pred),end=" ")
    print(mean_absolute_error(y_test,y_pred),end=" ")
    print(mean_absolute_percentage_error(y_test,y_pred))
    acc_scr.append(accuracy_score(y_test,y_pred))
    m_sq_error.append(mean_squared_error(y_test,y_pred))
    m_abs_error.append(mean_absolute_error(y_test,y_pred))
    m_abs_per.append(mean_absolute_percentage_error(y_test,y_pred))
    


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




