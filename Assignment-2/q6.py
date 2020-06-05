#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import homogeneity_score
from sklearn.metrics import pairwise_distances_argmin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


# In[23]:


class Cluster:
    def __init__(self):
        self.k=5
        self.dimen_Reduction=4
    def read_data(self,mypath):
        dirs = os.listdir( mypath )
        data=[]
        for i in dirs:
            with open(mypath + i,'rb') as f:
                a = f.read().decode('unicode_escape')
                a = "".join(i for i in a if ord(i) <128)
                data.append(a)
        data = np.array(data)
        return data
    def vectorize_data(self,data):
        tf_idf_vectorizor = TfidfVectorizer(stop_words = 'english',max_features = 20000)
        tf_idf = tf_idf_vectorizor.fit_transform(data)
        tf_idf_array = tf_idf.toarray()
        return tf_idf_array
    def reduce_dimension(self,data):
        pca = PCA(n_components = self.dimen_Reduction)
        reduce_data = pca.fit_transform(data)
        return reduce_data
    def init_centres(self,tf_idf_array,rseed=2):
        rng = np.random.RandomState(rseed)
        i = rng.permutation(tf_idf_array.shape[0])[:self.k]
        centers = tf_idf_array[i]
        return centers
    def e_step(self,data,centroids):
        labels = pairwise_distances_argmin(data, centroids)
        return labels
    def m_step(self,data,labels,centroids):
        new_centers = np.array([data[labels == i].mean(0) for i in range(self.k)])
        return new_centers
    def em_steps(self,data,centroids,iterations):
        for i in range(iterations):
            labels=self.e_step(data,centroids)
            centroids=self.m_step(data,labels,centroids)
        return centroids,labels
    def get_labels(self,mypath):
        y=[]
        dirs = os.listdir( mypath )
        for i in dirs:
            y.append(int(i.split('_')[1].split('.')[0]))
        return y
    def get_homo_score(self,mypath,labels):
        return homogeneity_score(self.get_labels(mypath),labels)
    def find_clusters(self,X, n_clusters,mypath, rseed=2):
        homo_score_1=[]
        rng = np.random.RandomState(rseed)
        i = rng.permutation(X.shape[0])[:n_clusters]
        centers = X[i]
        while True:
            labels = pairwise_distances_argmin(X, centers)
            new_centers = np.array([X[labels == i].mean(0) for i in range(n_clusters)])
            homo_score_1.append(homogeneity_score(self.get_labels(mypath),labels))
            if np.all(centers == new_centers):
                break
            centers = new_centers
        return max(homo_score_1)
    
    def cluster(self,mypath):
        data=self.read_data(mypath)
        data_tf_idf_array=self.vectorize_data(data)
        init_centers=self.init_centres(data_tf_idf_array)
        centroids,labels=self.em_steps(data_tf_idf_array,init_centers,5)
        #h1=self.get_homo_score(mypath,labels)
        

        #h2=self.find_clusters(data_tf_idf_array,5,mypath)
        return labels
        
    


# In[24]:


#cluster_algo = cl()
#predictions = cluster_algo.cluster('./Datasets/Question-6/dataset/')
#print(predictions)

