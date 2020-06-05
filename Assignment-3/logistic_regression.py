
import numpy as np
import pandas as pd
import seaborn as sns
import cv2
import sys
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from PIL import Image
from numpy.linalg import eig
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report as cr
import sys



class PCA:
    def __init__(self,n_components=200):
        self.n_components = n_components
        
    def load_train_images(self,train_data_path):
        f = open(train_data_path, "r")
        images=[]
        labels=[]
        for x in f:
            x=x.split(" ")
            img=cv2.imread(x[0],0)
            dim=(50,50)
            img=cv2.resize(img,dim)
            images.append(np.array(img).ravel())
            l=x[1].strip()
            labels.append(l)
        images=np.array(images)
        return images,labels
    
    def load_test_images(self,test_data_path):
        f1 = open(test_data_path, "r")
        images=[]
        for y in f1:
            y=y.strip()
            img=cv2.imread(y,0)
            dim=(50,50)
            img=cv2.resize(img,dim)
            images.append(np.array(img).ravel())
        images=np.array(images)
        return images
    
    def get_labels(self,path):
        files=os.listdir(path)
        labels=[int(i.split("_")[0]) for i in files]
        labels=np.array(labels)
        return labels
    
    def compute_mean(self,image_data):
        mean=np.mean(image_data.T, axis=1)
        return mean
        
    def compute_covariance(self,image_data, mean):
        return np.cov((image_data-mean).T)
        
    def get_eigens(self,covar_matrix):
        return np.linalg.eig(covar_matrix)
    
    def sort_eigens(self,eigen_values,eigen_vectors):
        index = eigen_values.argsort()[::-1]   
        eigen_values = eigen_values[index]
        eigen_vectors = eigen_vectors[:,index]
        return eigen_values, eigen_vectors
    
    def reduce_dim(self,image_data, eigen_values, eigen_vectors):
        new_data = eigen_vectors[:,:self.n_components]
        return image_data.dot(new_data)
    
    def pca_(self,path):
        image_data=self.load_images(path)
        labels=self.get_labels(path)
        mean=self.compute_mean(image_data)
        covariance=self.compute_covariance(image_data, mean)
        eigen_values,eigen_vectors=self.get_eigens(covariance)
        eigen_values,eigen_vectors=self.sort_eigens(eigen_values,eigen_vectors)
        reduced_data=self.reduce_dim(image_data, eigen_values, eigen_vectors)
        return reduced_data,labels


class logistic_regression:
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    def costFunction(self,theta, x, y):
        m = len(y)
        h = self.sigmoid(x @ theta)
        gradient = 1/m * ((y-h) @ x)
        return gradient
    
    def fit(self,x, y, max_iter=5000, alpha=0.0001):
        x = np.insert(x, 0, 1, axis=1)
        thetas = []
        classes = np.unique(y)
        for c in classes:
            binary_y = np.where(y == c, 1, 0)
            theta = np.zeros(x.shape[1])
            for epoch in range(max_iter):
                grad = self.costFunction(theta, x, binary_y)
                theta = theta + alpha * grad
            thetas.append(theta)
        return thetas, classes
    
    def predict(self,classes, thetas, x):
        x = np.insert(x, 0, 1, axis=1)
        preds = [np.argmax(
            [self.sigmoid(xi @ theta) for theta in thetas]) for xi in x]
        return [classes[p] for p in preds]
    
    def log_reg(self,reduced_train_data,train_labels,reduced_test_data):
        #X_train,X_test, y_train, y_test=self.split_data(PCA_data,labels)
        thetas, classes = self.fit(reduced_train_data, train_labels, 3000)
        pred = self.predict(classes, thetas, reduced_test_data)
        return pred
        


train_location = sys.argv[1]  #"sample_train_2.txt"
test_location = sys.argv[2]  #"sample_test_2.txt"
obj1=PCA()
train_images,train_labels=obj1.load_train_images(train_location)
test_images=obj1.load_test_images(test_location)

mean_train=obj1.compute_mean(train_images)
covariance_train=obj1.compute_covariance(train_images, mean_train)
eigen_values_train,eigen_vectors_train=obj1.get_eigens(covariance_train)
eigen_values_train,eigen_vectors_train=obj1.sort_eigens(eigen_values_train,eigen_vectors_train)
reduced_train_data=obj1.reduce_dim(train_images, eigen_values_train, eigen_vectors_train)

mean_test=obj1.compute_mean(test_images)
covariance_test=obj1.compute_covariance(test_images, mean_test)
eigen_values_test,eigen_vectors_test=obj1.get_eigens(covariance_test)
eigen_values_test,eigen_vectors_test=obj1.sort_eigens(eigen_values_test,eigen_vectors_test)
reduced_test_data=obj1.reduce_dim(test_images, eigen_values_test, eigen_vectors_test)

obj2=logistic_regression()
obj2.log_reg(reduced_train_data,np.array(train_labels),reduced_test_data)






