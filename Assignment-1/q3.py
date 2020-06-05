#!/usr/bin/env python
# coding: utf-8

# In[122]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')

import random
from pprint import pprint


# In[123]:


class DecisionTree:
    def __init__(self):
        self.categorical_threshold=15
        
    def fill_missing_values(self,df):
        for column in df.columns:
            mode = df.mode()[column][0]
            df[column].fillna(mode,inplace=True)   
        return df
    
    def check_purity(self,data):
        label_column=data[:,-1]
        i=unique_classes=np.unique(label_column)
        if len(unique_classes) == 1:
            return True
        else:
            return False
        
    def get_splits(self,data):
        potential_splits={} #dictionary
        _, n_columns = data.shape
        for column_index in range(n_columns - 1):#we dont consider last column as it is label
            #potential_splits[column_index] = []
            values = data[:, column_index]
            unique_values = np.unique(values)

            type_of_feature = self.feature_types[column_index]
            if type_of_feature == "continuous":
                potential_splits[column_index]=[]
                for index in range(len(unique_values)):
                    if index != 0:
                        current_value = unique_values[index]
                        previous_value = unique_values[index-1]
                        potential_split = (current_value + previous_value) /2

                        potential_splits[column_index].append(potential_split)
            else:
                potential_splits[column_index] = unique_values
        return potential_splits
    
    def split_data(self,data, split_column, split_value):
        split_column_values = data[:, split_column]

        type_of_feature = self.feature_types[split_column]

        if type_of_feature == "continuous":
            data_below = data[split_column_values <= split_value]
            data_above = data[split_column_values > split_value]
        else:
            data_below = data[split_column_values ==split_value]
            data_above = data[split_column_values !=split_value]

        return data_below, data_above
    
    def calculate_entropy(self,data):
        label_column = data[:,-1]
        _,counts = np.unique(label_column, return_counts=True)

        probabilities=counts / counts.sum()
        entropy = sum(probabilities * -np.log2(probabilities))

        return entropy
    
    def calculate_overall_entropy(self,data_below, data_above):
        n_data_points = len(data_below) + len(data_above)

        p_data_below = len(data_below) / n_data_points
        p_data_above = len(data_above) / n_data_points

        overall_entropy = (p_data_below * self.calculate_entropy(data_below) + p_data_above * self.calculate_entropy(data_above))


        return overall_entropy
    
    def best_split(self,data, potential_splits):
        overall_entropy=999
        for column_index in potential_splits:
            for value in potential_splits[column_index]:
                data_below, data_above = self.split_data(data, split_column=column_index, split_value = value)
                current_overall_entropy = self.calculate_overall_entropy(data_below, data_above)

                if current_overall_entropy <= overall_entropy:
                    overall_entropy = current_overall_entropy
                    best_split_column = column_index
                    best_split_value = value
        return best_split_column, best_split_value
    
    def decision_tree(self,df, counter=0, min_samples=2,max_dep=5):
        if (self.check_purity(df)) or (len(df)<min_samples) or (counter == max_dep):  #return the labels which appear maximum no of times
            unique_classes, counts_unique_classes=np.unique(df[:,-1], return_counts=True)
            index=counts_unique_classes.argmax()
            return unique_classes[index] 
        else:
            counter+=1
            splits=self.get_splits(df)
            split_column, split_value = self.best_split(df, splits)
            data_below, data_above = self.split_data(df, split_column, split_value)
            
            feature_name = self.column_headers[split_column]
            type_of_feature = self.feature_types[split_column]
            if type_of_feature == "continuous":
                question = "{} <= {}".format(feature_name, split_value)
                sub_tree = {question: []}
            else:
                question = "{} = {}".format(feature_name, split_value)
                sub_tree = {question: []}
            
            yes_answer = self.decision_tree(data_below, counter,min_samples,max_dep)
            no_answer = self.decision_tree(data_above, counter,min_samples,max_dep)

            if yes_answer == no_answer:
                sub_tree = yes_answer
            else:
                sub_tree[question].append(yes_answer)
                sub_tree[question].append(no_answer)

            return sub_tree
            
        
    def train(self,filename):
        df=pd.read_csv(filename)
        self.df_train_labels=df["SalePrice"]
        #df["label"]=df.SalePrice
        df=df.drop("Id", axis=1)
        #df=df.drop("SalePrice",axis=1)
        df=self.fill_missing_values(df)
        self.feature_types=self.determine_feature_type(df)
        self.column_headers=df.columns
        #print(df.head())
        #print(df_labels.head())
        
        self.tree=self.decision_tree(np.array(df))
        #print(tree)
        
            
    def determine_feature_type(self,df):
        feature_types = []
        self.categorical_threshold=15
        for column in df.columns:
            unique_values=df[column].unique()
            if(isinstance(unique_values[0], str)) or (len(unique_values) <= self.categorical_threshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
        return feature_types
    
    def classify_example(self,example, tree):
        question = list(tree.keys())[0]
        feature_name, comparision_operator, value = question.split()

        if comparision_operator == "<=":
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]


        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return self.classify_example(example, residual_tree)
    
    def calculate_accuracy(self,df,df_labels, tree):
    
        df["classification"] = df.apply(self.classify_example, axis=1, args=(tree,))
        predictions=df["classification"]
        return predictions
    
    def predict(self,filename):
        df_test=pd.read_csv(filename)
        df_test=df_test.drop("Id", axis=1)
        df_test=self.fill_missing_values(df_test)
        
        df_train_labels=self.df_train_labels

        return self.calculate_accuracy(df_test, df_train_labels, self.tree)

