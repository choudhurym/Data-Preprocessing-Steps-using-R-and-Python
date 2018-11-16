# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 14:53:39 2018

@author: Muntabir Choudhury
"""

# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

#import independent columns
X = dataset.iloc[:, :-1].values

#import dependent column
y = dataset.iloc[:, 3].values

# Taking care of missing data
#Imputer is a class and sklearn is from a scikit-learn library
#To fill the missing data with median and most_frquent value we just only need to modify strategy
#axis is the parameter, if axis = 1, then impute across all rows but if axis = 0, then impute acrosss all columns
#preprocessing and transform are the methods
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)  
imputer = imputer.fit(X[:, 1:3]) #by writing 'imputer' object in this line, we just fitted the imputer object into data set
X[:, 1:3] = imputer.transform(X[:, 1:3]) #we are replacing missing data by the mean of the column

#Encoding catergorial variable
#OneHotEncoder class is for dummy encoding
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#Encoding for independent column called 'Country' 
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
#Encoding column for dependent column called 'Purchased'
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

"""note: you would not need to add OnHotEncoder class when you can differentiate and 
can compare between things. For instance if you have to differentiate tshirt on the category S, L , M
in this case you can easily compare becuase L > M > S. However, in this particular dataset we have France, Germany and Spain. Therefore it would not make any sense 
if you categorize by 1 for France, 2 for Germany and 3 for Spain. Thus, the technique is to compare between these 3 categories by filling with dummy variables."""

#Splitting the dataset into the training set and test set
#we used random state because we don't want random value
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""note: we did not put train_size in the train_test_split() parameter because train_size and and test_size 
is equal to 1. Therefore, putting train_size in the parameter = 0.8 meaning that we are going to observe 20% of data 
test set and 80% data in the train set"""

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
