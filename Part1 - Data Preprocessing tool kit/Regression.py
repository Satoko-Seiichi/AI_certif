#! python3

#this is my first script yet
print("hel")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
print("hello")

#importing dataset
dataset = pd.read_csv(r"C:\Users\youne\Downloads\Data.csv")
X = dataset.iloc[:,:-1].values #dependent variable which we gonna use to predict
Y = dataset.iloc[:, -1].values #independent variable which 

print(X)
print(Y)

#Taking care of missing values

from sklearn.impute import SimpleImputer
"""#(SimpleImputer is a class) we import thr impute methode
#from the sklearn lib to replace the missing variables inside our code"""
imputer = SimpleImputer( missing_values = np.nan , strategy = 'mean')
imputer.fit(X[:, 1:3])#this line is only calculating the mean value of the column with missing variables

print("hi")
X[:, 1:3] = imputer.transform(X[:, 1:3])#and for this one he is only returning the new matrix of columns with the missing values filled

#Encoding the category column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

#Now we need to encode the dependent variable in our code to turn it into numerical values
#from yes and no to 1 & 0
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
