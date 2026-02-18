#importing the libraries

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

#importing the data set

dataset = pd.read_csv(r"50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(X,y)

#encoding the categorical data

Ct = ColumnTransformer(transformers=[('encoder' , OneHotEncoder(), [3])],remainder = 'passthrough')
X = np.array(Ct.fit_transform(X))

print('this is the second:',X)

#Spliting the Data set into training set and Test set

from sklearn.model_selection import train_test_split

X_train , X_test, y_train , y_test = train_test_split(X,y, test_size = 0.2, random_state = 1)

#Training the modek
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)

#Predicting
y_pred = reg.predict(X_test) #the predict function uses the test data to make a
# prediction with the help of the regressor model and stores it in y_pred

#visualisation
np.set_printoptions(precision = 2)#the numbers behind the comma
print(
    np.concatenate(
        (y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1))
        ,1)
      )


