import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the data set

dataset = pd.read_csv(r"Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#FEATUR SCALLING (its important in the SVR Regression)

"""#before we need to turn the y vector into a 2d array to do so we re gonna be using
reshape function"""
y = y.reshape(len(y),1)
from sklearn.preprocessing import StandardScaler

#importante the Standard scaler is calculating the mean of the matrix so you
#you should allways specify a knew one for each vector
sc_x = StandardScaler() #this one is calculating the mean of X to be filled
X = sc_x.fit_transform(X) #it expect a 2D array
sc_y = StandardScaler() #So you need one for Y to calculate its Y mean"""
y = sc_y.fit_transform(y)

#Training THE SVR MODEL ON THE WHOLE DATASET

from sklearn.svm import SVR
reg = SVR(kernel = 'rbf')
"a kernel is a function that learns"
reg.fit(X, y.ravel())

#Predicting a new result for X = 6.5

y_scaled = reg.predict(sc_x.fit_transform([[6.5]])) #it returns the result scaled it needs to be returned by its original value
y_pred = sc_y.inverse_transform(y_scaled.reshape(-1,1))

#Visualising the Results

plt.scatter(sc_x.inverse_transform(X), sc_y.inverse_transform(y), color ='red')
plt.plot(sc_x.inverse_transform(X), sc_y.inverse_transform(reg.predict(X).reshape(-1,1)))
plt.xlabel("level")
plt.ylabel("salary")
plt.title("graphe SVR prediction")
plt.show()
