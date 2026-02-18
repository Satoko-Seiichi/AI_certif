import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the data set
dataset = pd.read_csv(r"Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training the Linear Regressor (only for comparison not to do in the future)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,y)

#Training the Polynomlial Regressor

from sklearn.preprocessing import PolynomialFeatures

poly_matrix = PolynomialFeatures( degree = 7) #setting N to be equal to two (n = 2)

X_poly = poly_matrix.fit_transform(X) #this one is going to take controle to calculate the matrix

poly_reg = LinearRegression()

poly_reg.fit(X_poly, y) # this going to search for all the coeffitions in order to
#to find the relation between y and x

#comparing the result of the graphs
#simple linear graph
"""
plt.scatter(X,y , color = 'red')
plt.plot(X, reg.predict(X))
plt.xlabel('post level')
plt.ylabel('salary')
plt.show()
"""
plt.scatter(X,y , color = 'red')
plt.plot(X, poly_reg.predict(X_poly))
plt.xlabel('post level')
plt.ylabel('salarypol')
plt.title("polynomial graph")
plt.show()

