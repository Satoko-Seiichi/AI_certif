#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#importing the dataset

df = pd.read_csv(r"Salary_Data.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#split the dataset

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training the Simple linear regression model on the training set

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, Y_train) #calculat the regression the graph

Y_prdct = reg.predict(X_test)

#Visualising the data for training set

plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("graph 1")
plt.xlabel("years")
plt.ylabel("salary")
plt.show()

plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title("graph 1")
plt.xlabel("years")
plt.ylabel("salary")
plt.show()
