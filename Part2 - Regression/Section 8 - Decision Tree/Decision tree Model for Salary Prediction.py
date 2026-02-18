import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing the data set

dataset = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part2 - Regression\Section 7 - Support Vector Regression (SVR)\Position_Salaries.csv")
X = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

#Training THE SVR MODEL ON THE WHOLE DATASET

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state  = 0)
regressor.fit(X, y)

#Predicting a new result for X = 6.5

y_pred = regressor.predict([[6.5]])

#Visualising the Results
x_grid = np.arange(min(X),max(X),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
plt.scatter(X, y, color ='red')
plt.plot(x_grid,regressor.predict(x_grid), color = 'blue')
plt.xlabel("level")
plt.ylabel("salary")
plt.title("graphe SVR prediction")
plt.show()
