import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#importing dataset
dataset = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part10 - Model Selection & Boosting\Data\Data (2).csv")
X = dataset.iloc[:,:-1].values #dependent variable which we gonna use to predict
y = dataset.iloc[:, -1].values #independent variable which
le = LabelEncoder()
y = le.fit_transform(y)
#SPLITING THE DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Training XGBoost on the Training set
from xgboost import XGBClassifier #if you want a regressor model of Xgboost you only need to replace the by XGBRegressor
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
#MakingConfusing Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)
print(cm, ac)

#Applying the K-fold Cross validation
"""this section is like training and testing your model in the same time it makes some splitings and then it validates them"""
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

