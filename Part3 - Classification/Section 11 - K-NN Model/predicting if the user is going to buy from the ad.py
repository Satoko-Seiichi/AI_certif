import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Importing the Daa set
data_set  = pd.read_csv(r"Social_Network_Ads.csv")
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

#SPLITING THE DATA
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

#Training the model
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)

#Prediction a new result 

print(neigh.predict(sc_x.fit_transform([[23, 0]])))

#Predicting the Test set results
y_pred = neigh.predict(X_test)

#Confusing Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)


