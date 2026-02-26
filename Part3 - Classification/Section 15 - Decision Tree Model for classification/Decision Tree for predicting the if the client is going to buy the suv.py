import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#IMPORTING THE DATA SET
data_set = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part3 - Classification\Section 11 - K-NN Model\Social_Network_Ads.csv")
X = data_set.iloc[:,:-1].values
y = data_set.iloc[:,-1].values

#SPLITING THE DATA
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25 , random_state = 0)

#Feature SCALING
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)


#TRAINING THE Decision Tree MODEL for calssifying
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion = 'entropy' , random_state = 0)
#entropy is an algorithm that going to make the splits 
clf.fit(X_train, y_train)

#PREDICTING IF THIS COSTUMER IS GOING TO BUY
print(clf.predict(sc_x.fit_transform([[30, 87000]])))

#predicting the y
y_pred = clf.predict(X_test)
print(
    np.concatenate(
    (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1))
    ,1)
      )


#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)

