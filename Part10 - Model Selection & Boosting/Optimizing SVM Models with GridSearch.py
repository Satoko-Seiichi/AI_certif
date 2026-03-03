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

#TRAINING THE SUPORT VECTOR MACHINE MODEL (Kernel SVM)
from sklearn.svm import SVC
clf = SVC(kernel = 'linear', random_state = 0)
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

#Applying the K-fold Cross validation
"""this section is like training and testing your model in the same time it makes some splitings and then it validates them"""
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridsearchCV
parametrs = [{'C': [0.25, 0.5, 0.75, 1], 'kernel':['linear']},#this dictionary for the Linear kernel
             {'C': [0.25, 0.5, 0.75, 1], 'kernel':['rbf'], 'gamma':[0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}#this one is for the Rbf kernel
             ]
grid_search = GridSearchCV(estimator = clf, 
                           param_grid = parametrs, #to choose from the dictionnary the best parameter
                           scoring = 'accuracy',
                           cv = 10 #cross validation
                           n_jobs = -1 #it means all the processors in my pc should work to search the best parameters
                    
                           ) 
grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
print("best Accuracy ", best_accuracy)
print("Best parameters", best_prameters)

