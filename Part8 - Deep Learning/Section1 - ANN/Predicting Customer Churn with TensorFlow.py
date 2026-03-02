import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

######### PART 1- DATA PREPROCESSING #######################
data_set = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part8 - Deep Learning\Data\Churn_Modelling.csv")
X = data_set.iloc[:,3:-1].values
y = data_set.iloc[:,-1].values
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
#Label incoding to the gender (beceause its either 1 or 0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:,2] = le.fit_transform(X[:,2])
#One Hot encoding for the Geographical column beceause there is a lot
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))
#SPLITING THE DATA
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25 , random_state = 0)

#Feature SCALING #######Important for deep learning
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)

"""************************************************************************************************************************************************************************************"""
######### PART 2 - BUILDING THE ANN ########################
"""Initializing the ANN"""
import tensorflow as tf
ann = tf.keras.models.Sequential() #Ann as a sequence of layers
"""Adding the input layer and the first hidden layer"""
ann.add(tf.keras.layers.Dense(units = 6#how many neurons you put in the hidden layers
,activation = 'relu'))#this is a fully connected hidden layer
"""Adding the second hidden layer is just the same as adding the first hidden layer"""
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu'#relu is the rectifier function
                              ))
"""Adding the Output layer"""
ann.add(tf.keras.layers.Dense(units = 1#one output neuron beceause we have a single a binary output 
                              , activation = 'sigmoid'))
######### PART 3 - TRAINING THE ANN ########################
"""Compiling tha ANN"""
ann.compile(optimizer = "adam"#(stachastic gradient descent)update the weights in order to reduce the loss errors between the predicted and the original values
            ,loss = "binary_crossentropy"#for binary classification for non binary CLF use 'categorical_crossentropy'
            ,metrics =["accuracy"])
"""training the ANN on the Training set"""
ann.fit(X_train, y_train, batch_size = 32, epochs = 20)
######### PART 4 - MAKING PREDICTIONS AND EVALUATING THE MODEL ######################## 
ann.predict(sc_x.transform([[1,0,0,600,1,40,3,60000,2,1,1,5000]]))
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
