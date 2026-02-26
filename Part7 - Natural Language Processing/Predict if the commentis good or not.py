import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part6 - Natural Language Processing\Restaurant_Reviews.tsv", delimiter = '\t',quoting = 3 )

#Cleaning the text
#we need to import the necessary libraries
import re, nltk
"""nltk.download('stopwords')"""

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
stop_words = stopwords.words('english')
stop_words.remove("not")
for i in range(len(dataset)):
#we should replace, lower and split every word
    review = re.sub('[^a-zA-Z]',' ',dataset.iloc[i,0]).lower().split()
    ps = PorterStemmer()
#check if the word is not in the stopword list and then change it into its root form"
    review = [ps.stem(word) for word in review if not word in set(stop_words)]
    corpus.append(' '.join(review))

#now creating the Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #take out the non repetetive words that are not necessary in our training
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

##############Spliting the dat#######
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#############Training the random forest########
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(X_train, y_train)
#predicting the y
y_pred = neigh.predict(X_test)
print(
    np.concatenate(
    (y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1))
    ,1)
      )
#CONFUSION MATRIX
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test, y_pred)






