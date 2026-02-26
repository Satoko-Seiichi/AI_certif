import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the Data Set
data_set = pd.read_csv(r"C:\Users\youne\Desktop\AI_certif\Part4 - Clustering\Data\Mall_Customers.csv")
X = data_set.iloc[:,[3,4]].values


print(X)
#Using the elbow methode to find the optimale number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_ #THIS IS THE WCSS CALCULATED BY KMEANS
                )
plt.plot(range(1,11), wcss, color = 'blue')
plt.title("the Elbow Method")
plt.xlabel("Numer of clusters")
plt.ylabel("wcss")
plt.show()

#training the K-Mean model
kmeans = KMeans(n_clusters = 5, init = 'k-means++' #(the wheighted method)
                , random_state = 42)
y_kmeans = kmeans.fit_predict(X)
print(y_kmeans)

#Visualising the clusters
plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],s=100, c="red", label="Cluster 0")
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],s=100, c="blue", label="Cluster 1")
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],s=100, c="green", label="Cluster 2")
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],s=100, c="yellow", label="Cluster 3")
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],s=100, c="cyan", label="Cluster 4")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "magenta", label = "centroids")
plt.title("Clusters of customers")
plt.xlabel("Annuel income")
plt.ylabel("Spding Score")
plt.legend()
plt.show()





