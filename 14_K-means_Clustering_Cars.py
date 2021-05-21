# In[1] - Documentation
"""
Script - 14_K-means_Clustering_Cars.py
Decription - 
Author - Rana Pratap
Date - 2020
Version - 1.0
https://www.kaggle.com/thebrownviking20/cars-k-means-clustering-script
"""
print(__doc__)


# In[1] -  Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:/PROJECTS/PYTHON/08_SciKit_Python/autos.csv')

#X = dataset.iloc[:,:-1].values
X = dataset.iloc[:,:].values

X = pd.DataFrame(X)
X = X.convert_objects(convert_numeric=True)
#X.columns = ['mpg', ' cylinders', ' cubicinches', ' hp', ' weightlbs', ' time-to-60', 'year']
#X.columns = ['mpg', ' cylinders', ' cubicinches', ' hp', ' weightlbs', ' time-to-60', 'year']
X.columns = ["symbol","loss","make","fuel","aspir","doors","style","drive","eng_loc","wb","length","width","height","weight","eng_type","cylinders","eng_cc","fuel.sys","bore","stroke","comp.ratio","hp","rpm","city_mpg","hw_mpg","price"]

# In[2] - Eliminating null values
for i in X.columns:
    X[i] = X[i].fillna(int(X[i].mean()))
for i in X.columns:
    print(X[i].isnull().sum())

# In[3] - Using the elbow method to find  the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# In[5] - Applying k-means to the cars dataset
kmeans = KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0) 
y_kmeans = kmeans.fit_predict(X)
X = X.as_matrix(columns=None)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0,1],s=100,c='red',label='US')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1,1],s=100,c='blue',label='Japan')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2,1],s=100,c='green',label='Europe')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of car brands')
plt.legend()
plt.show()

# In[] - 