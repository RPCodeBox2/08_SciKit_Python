# In[1] - Documentation
"""
Script - 12-k-means-clustering.py
Decription - 
Author - Rana Pratap
Date - 2020
Version - 1.0
https://nickmccullum.com/python-machine-learning/k-means-clustering-python/
"""
print(__doc__)


# In[2] - Create artificial data set
from sklearn.datasets import make_blobs
raw_data = make_blobs(n_samples = 200, n_features = 2, centers = 4, cluster_std = 1.8)

#Visualization imports
#import seaborn
import matplotlib.pyplot as plt
#%matplotlib inline

#Visualize the data
plt.scatter(raw_data[0][:,0], raw_data[0][:,1])
plt.scatter(raw_data[0][:,0], raw_data[0][:,1], c=raw_data[1])

# In[3] - Build and train the model
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(raw_data[0])

#See the predictions
model.labels_
model.cluster_centers_

# In[4] - PLot the predictions against the original data set
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('Our Model')
ax1.scatter(raw_data[0][:,0], raw_data[0][:,1],c=model.labels_)
ax2.set_title('Original Data')
ax2.scatter(raw_data[0][:,0], raw_data[0][:,1],c=raw_data[1])

# In[] - 
