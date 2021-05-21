# In[1] - Documentation
"""
Script - 15_KNN_BrestCancer_Prediction.py
Decription - 
Author - Rana Pratap
Date - 2020
Version - 1.0
https://www.engineeringbigdata.com/breast-cancer-prediction-using-k-nearest-neighbors-algorithm-in-python/

"""
print(__doc__)

# In[2] - Import Packages and Dataset
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
 
breast_cancer = pd.read_csv('wisc_bc_data.csv')

# In[3] - Train and Testing Data
del breast_cancer['id']
X_train, X_test, y_train, y_test = train_test_split(breast_cancer.loc[:, breast_cancer.columns != 'diagnosis'],
                                                    breast_cancer['diagnosis'], stratify=breast_cancer['diagnosis'], random_state=66)
train_accuracy = []
test_accuracy = []

# In[4] - Build the k-NN Model
k = range(1, 50)
for n_neighbors in k:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    # record training set accuracy
    train_accuracy.append(knn.score(X_train, y_train))
    # record test set accuracy
    test_accuracy.append(knn.score(X_test, y_test))
plt.plot(k, train_accuracy, label="Train Accuracy")
plt.plot(k, test_accuracy, label="Test Accuracy")
plt.title('Breast Cancer Diagnosis k-Nearest Neighbor Accuracy')
plt.ylabel("Accuracy")
plt.xlabel("k")
plt.legend()
plt.show()

# In[5] - Accuracy Scores

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("k-Nearest Neighbor 3")
print(f"k-Nearest Neighbor classifier on training set: {format(knn.score(X_train, y_train), '.4f')} ")
print(f"k-Nearest Neighbor classifier on testing set: {format(knn.score(X_test, y_test), '.4f')} ")

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("k-Nearest Neighbor 5")
print(f"k-Nearest Neighbor classifier on training set: {format(knn.score(X_train, y_train), '.4f')} ")
print(f"k-Nearest Neighbor classifier on testing set: {format(knn.score(X_test, y_test), '.4f')} ")

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
print("k-Nearest Neighbor 15")
print(f"k-Nearest Neighbor classifier on training set: {format(knn.score(X_train, y_train), '.4f')} ")
print(f"k-Nearest Neighbor classifier on testing set: {format(knn.score(X_test, y_test), '.4f')} ")

knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
print("k-Nearest Neighbor 30")
print(f"k-Nearest Neighbor classifier on training set: {format(knn.score(X_train, y_train), '.4f')} ")
print(f"k-Nearest Neighbor classifier on testing set: {format(knn.score(X_test, y_test), '.4f')} ")

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
print("k-Nearest Neighbor 50")
print(f"k-Nearest Neighbor classifier on training set: {format(knn.score(X_train, y_train), '.4f')} ")
print(f"k-Nearest Neighbor classifier on testing set: {format(knn.score(X_test, y_test), '.4f')} ")

# In[6] - 
