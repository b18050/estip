import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
#def _load_data(sklearn_load_ds):
 #       data = sklearn_load_ds
  #      X = pd.DataFrame(data.data)
   #     self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, data.target, test_size=0.3, random_state=42)
def load_data(X):
    df = pd.read_csv(r"___path_to_file___")
    pca = PCA(n_components=10).fit(df)
    reduced_data = PCA(n_components=2).fit_transform(data)
    return(reduced_data)
    
def distance(point,center):
    d=(point[0]-center[0])**2+(point[1]-center[1])**2
    return(d)
    
def K_Means(data):
    knn=KMeans(n_clusters=10)
    knn.fit(data)
    pred_X=knn.labels_#centers
    #SSE(data,pred_X)
    plt.scatter(data[:,0],data[:,1],c=pred_X,cmap="rainbow")
    centers=knn.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1],marker='x')
    plt.show()
    sum=0
    for i in range(len(data)):
        sum=sum+distance(data[i],centers[pred_X[i]])
    print("SSE:",sum)

def GMM(data):
    gmm=GaussianMixture(n_components=10)
    gmm.fit(data)
    labels = gmm.predict(data)
    plt.scatter(data[:,0],data[:,1],c = labels,cmap ="viridis")
    plt.show()
    print("Purity Score for K=10 :",purity_score(digits.target,labels))
    

  
def purity_score(y_true, y_pred):
# compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix,axis=0))/np.sum(contingency_matrix) 
    print("Purity Score for K=10 :",purity_score(____))
    
    
    
data=load_data() 
    
X=1    
data=load_data(X)
L=[2,5,7,8,10,12,15,17]

#for i in L:
K_Means(data)
arr = [2,3,4,5,8,10,12,15,17]
kmeans_ = [KMeans(n_clusters = i,random_state=0).fit(data) for i in arr]
pred = [kmeans_[i].labels_ for i in range(len(arr))]
x=[kmeans_[i].inertia_ for i in range(len(arr))]

plt.xlabel("Elbow curve for k")
plt.ylabel("Cost Squared Error")
plt.plot(arr,x,color="r",linewidth=3)
plt.show()

GMM(data)

gmm_ = [GaussianMixture(n_components=i).fit(data) for i in L]
pred = [gmm_[i].predict(data) for i in range(len(L))]
x_=[gmm_[i].score(data) for i in range(len(L))]

plt.xlabel("Elbow curve for k")
plt.ylabel("Log Likelihood")
plt.plot(L,x_,color="r",linewidth=3)
plt.show()
 
