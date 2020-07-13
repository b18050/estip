"""
//-------------Agglomerative Clustering--------------------//
"""
"""
//--------------Import Libraries------------------------//
"""
import matplotlib.pyplot as     plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

def load_data(X):
    digits = load_digits()
    data = scale(digits.data)
    pca = PCA(n_components=10).fit(data)
    reduced_data = PCA(n_components=2).fit_transform(data)
    return(reduced_data)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix=metrics.cluster.contingency_matrix(y_true,y_pred)
    #print(contingency_matrix)
    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
    # Return cluster accuracy
    return (contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix))
   

data=load_data()

def K_Means(data):
    knn=KMeans(n_clusters=10)
    knn.fit(data)
    labels=knn.labels_#centers
    #SSE(data,pred_X)
    plt.scatter(data[:,0],data[:,1],c=labels)
    centers=knn.cluster_centers_
    plt.scatter(centers[:,0],centers[:,1],marker='x')
    plt.title("K-Means")
    acc=purity_score(____)
    print("accuracy:",acc)
    plt.show()

def AggloClustering(data):
    cluster = AgglomerativeClustering(n_clusters=10, affinity='euclidean', linkage='ward')
    cluster.fit(data)
    labels=cluster.labels_
    plt.scatter(data[:,0],data[:,1], c=cluster.labels_, cmap='rainbow')
    plt.title("Agglomerative Clustering")
    acc=purity_score(______)
    print("accuracy:",acc)
    plt.show()
    
X=1    
data=load_data(X)
K_Means(data)
AggloClustering(data)


eps=[0.05,0.5,0.95]
samples=[1,10,30,50]
for i in eps:
    print("epsilon:",i)
    for j in samples:
        db = DBSCAN(eps = i, min_samples =j).fit(data)

        #print(labels)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        #print(labels)
#        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#        unique_labels = set(labels)
#        colors = ['y', 'b', 'g', 'r']
#        #print(colors)
#        for k, col in zip(unique_labels, colors):
#            if k == -1:
#                # Black used for noise.
#                col = 'k'
#
#            class_member_mask = (labels == k)
#
#            xy = data[class_member_mask & core_samples_mask]
#            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#                                              markeredgecolor='k',
#                                              markersize=6)
#
#            xy = data[class_member_mask & ~core_samples_mask]
#            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
#                                              markeredgecolor='k',
#                                              markersize=6)
        #plt.show()
        plt.scatter(data[:,0],data[:,1], c=labels, cmap='rainbow')
#        plt.title('number of clusters: %d' %n_clusters_)
#        print("eps:",i , " " ,"min_samples" ,j)
#        acc=purity_score(digits.target,labels)
#        print("accuracy:",acc)
        plt.show()


