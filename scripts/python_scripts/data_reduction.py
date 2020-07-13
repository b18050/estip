"""import libraries"""
import numpy as np
import random
import matplotlib.pyplot as plt

#mean=list(map(float,input("enter the means").split()))
#cov=list(map(float,input("enter the covariances").split()))
#cov=np.reshape(cov,(2,2))

#----------------generating the data----------------------------#

mean=[0,0]  #mean of the data set
cov=[[7,10],[10,18]]   #covariance matrix
l=np.random.multivariate_normal(mean,cov,1000)#dimension =(1000 X 2)  1000 vectors
x=[l[i][0] for i in range(1000)]   # array of x coordinate
y=[l[i][1] for i in range(1000)]   #array of y coordinate
plt.scatter(x,y)   #scatter plot for the generated data

#------covariance matrix and eigen values and eigen vectors----------------#

cov_matrix=np.cov(l.T)   #covariace matrix dimension (2 X 2) of features
eig_vals, eig_vecs = np.linalg.eig(cov_matrix)   # eigen values and eigen vectors respect..
qt=eig_vecs.transpose()   # transpose of eigen_vectors_matrix

#-----------plotting direction of eigen vectors q1 and q2------------------#

plt.quiver(qt[0][0],qt[0][1],scale=5,color='r')   #direction of proection of q1 eigen vector
plt.quiver(qt[1][0],qt[1][1],scale=5,color='r')     #direction of proection of q2 eigen vector

#--------------------Dot Products with both eigen vectors and also with only one of them-----#

Dot_1_2=qt.dot(l.T)   # taking dot product with both eigen vectors
Dot_1=qt[0].dot(l.T)   #taking dotproduct with q1 eigen vector
Dot_2=qt[1].dot(l.T)   #taking  dotproduct with q2 eigen vector

#----------------------Projection within three directions---------------3
Pro_2=[]   #initialise  list to store projection on q2
Pro_1=[]    #initialise  list to store projection on q1
Pro_1_2=[]  #initialise  list to store projection on q1 and q2 both

#now projecting on every eigen vectors i.e.multiply by data
for i in range(1000):
    Pro_1.append(Dot_1[i]*eig_vecs.T[0])     #for q1 eigen vectr
    Pro_2.append(Dot_2[i]*eig_vecs.T[1])     #for q2 eigen vectr
    z1=Dot_1_2.T[i][0]*eig_vecs.T[0]      #for q1 and q2 eigen vectr
    z2=Dot_1_2.T[i][1]*eig_vecs.T[1]       #for q1 and q2 eigen vectr
    Pro_1_2.append(z1+z2)

Pro_1=np.array(Pro_1).transpose()
Pro_2=np.array(Pro_2).transpose()
#-------------------plotting projected data on q1 an q2 eigen vector
plt.scatter(Pro_1[0],Pro_1[1],marker='x')
plt.scatter(Pro_2[0],Pro_2[1],marker='x')
plt.xlim([-25,20])
plt.ylim([-15,15])

#-------------calculating errors ------------------------#

error_pro_1=0
error_pro_2=0
error_pro_1_2=0
for i in range(1000):
    error_pro_1=error_pro_1 + (l[i][0]-Pro_1.T[i][0])**2 + (l[i][1]-Pro_1.T[i][1])**2
    error_pro_2=error_pro_2 + (l[i][0]-Pro_2.T[i][0])**2 + (l[i][1]-Pro_2.T[i][1])**2
    error_pro_1_2=error_pro_1_2 + (l[i][0]-Pro_1_2[i][0])**2 + (l[i][1]-Pro_1_2[i][1])**2

# print("Error when both are taken",error_pro_1_2**0.5)
# print("Error when Projected on 1st q1",error_pro_1**0.5)
# print("Error when  Projected on 2nd q2",error_pro_2**0.5)
