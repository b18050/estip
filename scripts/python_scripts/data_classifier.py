"""
Import Libraries
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture
import scipy

"""
Try to use the functionalities of the libraries imported.
For example, rather than converting Pandas dataframe into
a list and then perform calculations, use methods of Pandas
library.
"""
###################################################################################


###################################################################################

# Import or Load the data
def load_dataset(path_to_file):
    df=pd.read_csv(path_to_file)
    return df
###################################################################################


###################################################################################

# Data Preprocessing (Use only the required functions for the assignment)
"""
- Check for outliers.
- Check for missing values.
- Encoding categorical data
- Standardization/Normalization
- Dimensionality Reduction (PCA)
- Shuffle
- Train/Test Split
"""
def prob(x, w, mean, cov):
     p = 0
     for i in range(len(w)):
         p += w[i] * scipy.stats.multivariate_normal.pdf(x, mean[i],
cov[i], allow_singular=True)

     return p
def outliers_detection(function_parameters):
     ...
     ...
     ...

def missing_values(function_parameters):
     ...
     ...
     ...

def encoding(function_parameters):
     """
     Encode the categorical data in your dataset using One-Hot
     encoding. Very important if your dependent variable is
     categorical.
     """

def normalization(df,function_parameters):
     df2=df
     df2[function_parameters]=(df2[function_parameters]-df2[function_parameters].min())/(df2[function_parameters].max()-df2[function_parameters].min())
     return df2

def standardization(df,parameter):
     df2=df
     df2[parameter]=(df2[parameter]-df2[parameter].mean())/df2[parameter].std()
     return df2

def dimensionality_reduction(df,a):
     """
     Pass the respective function parameters needed by the function
     and perform dimentionality reduction. Retain the useful and
     significant principal components. Dimensionality reduction
     using PCA comes at a cost of interpretibility. The features in
     original data (age, height, income, etc.) can be intrepreted
     physically but not principal components. So decide accordingly.
     Then return the dimension reduced data.
     """
     X=df.drop(columns='class')
     pca=PCA(n_components=a)
     df2=pca.fit_transform(X)
     return df2


def shuffle(function_parameters):
     """
     Now your data is preprocessed. Shuffle to 'randomize' the data
     for next step of machine learning. Pass the respective parameters
     needed by the function and shuffle the data. Then return the
     shuffled data for next step of splitting it into training and test
     data.
     """

def train_test_data(df):
     X=df.drop(columns='class')
     Y=df['class']
     X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3, random_state=42)
     return X_test,X_train,Y_test,Y_train

###################################################################################


###################################################################################

# Perform classification

def classification(X_train, Y_train, n_neighbours):
    clf = KNeighborsClassifier(n_neighbours)
    clf.fit(X_train, Y_train)
    return clf

def bayes_classifier(X_train,Y_train):
     gnb=GaussianNB()
     gnb.fit(X_train,Y_train)
     return gnb

def percentage_accuracy(Y_pred, Y_test):
     classification_accuracy = sklearn.metrics.accuracy_score(Y_pred,Y_test)
     return 100*classification_accuracy


def confusion_matrix(Y_pred, Y_test):
     con_mat = sklearn.metrics.confusion_matrix(Y_pred, Y_test)
     return con_mat

def bayes_classifier_2(k,df):
     df_0 = df[df['class']==0]
     df_1 = df[df['class']==1]
     X_train0, X_test0, y_train0, y_test0 = train_test_data(df_0)
     X_train1, X_test1, y_train1, y_test1 = train_test_data(df_1)

     test = np.concatenate((X_test0, X_test1))
     pred = np.concatenate((y_test0, y_test1))

     gmm = GaussianMixture(n_components=k)
     gmm.fit(X_train0)

#After our model has converged, the weights, means, and covariancesshould be solved! We can print them out.

#    print("gmm mean_ ", gmm.means_)
     gmm2 = GaussianMixture(n_components=k)
     gmm2.fit(X_train1)

     print('for class 0')
     #print('means\n',gmm.means_);print('covariances\n',gmm.covariances_);print('weights\n',gmm.weights_)
     print('for class 1')
     #print('means\n',gmm2.means_);print('covariances\n',gmm2.covariances_);print('weights\n',gmm2.weights_)

     ypred = []
     for i in test:
         ypred.append(  0 if prob(i, gmm.weights_, gmm.means_,gmm.covariances_)\
                            > prob(i, gmm2.weights_, gmm2.means_,gmm2.covariances_) else 1 )
     print("Accuracy for GMM  Bayes Classifier: ")
     print(percentage_accuracy(pred, ypred))
     print(confusion_matrix(pred, ypred))
     print()

###################################################################################


###################################################################################

# Calculate model evaluation scores like
"""
- Accuracy
- Confusion Matrix
"""



###################################################################################


df=load_dataset("___path_to_file___")
a=df.columns
#print(df)


dfX_test,dfX_train,dfY_test,dfY_train=train_test_data(df)



dfh= pd.concat([dfX_train,dfY_train], axis=1)
dft= pd.concat([dfX_test,dfY_test],axis=1)
print("-------------------without PCA-----------")
Q=[2,4,8,16]
for i in Q:
    print("Q:",i)
    bayes_classifier_2(i,df)

print("bayes classification")
bay = bayes_classifier(dfX_train,dfY_train)
dfbY_pred = bay.predict(dfX_test)
con_mat = confusion_matrix(dfbY_pred, dfY_test)
print(con_mat)
print(percentage_accuracy(dfbY_pred, dfY_test))

#applying pca
print("APPLYING PCA")
from sklearn.decomposition import PCA
dataframe=df.copy()
N=[2,4,8,10,12,14,16,18]
for i in N:
    pca = PCA(n_components = i )
    c=df.columns
    X=dataframe[c[:19]]
    df1=pca.fit_transform(X)
    X_label=dataframe["class"]
    df1=pd.DataFrame(df1)
    df1["class"]=X_label
    print("N_COMPONENTS:",i)
    Q=[2,4,8,16]
    for j in Q:
        print("Q:",j)
        bayes_classifier_2(j,df1)