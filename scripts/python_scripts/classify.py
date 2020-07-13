"""_____________________Classification_______________________"""

# Import the necessary libraries
import numpy as np
from collections import Counter
import scipy.stats as sta
import statistics as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB


"""
Try to use the functionalities of the libraries imported.
For example, rather than converting Pandas dataframe into
a list and then perform calculations, use methods of Pandas
library.
"""
###################################################################################


###################################################################################

# Import or Load the data
def load_dataset(file_name):
    df=pd.read_csv(file_name)
    return df

    """
    Load the dataset using this function and then return the
    dataframe. The function parameters can be as per the code
    and problem. Return the loaded data for preprocessing steps.
    """
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

def standardization(df,key_attribute):
    re=df.copy()
    re=(re-re.mean())/re.std()
    re[key_attribute]=df[key_attribute]
    re.to_csv("pima-indians-diabetes-Standardised.csv")
    return re

#def dimensionality_reduction(function_parameters):


#def shuffle(function_parameters):


def split(df):
    class_1=df[df["class"]==1]
    class_0=df[df["class"]==0]
    train1, test1 = train_test_split(class_1,test_size=0.3,random_state=42)
    train0, test0 = train_test_split(class_0,test_size=0.3,random_state=42)
    train=train1.append(train0)
    test=test1.append(test0)
    train.to_csv("diabetes_train.csv")
    test.to_csv("diabetes_test.csv")
    return train,test
    """
    Now your data is preprocessed and shuffle. It's time to divide it
    into training and test data.

    Example:
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
    ...                                 test_size=0.3, random_state=42)

            X: independent features
            Y: dependent features
            test_size: fraction of data to be splitted into test data
            random_state: a seed for random generator to produce same
                            "random" results each time the code is run.

    Now your data is ready for classification.
    """

###################################################################################


###################################################################################

# Perform classification

def classification(train,test,k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train.loc[:,"pregs":"Age"],train["class"])
    rf=knn.predict(test.loc[:,"pregs":"Age"])
    return rf
    """
    Pass the respective function parameters and perform classification.
    """

###################################################################################
def Gaussian(train,test):
    gnb=GaussianNB()
    gnb.fit(train.loc[:,"pregs":"Age"],train["class"])
    rf=gnb.predict(test.loc[:,"pregs":"Age"])
    return rf


# Calculate model evaluation scores like
"""
- Accuracy
- Confusion Matrix
"""

def percentage_accuracy(orig,pred):
    print(accuracy_score(orig,pred)*100,"%")
    return accuracy_score(orig,pred)*100

def confusion_matrix(orig,pred):
    print(pd.crosstab(orig, pred, rownames=['True'], colnames=['Predicted'], margins=True))
    ...
    ...
    ...

###################################################################################
k=[1, 3, 5, 7, 9, 11, 13, 15, 17, 21]
df=load_dataset(r"pima-indians-diabetes.csv")

standardized_df=standardization(df,"class")
l1=[]
l2=[]
train, test = split(standardized_df)
for i in k:
    print("K = ",i)
    rf=classification(train,test,i)
    confusion_matrix(test["class"],rf)
    l1.append(percentage_accuracy(test["class"],rf))



plt.scatter(k,l1,color="red",label="Original")

for i in k:
    print("K = ",i)
    rf=Gaussian(train,test)
    confusion_matrix(test["class"],rf)
    l2.append(percentage_accuracy(test["class"],rf))

plt.scatter(k,l2,color="green",label="Original")
