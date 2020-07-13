"""
//--------------Data Preprocessing-----------//
"""

"""
//----------------Import the necessary libraries--------------------//
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

""" 
//-----------------Preprocessing of data --------------------------//
"""

""" 
//---------------Normalization of processed data -----------------//
""" 
def normalization(dataframe):
    rw={}
    cols=dataframe.columns
    for i in cols[:8]:
        min_max_scaler=preprocessing.MinMaxScaler()
        x=np.array(dataframe[i]).reshape(-1,1).astype(float)
        y=min_max_scaler.fit_transform(x)
        rw[i]=list(y)
    rw['class']=dataframe['class']
    d=pd.DataFrame(rw)
    return(d)

"""
//--------------Standardisation of processed data-------------------//
"""

def standardize(dataframe):
    cols=dataframe.columns
    rw={}
    for i in cols[:8]:
        l=np.array(dataframe[i]).astype(float)
        y=preprocessing.scale(l)
        rw[i]=y
    rw['class']=dataframe['class']
    df=pd.DataFrame(rw)
    return(df)

"""
//---------------Splitting the data--------------------------//
"""

# ----------- @ Normalised data split--------------------#
def my_train_test_split1(df):
    df0=df[df['class']==0]
    df1=df[df['class']==1]
    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(df0,df0['class'] , test_size=0.3,random_state=42,shuffle=True)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(df1,df1['class'], test_size=0.3,random_state=42,shuffle=True)
    df_conc_train = pd.concat([X_train0, X_train1], ignore_index=True)
    df_conc_test = pd.concat([X_test0, X_test1], ignore_index=True)
    df_conc_train.to_csv(r"___path_to_file___")
    df_conc_test.to_csv(r"___path_to_file___")

# -----------------------@ Standardised split data -----------------#
def my_train_test_split2(df):
    df0=df[df['class']==0]
    df1=df[df['class']==1]
    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(df0,df0['class'] , test_size=0.3,random_state=42,shuffle=True)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(df1,df1['class'], test_size=0.3,random_state=42,shuffle=True)
    df_conc_train = pd.concat([X_train0, X_train1], ignore_index=True)
    df_conc_test = pd.concat([X_test0, X_test1], ignore_index=True)
    df_conc_train.to_csv(r"___path_to_file___")
    df_conc_test.to_csv(r"___path_to_file___")

# -------------------- @  data is splitted and now concatenated--------------//

def my_train_test_split3(df):
    df0=df[df['class']==0]
    df1=df[df['class']==1]
    X_train0, X_test0, Y_train0, Y_test0 = train_test_split(df0,df0['class'] , test_size=0.3,random_state=42,shuffle=True)
    X_train1, X_test1, Y_train1, Y_test1 = train_test_split(df1,df1['class'], test_size=0.3,random_state=42,shuffle=True)
    df_conc_train = pd.concat([X_train0, X_train1], ignore_index=True)
    df_conc_test = pd.concat([X_test0, X_test1], ignore_index=True)
    df_conc_train.to_csv(r"___path_to_file___")
    df_conc_test.to_csv(r"___path_to_file___")

            

"""
// -----------------------Calcualtion of model evaluation score ---------------------- //


- Accuracy
- Confusion Matrix
"""

# def percentage_accuracy(function_parameters):
#     ...
#     ...
#     ...

# def confusion_matrix(function_parameters):
#     ...
#     ...
#     ...

"""--------------------- Main function--------------------"""

df=pd.read_csv(r"___path_to_file___")
df_norm=normalization(df)
df_znorm=standardize(df)
df_norm.to_csv(r"___path_to_file___")
df_znorm.to_csv(r"___path_to_file___")
 

""" ----------------- Splitting the data ----------------"""
my_train_test_split1(df)
my_train_test_split2(df_norm)
my_train_test_split3(df_znorm)


#train the data and predict

#original data
df_conc_train_ori=pd.read_csv(r"___path_to_file___")
df_conc_test_ori=pd.read_csv(r"___path_to_file___")










