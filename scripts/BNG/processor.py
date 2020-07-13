# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 17:32:35 2020

@author: chandan prakash
"""

# Importing the libraries
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns

#Import Libraries
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r'group1.csv')
df = df.drop(['CreationTime'],axis=1)
#print (dataset)
#print (df.describe())

def heatMap():
    corr = df.corr()    
    sns.set(style="white")

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    #Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

def predictProcessor(auth_count,in_band , in_totalPPS):
    
    '''
    Polynomial regression on authenticate count, bandwidth, PPS  to predict memoryFree
    Input features - 
        1. "ActiveCount"  - number of subscribers 
        2. "Bandwidth"    - bandwidth required
        3. "TotalPPS"     - total packets per second required
    Output feature 
        1. "Processor(CPUUtil)"   - used by BNG device in GHz.
    '''
    
    # Load dataset required for model
    X = df.loc[:, ["ActiveCount","InBandwidth","InTotalPPS"]].values
    y = df["CPUUtil"]
    
    # Data splitting 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # build model
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(X_train)
    
    #trained model
    regressor = LinearRegression().fit(X_train, y_train)
    
    pickle.dump(regressor, open('memoryfree.pkl','wb'))
    
    # prediction of test case
    x = np.array([[auth_count, in_band , in_totalPPS]])
    
    #make data compaitible 
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    
    #prediction using model
    cpu = regressor.predict(x_poly)/10**8
    cpuinGHz = abs(round(memory[0],2))
    
    print("Memory free by BNG device is ",cpuinGHz," GB")
    
    return cpuinGHz

'''
    Linear regression on authenticate count, bandwidth, PPS  to predict memoryFree
    Input features - 
        1. "ActiveCount"  - number of subscribers 
        2. "Bandwidth"    - bandwidth required
        3. "TotalPPS"     - total packets per second required
    Output feature 
        1. "MemoryFree"   - used by BNG device(in GB).
'''
    
# Load dataset required for model
X = df.loc[:, ["ActiveCount","InBandwidth","InTotalPPS"]].values
y = df["CPUUtil"]
    
# Data splitting 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        
#trained model
regressor = LinearRegression().fit(X_train, y_train)
    
#save the file to the disk
pickle.dump(regressor, open('processor.pkl','wb'))
    
model = pickle.load(open('processor.pkl','rb'))
print(model.predict([[200,30000,400000]]))

heatMap()