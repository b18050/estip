# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import seaborn as sns

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

#heatMap()

#Import Libraries
from sklearn.linear_model import LinearRegression  
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def predictMemoryUsed(auth_count,in_band , in_totalPPS):
    
    ''' 
    predict Memory Used By BNG device using Polynomial Regression
    Input features - 
        1. "ActiveCount"  - number of subscribers 
        2. "Bandwidth"    - bandwidth required
        3. "TotalPPS"     - total packets per second required
    Output feature 
        1. "MemoryUsed"   - used by BNG device(in GB).
    '''
    
    # Load dataset required for model
    X = df.loc[:, ["ActiveCount","InBandwidth","InTotalPPS"]].values
    y = df["MemoryUsed"]
    
    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # build model
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(X_train)
    
    #trained model
    model = LinearRegression().fit(x_poly, y_train)
    
    pickle.dump(model, open('modelUsed.pkl','wb'))
    
    # prediction of test case
    x = np.array([[auth_count, in_band , in_totalPPS]])
    
    #make data compaitible 
    polynomial_features= PolynomialFeatures(degree=2)
    x_poly = polynomial_features.fit_transform(x)
    
    #prediction using model
    
    memory = model.predict(x_poly)/10**8
    memoryinGB = abs(round(memory[0],2))
    
    #print("Memory used by BNG device is ",memoryinGB," GB")
    
    return memoryinGB




def predictMemoryFree(auth_count,in_band , in_totalPPS):
    
    '''
    Polynomial regression on authenticate count, bandwidth, PPS  to predict memoryFree
    Input features - 
        1. "ActiveCount"  - number of subscribers 
        2. "Bandwidth"    - bandwidth required
        3. "TotalPPS"     - total packets per second required
    Output feature 
        1. "MemoryFree"   - used by BNG device(in GB).
    '''
    
    # Load dataset required for model
    X = df.loc[:, ["ActiveCount","InBandwidth","InTotalPPS"]].values
    y = df["MemoryFree"]
    
    # Data splitting 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    # build model
    # polynomial_features= PolynomialFeatures(degree=2)
    #x_poly = polynomial_features.fit_transform(X_train)
    
    #trained model
    regressor = LinearRegression().fit(X_train, y_train)
    
    pickle.dump(regressor, open('modelfree.pkl','wb'))
    
    # prediction of test case
    #x = np.array([[auth_count, in_band , in_totalPPS]])
    
    #make data compaitible 
    #polynomial_features= PolynomialFeatures(degree=2)
    #x_poly = polynomial_features.fit_transform(x)
    
    #prediction using model
    #memory = regressor.predict(x_poly)/10**8
    #memoryinGB = abs(round(memory[0],2))
    
    #print("Memory free by BNG device is ",memoryinGB," GB")
    
    #return memoryinGB



def predictMemory(auth_count, in_band , in_totalPPS):
    
    ''' 
    predict Total Memory required By BNG device
    with given specifications
    
    Input features - 
        1. "ActiveCount"  - number of subscribers 
        2. "Bandwidth"    - bandwidth required
        3. "TotalPPS"     - total packets per second required
        
    Output feature 
        1. "TotalMemory"   - required by BNG device(in GB).
    '''
    
    # memory Used 
    memoryUsed = predictMemoryUsed(auth_count, in_band , in_totalPPS)
    #memoeryFree
    memoryFree = predictMemoryFree(auth_count, in_band , in_totalPPS)
    
    return memoryUsed + memoryFree

#m = predictMemoryFree(200,3000,344400)
#print(m)
model = pickle.load(open('modelfree.pkl','rb'))
print(model)
print(model.predict([[200,30000,400000]]))