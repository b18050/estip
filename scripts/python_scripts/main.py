"""
//------------ Linear Regression-----------------//
//------------ Polynomial Regression-------------//
//------------ Visualisation of Actual and Predicted Data-------------//
"""

"""
//-----------------Import Libraries------------------//
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
import scipy

"""
// ------------------ Data Split --------------------- //
"""

def split(dfn):
    X_train, X_test, y_train, y_test = train_test_split(dfn[df.columns[:len(df.columns)-1]], dfn[df.columns[len(df.columns)-1]] , test_size = 0.3, random_state = 0) 
    return X_train, X_test, y_train, y_test

"""
// ---------load data as csv from file  ---------------//
"""
df = pd.read_csv("__path_to_file___")


## ----------- data split 70-30 ----------------- ##
X_train, X_test, y_train, y_test = split(df)

## --------------- customer data csv ----------- ##
train = X_train.copy()
train["price"] = y_train
train.to_csv('sentiment.csv') 

## ----------- bng data csv -------------- ##
test = X_test.copy()
test["comments"] = y_test
test.to_csv('bng.csv') 

"""
Visulaise using scatter Plot for the graph
"""
plt.scatter(X_train["sentiment"],y_train)
att1 = "comments"
xt = np.linspace( X_train[att1].mean()-4*X_train[att1].std() , X_train[att1].mean()+4*X_train[att1].std() , 10)
xtp = np.reshape(xt, (len(xt), 1))
x=np.reshape(np.array(X_train["comments"]), (len(np.array(X_train["comments"])), 1))

"""
//------------Linear Regression ---------------//
"""
reg = linear_model.LinearRegression() 
reg.fit(x,y_train)
y_pred = reg.predict(xtp)
plt.plot(xt,y_pred,c="red")
plt.xlabel("price")
plt.ylabel("CPUUtil")
plt.title("best fit line")
plt.show()

"""
// ----------- LInear Model fit to predict memory requirements----------//
"""
xtr=np.reshape(np.array(X_train["price"]), (len(np.array(X_train["price"])), 1))
xts=np.reshape(np.array(X_test["price"]), (len(np.array(X_test["price"])), 1))
reg = linear_model.LinearRegression() 
reg.fit(xtr, y_train) 
train_pred = reg.predict(xtr)
test_pred = reg.predict(xts)

"""
// ---------------Model Validation using RMSE(root mean squared error)---------//
"""
print("rmse (simple lr) for train data =",(np.sum((train_pred-y_train)**(2))/len(y_train))**(0.5))
print("rmse (simple lr) for test data =",(np.sum((test_pred-y_test)**(2))/len(y_test))**(0.5))

"""
//---------------Visulaisation of actual and predicted value-------------------//
"""
plt.scatter(y_test,test_pred)
plt.xlabel("actual quality")
plt.ylabel("predicted quality")
plt.title("scatter plot on test data predictions (simple lr)")
plt.show()

"""
//---------------Polynomial Regression-------------------//
"""
plt.scatter(X_train["price"],y_train)
att1 = "price"
xt = np.linspace( X_train[att1].mean()-4*X_train[att1].std() , X_train[att1].mean()+4*X_train[att1].std() , 10)
xt=np.reshape(xt, (len(xt), 1))
polynomial_features= PolynomialFeatures(degree=3)
x_poly = polynomial_features.fit_transform(x)
xtp = polynomial_features.fit_transform(xt)
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_train)
y_pred = regressor.predict(xtp)

"""
// ------------------ Visualisation of Actual and Predicted Data --------------------//
"""
plt.plot(xt,y_pred,c="InPPS")
plt.xlabel("memory")
plt.ylabel("price")
plt.title("best fit curve")
plt.show()

"""
//---------------Polynomial Regression with varying values of p ------------------//
"""
print("\nfor traning data (simple lr)\n\n")
xtr=np.reshape(np.array(X_train["price"]), (len(np.array(X_train["price"])), 1))
rmse=[]

"""
//-------------------------p = [ 1 , 2 , 3 , 4 , 5]------------------------------//
"""
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(xtr)
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_train)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_train)**(2))/len(y_train))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])
    
"""
//----------------------- Visualisation of Train Data -----------------------------//
"""
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("accuracy")
plt.title("for traning data")
plt.show()

"""
//-----------------------Model fitting on test data----------------------//
"""
print("\n\nfor test data (simple lr)\n")
xts=np.reshape(np.array(X_test["pH"]), (len(np.array(X_test["pH"])), 1))
rmse=[]
"""
//-------------------------p = [ 1 , 2 , 3 , 4 , 5]------------------------------//
"""
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(xts)
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_test)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_test)**(2))/len(y_test))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])


"""
//----------------------- Visualisation of Train Data -----------------------------//
"""   
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("accuracy")
plt.title("for test data")
plt.show()


"""
//-----------------------Validation of Polynomial regressor model---------------//
"""
print("\nrmse is almost same at every value of p\n")


"""
//------------Polynomial Regression on optimized value of p above calculated-----------------//
"""
"""// -----------------  p = 4 ------------------------"""
polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(xts)
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_test)
y_pred = regressor.predict(x_poly)

"""
//---------------Visualisation of test data -------------------//
"""
plt.scatter(y_test,y_pred)
plt.xlabel("actual quality")
plt.ylabel("predicted quality")
plt.title("scatter plot on test data predictions when p = 4")
plt.show()



"""
// --------------- RMSE (root mean squared error-----------------//
"""
reg.fit(X_train, y_train) 
train_pred = reg.predict(X_train)
test_pred = reg.predict(X_test)
print("rmse (multiple lr) for train data =",(np.sum((train_pred-y_train)**(2))/len(y_train))**(0.5))
print("rmse (multiple lr) for test data =",(np.sum((test_pred-y_test)**(2))/len(y_test))**(0.5))

"""
// ---------------- Visualisation of actual and predicted value------------//
"""
plt.scatter(y_test,test_pred)
plt.xlabel("actual quality")
plt.ylabel("predicted quality")
plt.title("scatter plot on test data predictions (multiple lr)")
plt.show()


print("\nfor traning data (multiple lr)\n\n")

"""
// ==================== RMSE for different values of 'p'(degree) in Polynomial Regression on train data==============//
"""
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(X_train)
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_train)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_train)**(2))/len(y_train))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])

"""
//------------------------- Visualisation of Data---------------------------//
"""
    
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("accuracy")
plt.title("for traning data")
plt.show()

"""
// ==================== RMSE for different values of 'p'(degree) in Polynomial Regression on test data==============//
"""
print("\n\nfor test data (multiple lr)\n")
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(X_test)
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_test)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_test)**(2))/len(y_test))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])

"""
//------------------------- Visualisation of Data---------------------------//
"""
    
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("accuracy")
plt.title("for test data")
plt.show()

print("\nrmse is least at p = 5\n")

"""
//------------Polynomial Regression on optimized value of p above calculated-----------------//
"""
"""// -----------------  p = 5 for test data ------------------------"""
polynomial_features= PolynomialFeatures(degree=5)
x_poly = polynomial_features.fit_transform(X_test)
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_test)
y_pred = regressor.predict(x_poly)

"""
//------------Visualisation of Actual and Predicted Data on Test data-----------------//
"""
"""// -----------------  p = 5  ------------------------"""
plt.scatter(y_test,y_pred)
plt.xlabel("actual quality")
plt.ylabel("predicted quality")
plt.title("scatter plot on test data predictions when p = 5")
plt.show()


"""
//------------ Pearson correlation coefficient between each pair of columns----------------//
"""
columns = X_train.columns
pcorrcoef = []
for i in columns:
    pcorrcoef.append([i,scipy.stats.pearsonr(X_train[i],y_train)[0]])
print("\nattribute alcohol and volatile acidity is most correlated with quality\n")

att1 = "memory"
att2 = "CPUUtil"

"""
//----------------- Linear Regression without using library function------------------------//
"""
reg = linear_model.LinearRegression() 
reg.fit(X_train[[att1,att2]], y_train)
coef =  reg.coef_
incpt = reg.intercept_ 
a,b,c,d=coef[0],coef[1],-1,-incpt
x = np.linspace( X_train[att1].mean()-2*X_train[att1].std() , X_train[att1].mean()+2*X_train[att1].std() , 10)
y = np.linspace( X_train[att2].mean()-2*X_train[att2].std() , X_train[att2].mean()+2*X_train[att2].std() , 10)
X,Y = np.meshgrid(x,y)
Xi = np.array([X.ravel().tolist(),Y.ravel().tolist()])
#Z = (d - a*X - b*Y) / c
Z = reg.predict(Xi.T)
Z = Z.reshape(X.shape)
fig = plt.figure()
ax = fig.gca(projection='3d')

"""
// -------------------- 3-D Projection--------------------------------------//
"""
ax.scatter(X_train[att1][:200] ,X_train[att2][:200], y_train[:200])
ax.plot_surface(X, Y, Z,color="red")
ax.set_xlabel(att1)
ax.set_ylabel(att2)
ax.set_zlabel('memory')
plt.show()


"""
//----------Linear Regression on two most related to it-------------------//
"""
reg = linear_model.LinearRegression() 
reg.fit(X_train[["subscribers","InPps"]], y_train) 
train_pred = reg.predict(X_train[["subscribers","InPps"]])
test_pred = reg.predict(X_test[["subscribers","InPps"]])
print("rmse (two most correlated attribute simple lr) for train data =",(np.sum((train_pred-y_train)**(2))/len(y_train))**(0.5))
print("rmse (two most correlated attribute simple lr) for test data =",(np.sum((test_pred-y_test)**(2))/len(y_test))**(0.5))

plt.scatter(y_test,test_pred)
plt.xlabel("actual quality")
plt.ylabel("predicted quality")
plt.title("scatter plot on test data predictions (two most correlated attribute simple lr)")
plt.show()

"""
//-------------------Polynomial regression on two most related columns for train data---------------//
"""

print("\nfor traning data (two most correlated attribute multiple lr)\n\n")
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(X_train[["subscribers","InPps"]])
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_train)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_train)**(2))/len(y_train))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])

"""
//------------------- Data Visualisation----------------------//
""" 
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("accuracy")
plt.title("for traning data")
plt.show()

"""
//-------------------Polynomial regression on two most related columns for test data---------------//
"""

print("\n\nfor test data (two most correlated attribute multiple lr)\n")
rmse=[]
for p in [2,3,4,5]:
    polynomial_features= PolynomialFeatures(degree=p)
    x_poly = polynomial_features.fit_transform(X_test[["subscribers","InPps"]])
    regressor = linear_model.LinearRegression()
    regressor.fit(x_poly, y_test)
    y_pred = regressor.predict(x_poly)
    rmse.append((np.sum((y_pred-y_test)**(2))/len(y_test))**(0.5))
    print("rmse for p =",p,": ",rmse[len(rmse)-1])

"""
//------------------- Data Visualisation----------------------//
"""     
plt.bar([2,3,4,5],rmse)
plt.xlabel("value of p")
plt.ylabel("accuracy")
plt.title("for test data")
plt.show()

print("\nrmse is almost same at every value of p\n")

"""
//-------------------Polynomial regression on two most related columns for test data---------------//
"""
"""//-------------- p  = 4 -------------------------// """
polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(X_test[["subscribers","InPps"]])
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_test)
y_pred = regressor.predict(x_poly)
plt.scatter(y_test,y_pred)
plt.xlabel("subscribers")
plt.ylabel("InPps")
plt.title("scatter plot on test data predictions when p = 4 (two most correlated attribute multiple lr)")
plt.show()



att1 = "InPps"
att2 = "subscribers"

polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(X_train[["subscribers","InPps"]])
regressor = linear_model.LinearRegression()
regressor.fit(x_poly, y_train)
coef =  regressor.coef_
incpt = regressor.intercept_ 


xc = np.linspace( X_train[att1].mean()-2*X_train[att1].std() , X_train[att1].mean()+2*X_train[att1].std() , 10)
yc = np.linspace( X_train[att2].mean()-2*X_train[att2].std() , X_train[att2].mean()+2*X_train[att2].std() , 10)

X,Y = np.meshgrid(xc,yc)
Xi = np.array([X.ravel().tolist(),Y.ravel().tolist()])

polynomial_features= PolynomialFeatures(degree=4)
x_poly = polynomial_features.fit_transform(Xi.T)
#Z=np.sum(x_poly*coef,axis=1)+incpt
Z = regressor.predict(x_poly)
Z = Z.reshape(X.shape)
fig = plt.figure()

"""
// ------------------ 3 - D Projection --------------------//
"""
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z,color="red")
ax.scatter(X_train[att1][:300] ,X_train[att2][:300], y_train[:300])
ax.set_xlabel(att1)
ax.set_ylabel(att2)
ax.set_zlabel('quality')
print(Z.min(),Z.max())
plt.show()









