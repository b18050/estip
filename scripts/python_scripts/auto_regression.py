"""
<---------------------Auto Regression ------------------------------>
"""
"""
//----------- IMporting Libraries--------------------//
"""
import pandas as pd
import pandas.plotting
from matplotlib import pyplot
from pandas.plotting import lag_plot
from pandas.plotting import autocorrelation_plot
from pandas import concat
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AR


def load_dataset(file_name):
    df=pd.read_csv(file_name)
    return df
    
def split(dataframe):
    X = dataframe.values
    train,test = X[1:len(X)-7],X[len(X)-7:]
    return(train,test)
    
def persistence_model(X):
    return(X)
    
def Autoregression(train,test):
    model=AR(train)
    model_fit=model.fit()
    predictions=model_fit.predict(start=len(train),end=len(train)+len(test)-1,dynamic=False)
    for i in range(len(predictions)):
        print('predicted=%f,expected=%f' %(predictions[i],test[i]))
    error=mean_squared_error(test,predictions)
    print('Test MSE: %.3f' %error)
    pyplot.plot(test)
    pyplot.plot(predictions,color='red')
    pyplot.show()
    
def corr(dataframe):
    #values=pd.DataFrame(dataframe['Temp'].values)
    #dataframe=concat([values.shift(1),values],axis=1)
    dataframe.columns=['t-1','t+1']
    result=dataframe.corr()
    print(result)

df=load_dataset(r"___path_to_file___")
df1=df.copy()
print(df.head())
df.plot()
pyplot.show()
lag_plot(df['memory'])
pyplot.show()
autocorrelation_plot(df['memory'])
pyplot.show()
plot_acf(df['memory'],lags=31)
pyplot.show()
values=pd.DataFrame(df['memory'].values)
df=concat([values.shift(1),values],axis=1)
corr(df)

train,test =split(df)
train_X,train_y = train[:,0],train[:,1]
test_X,test_y=test[:,0],test[:,1]
predictions=[]
for x in test_X:
    yhat=persistence_model(x)
    predictions.append(yhat)
test_score=mean_squared_error(test_y,predictions)
print("TEST MSE: %.3f" %test_score)
pyplot.plot(test_y)
pyplot.plot(predictions,color='red')
pyplot.show()
X=df1.values
train,test=X[1:len(X)-7],X[len(X)-7:]
Autoregression(train,test)