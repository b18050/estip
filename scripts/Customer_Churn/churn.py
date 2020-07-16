# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 20:03:05 2020

@author: chandan prakash
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metric
import pickle

df = pd.read_csv(r'telecom_churn.csv')
df.head()
df.columns
df = df.dropna(axis='columns', inplace=True)
df.head()
df.Partner.unique()
df.PaymentMethod.unique()
numeric_features = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity', 'OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn']
def to_numeric(s):
    if(s=="Yes"):
        return 1
    elif s=="No":
        return 0
    else:
        return -1
    
for feature in numeric_features:
    df[feature]=df[feature].apply(to_numeric)
    
df.head()
df.Contract.unique()

categorical_features = [
 'PhoneService',
 'MultipleLines',
 'InternetService',
 'OnlineSecurity',
 'OnlineBackup',
 'DeviceProtection',
 'TechSupport',
 'StreamingTV',
 'StreamingMovies',
 'Contract',
 'PaperlessBilling',
 'PaymentMethod']

pd.get_dummies(df, columns=categorical_features)

X = df.drop(labels='Churn',axis=1)
Y = df.Churn
print(X.shape,Y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
df.head()

features = ['Partner',
            'Dependents',
            'tenure',
            'PhoneService',
            'MultipleLines',
            'InternetService',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies',
            'Contract',
            'PaperlessBilling',
            'PaymentMethod',
            'Churn']

df = df[features]
df = df.dropna(axis='columns')
df['Churn'] = churn
model_columns = list(df.columns)
model_columns.remove('Churn')

pickle.dump(model_columns,open( 'model_columns.pkl','wb'))
 prediction_test = model.predict(X_test)
 print (metrics.accuracy_score(y_test, prediction_test))
 pickle.dump(lr, open('model.pkl','wb'))
 import random
 def random_bool():
    return random_number()

def random_number(low=0, high=1):
    return random.randint(low,high)

def generate_data():
    internetServices = ['DSL', 'Fiber optic', 'No']
    contracts = ['Month-to-month', 'One year', 'Two year']
    paymentMethods = ['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)']

    random_data = {
            'name':'customer',
            'Partner': random_bool(),
            'Dependents': random_bool(),
            'tenure': random_number(0,50),
            'PhoneService': random_bool(),
            'MultipleLines': random_number(-1),
            'InternetService': random.choice(internetServices),
            'OnlineSecurity': random_number(-1),
            'OnlineBackup': random_number(-1),
            'DeviceProtection': random_number(-1),
            'TechSupport': random_number(-1),
            'StreamingTV': random_number(-1),
            'StreamingMovies': random_number(-1),
            'Contract': random.choice(contracts),
            'PaperlessBilling': random_bool(),
            'PaymentMethod': random.choice(paymentMethods)
        }
    return random_data

random_user_data = generate_data()
query = pd.get_dummies(pd.DataFrame(random_user_data, index=[0]))
query = query.reindex(columns=model_columns, fill_value=0)
print(model.predict_proba(query))
prediction = round(model.predict_proba(query)[:,1][0], 2)* 100
print(prediction )