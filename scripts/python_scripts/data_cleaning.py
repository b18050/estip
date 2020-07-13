#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#load_data from csv file
df=pd.read_csv("__path_to_file__",na_values="na")

miss=df.isnull().sum()
#print(df.isnull().sum())

# print("total misiing values",sum(list(miss.values)))

# cleaning the data remove null values
l=[]
index=[]
for i in range(len(df)):
    m=df.iloc[i].isnull().sum()
    l.append(m)
    if(m>=6):
        index.append(i)

# Dictionary to store index of null values
dict={}

for i in range(0,12):
    dict[i+1]=l[i]

x=list(dict.keys())
y=list(dict.values())

#Graph Plot
plt.plot(x,y)
plt.xlabel("number of missing values")
plt.ylabel("number of tuples")
plt.show()

#count of null values
count=0
for i in range(5,13):
    if(dict[i]>0):
        count=count+dict[i]

# print("Total number of tuples having more than 50% of missing data:",count)

# drop null values
for i in range(len(index)):
    df=df.drop([index[i]])

# delete row with null values
delete_row_quality = df[df["quality"].isnull()].index
df = df.drop(delete_row_quality)

#print(df)
print("after deletion of tuples")

# cheecking if we have successfully removed null values
miss1=df.isnull().sum()
# print(miss1)
# print("total misiing values:",sum(list(miss1.values)))
