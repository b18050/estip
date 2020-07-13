# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import math

# load data from csv file
df=pd.read_csv("__path_to_file__",na_values="na")

# make copy of data_frame
df_copy=df.copy()

# check for null values
miss=df.isnull().sum()
# print(df.isnull().sum())

# print("total misiing values",sum(list(miss.values)))

# store index of null values
l=[]
index=[]
for i in range(len(df)):
    m=df.iloc[i].isnull().sum()
    l.append(m)
    if(m>4):
        index.append(i)

#Dictionary
dict={}
for i in range(0,8):
    dict[i+1]=l.count(i+1)

# key and value pair in dictionary key are null_index and value their count

x=list(dict.keys())
y=list(dict.values())

print(dict)

#Plot Graph 
plt.bar(x,y)
plt.grid()
plt.xlabel("number of missing values")
plt.ylabel("number of tuples")
plt.show()

# count of null values
count=0
for i in range(4,8):
    if(dict[i]>0):
        count=count+dict[i]


# print("Total number of tuples having more than 50% of missing data:",count)

# drop missing values
for i in range(len(index)):
    df=df.drop([index[i]])

# extract row with many null values
delete_row_quality = df[df["stationid"].isnull()].index
# print(len(delete_row_quality))

# delete the particular row
df = df.drop(delete_row_quality)
#print(df)

""" 
after deletion 
"""

# print("after deletion of tuples")
miss1=df.isnull().sum()
# print(miss1)
# print("total misiing values:",sum(list(miss1.values)))

#mean,median,mode

# making two copies 
df2=df.copy()
df3=df.copy()

# columns in dataframe
cols=list(df.columns)

#median calculation and filling
df[df.columns] = df[df.columns].apply(pd.to_numeric, errors='coerce')

# fill the missing values with median

df = df.fillna(df.median())  # works

# dropping first column as it is Date / Time
for i in df.columns[2:]:
    print('    ',i,'   ')
    print("mean ", np.mean(df[i]))
    print("median ", np.median(df[i]))
    print("mode " ,st.mode(np.array(df[i]))[0][0])
    plt.boxplot(df[i])
    plt.title(i)
    plt.show()



dfo=pd.read_csv("__path_to_file__",delimiter=",")

#using median
# print("after filling with median\n")

# List of columns in dataframe
cols=list(df.columns)

# dropping first column as it is Date / Time

# Calculate RMSE after filling with median
for i in cols[2:]:
    index1=df_copy[i].index[df_copy[i].apply(np.isnan)]
    med=df[i].median()
    #print(med)
    residue=0
    for j in index1:
        residue=(float(med)-float(dfo[i][j]))**2+residue
        rmse=residue/104.0
        rmse=math.sqrt(rmse)
    print("rmse",i,"==",rmse)



# print("---------------------")
# print("after filling with interploated data")

#--------------------------- fill  interpolate data-----------------------------#

#dataframe after deletion is df2

# fill null values with interpolated data 
df2 = df2.fillna(df2.interpolate())

# dropping first column as it is Date / Time
for i in df2.columns[2:]:
    print('    ',i,'   ')
    print("mean ", np.mean(df2[i]))
    print("median ", np.median(df2[i]))
    print("mode " ,st.mode(np.array(df2[i]))[0][0])
    plt.boxplot(df2[i])
    plt.title(i)
    plt.show()


# List of columns
cols=list(df2.columns)

# dropping first column as it is Date / Time
for i in cols[2:]:
    index2=df_copy[i].index[df_copy[i].apply(np.isnan)]
    residue=0
    for j in index2:
        residue=(df2[i][2]-float(dfo[i][j]))**2+residue
        rmse=residue/104.0
        rmse=math.sqrt(rmse)
    print("rmse",i,"==",rmse)


"""
df3 is filling with 0
"""

# print("after filling with 0")
df3.fillna(0,inplace=True)


# Plot Histogram
plt.hist(df3['temperature'])
plt.title("temperature:")
plt.show()

"""
df is after filling the median
"""
# print("after filling with median")

# Plot the Graph
plt.hist(df['temperature'])
plt.title('temperature')
plt.show()

"""
df2 is after filling with interplotion
"""

# print("after filling with interpolated data")
# plt.hist(df2['temperature'])
# plt.title('temperature')
# plt.show()

""""
groupby 
""""

date=[i for i in range(len(df))]
p1=df_copy.groupby('customer')
for i in p1:
    plt.plot(date,i[1]['sentiments'])
