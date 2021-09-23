#Main Script for Preprepessing for data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, FactorAnalysis, FastICA
from sklearn.manifold import MDS, TSNE

#Preprocessing functions
def Read_Data(filepath):
    #read dataset
    train_data=pd.read_csv(filepath)
    #print(train_data.head())
    #print(train_data.describe())
    return train_data
#############################################################################################################
def plot_boxplot(df,ft):
        df.boxplot(column=[ft])
        plt.grid(False)
        plt.show()
#############################################################################################################
def Remove_Outliers(df):
    Q1=df['Delay'].quantile(0.25)
    Q3=df['Delay'].quantile(0.75)
    IQR=Q3-Q1
    df=df[~((df['Delay'] <(Q1-1.5*IQR))|(df['Delay'] > (Q3+1.5*IQR)))]
    plot_boxplot(df, "Delay")
    df=df[(df.Delay <6)  & (df['Delay'] >0 )]
    plot_boxplot(df, "Delay")
    print("Removing outliers is finshed ")
    return df

#############################################################################################################
def Sorting_Data(df):
    df.sort_values(by=["Scheduled arrival time"],inplace=True)
    print("Sorting is finshed ")
    return df

#############################################################################################################
def calFT(Arr,Dep):
    fds=[]
    for i,j in zip(Arr,Dep):
        fds.append(pd.Timedelta(pd.to_datetime(i)-pd.to_datetime(j)).seconds/60*60)
    return fds   
#############################################################################################################
#Data encoding: to convert categorical features to numberical
def ohe_new_features(df, features_name, encoder):
    for feature in features_name:
        name=feature+' New'
        df[name] = encoder.fit_transform(df[feature])
        df.drop(feature, axis=1, inplace=True)
        print("Encoding is finished for",feature)
    return df

#############################################################################################################
def Scaling_MinMax(df,df_flht):
    scaler=StandardScaler()
    scaler.fit(np.array(df))
    df=pd.DataFrame(scaler.transform(df),columns=df.columns)
    scaler.fit(np.array(df_flht))
    df_flht=pd.DataFrame(scaler.transform(df_flht),columns=df_flht.columns)
    print("Scaling is finished for",df)
    print("Scaling is finished for",df_flht)

    return df,df_flht

#############################################################################################################
def Scaling_Standard(df,df_flht):
    scaler=MinMaxScaler()
    scaler.fit(df)
    df=pd.DataFrame(scaler.transform(df),columns=df.columns)
    scaler.fit(df_flht)
    df_flht=pd.DataFrame(scaler.transform(df_flht),columns=df_flht.columns)
    print("Scaling is finished for",df)
    print("Scaling is finished for",df_flht)

    return df,df_flht

#############################################################################################################
def Select_Features(df,df_flht):
    new_df=df[['Depature Airport New','Destination Airport New','Delay New']].copy()
    new_df=pd.concat([new_df,df_flht],axis=1)
    return new_df

#############################################################################################################
# Split data to train and test
"""to split to train and test such that The data is split based on Scheduled departure time. 
The train data is all the data from year 2015 till 2017.
All the data samples collected in year 2018 are to be used as testing set.
so test_size=0.105359 to achieve that"""
def Splitting_Data(df):
    x_train,x_test,y_train,y_test=train_test_split(df.drop('Delay New', axis = 1),df['Delay New'],test_size=0.105359,shuffle=False)
    print('Splitting is Finished')
    return x_train,x_test,y_train,y_test

#############################################################################################################
#use PCA to reduce dataset to be 2D
def Reduce_df2D(data):
    dim_reducer=PCA(n_components=2)
    data_reduced=dim_reducer.fit_transform(data)
    return data_reduced
    
#############################################################################################################
#use PCA to reduce dataset to be 3D
def Reduce_df3D(data):
    dim_reducer=PCA(n_components=3)
    data_reduced=dim_reducer.fit_transform(data)
    return data_reduced

#############################################################################################################
def Visualize_df2D(data):
    fig=plt.figure()
    plt.scatter(data[:,0],data[:,1])
    plt.show()
#############################################################################################################
def Visualize_df3D(data):
    fig=plt.figure()
    ax=plt.axes(projection="3d")
    ax.scatter(data[:,0],data[:,1],data[:,2],marker='.')
    plt.show()
#############################################################################################################
