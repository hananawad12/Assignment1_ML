#Main Script for Calling the Models
from sklearn.preprocessing import LabelEncoder
from Preprocessing import *
from MultipleLinearRegression import *
from PolynomialRegression import *
from Lasso import *
from SupportVectorRegression import *
from sys import exit


def preprocess1():
    #reading
    train_data=Read_Data('../Data/flight_delay.csv')
    print(train_data)
    #-----------------------------------------------------------------------------------
    #Removing Outliers
    plot_boxplot(train_data, "Delay")
    train_data=Remove_Outliers(train_data)
    #-----------------------------------------------------------------------------------
    
    #Calculate Flight Duration in minutes and create dataframe for it
    train_data=Sorting_Data(train_data)
    new_cols=calFT(train_data['Scheduled arrival time'],train_data['Scheduled depature time'])
    df_flight=pd.DataFrame(new_cols,columns=['Flight Duration'])
    print(df_flight)
    #-----------------------------------------------------------------------------------

    #Encoding
    print(train_data.head())
    encoder = LabelEncoder()
    f_names = ['Depature Airport', 'Destination Airport', 'Scheduled depature time', 'Scheduled arrival time','Delay']
    train_data = ohe_new_features(train_data, f_names, encoder)
    print("-------train_data After Encoding--------------")
    print(train_data.head(10))
    #-----------------------------------------------------------------------------------

    #Scaling
    train_data,df_flight=Scaling_MinMax(train_data,df_flight)
    print("-------train_data After Scaling--------------")
    print(train_data.head(10))
    print("-------Flight Duration After Scaling--------------")
    print(df_flight.head(10))
    #-----------------------------------------------------------------------------------

    #Selecting features
    new_df=Select_Features(train_data,df_flight)
    print("-------New Data Frame After Selecting--------------")
    print(new_df.head(10))
    #-----------------------------------------------------------------------------------
    
    #splitting
    """
    Before Splitting: (122428, 4)
    After Splitting
    (109529, 3)
    (12899, 3)
    """
    print("Berfore Splitting: ",new_df.shape)
    x_train,x_test,y_train,y_test=Splitting_Data(new_df)
    print("-------After Splitting--------------")
    print("x_train: ",x_train.shape)   #109529
    print("x_test: ",x_test.shape)    #12899
    #-----------------------------------------------------------------------------------
    
    #Data Visualization
    #2D
    x_train_reduced2D=Reduce_df2D(x_train)
    Visualize_df2D(x_train_reduced2D)
    
    x_test_reduced2D=Reduce_df2D(x_test)
    Visualize_df2D(x_test_reduced2D)
    #3D
    x_train_reduced3D=Reduce_df3D(x_train)
    Visualize_df3D(x_train_reduced3D)
    
    x_test_reduced3D=Reduce_df3D(x_test)
    Visualize_df3D(x_test_reduced3D)
    #-----------------------------------------------------------------------------------
    return x_train,x_test,y_train,y_test
    
def preprocess2():
    #reading
    train_data=Read_Data('../Data/flight_delay.csv')
    print(train_data)
    #-----------------------------------------------------------------------------------
    #Removing Outliers
    plot_boxplot(train_data, "Delay")
    train_data=Remove_Outliers(train_data)
    #-----------------------------------------------------------------------------------
    
    #Calculate Flight Duration in minutes and create dataframe for it
    train_data=Sorting_Data(train_data)
    new_cols=calFT(train_data['Scheduled arrival time'],train_data['Scheduled depature time'])
    df_flight=pd.DataFrame(new_cols,columns=['Flight Duration'])
    print(df_flight)
    #-----------------------------------------------------------------------------------

    #Encoding
    print(train_data.head())
    encoder = LabelEncoder()
    f_names = ['Depature Airport', 'Destination Airport', 'Scheduled depature time', 'Scheduled arrival time','Delay']
    train_data = ohe_new_features(train_data, f_names, encoder)
    print("-------train_data After Encoding--------------")
    print(train_data.head(10))
    #-----------------------------------------------------------------------------------

    #Scaling
    train_data,df_flight=Scaling_Standard(train_data,df_flight)
    print("-------train_data After Scaling--------------")
    print(train_data.head(10))
    print("-------Flight Duration After Scaling--------------")
    print(df_flight.head(10))
    #-----------------------------------------------------------------------------------

    #Selecting features
    new_df=Select_Features(train_data,df_flight)
    print("-------New Data Frame After Selecting--------------")
    print(new_df.head(10))
    #-----------------------------------------------------------------------------------
    
    #splitting
    """
    Before Splitting: (122428, 4)
    After Splitting
    (109529, 3)
    (12899, 3)
    """
    print("Berfore Splitting: ",new_df.shape)
    x_train,x_test,y_train,y_test=Splitting_Data(new_df)
    print("-------After Splitting--------------")
    print("x_train: ",x_train.shape)   #109529
    print("x_test: ",x_test.shape)    #12899
    #-----------------------------------------------------------------------------------
    
    #Data Visualization
    #2D
    x_train_reduced2D=Reduce_df2D(x_train)
    Visualize_df2D(x_train_reduced2D)
    
    x_test_reduced2D=Reduce_df2D(x_test)
    Visualize_df2D(x_test_reduced2D)
    #3D
    x_train_reduced3D=Reduce_df3D(x_train)
    Visualize_df3D(x_train_reduced3D)
    
    x_test_reduced3D=Reduce_df3D(x_test)
    Visualize_df3D(x_test_reduced3D)
    #-----------------------------------------------------------------------------------
    return x_train,x_test,y_train,y_test

#############################################################################################################
#Calling the Machine Learning Models

try:
    n=int(input("""Enter the Model number to execute it ( 1==>Mutilple Linear Regression
                                      2==>Polynomial Regression
                                      3==>Regularization using Lasso
                                      4==>Support Vector Regression)
                                      (1,2,3,4) : """))
except ValueError:
    print("Enter the integer number from 1 to 4 only!!!")
    exit()
#-------------------------------------------------------------------------------------   
#Multiple Linear Regression 
if n==1 and __name__=="__main__":
    x_train,x_test,y_train,y_test=preprocess2()
    y_pred,y_pred2=linear_regression(x_train,y_train,x_test,y_test)
    Visualize_linear_regression(x_train,y_train,x_test,y_test,y_pred,y_pred2)
#-------------------------------------------------------------------------------------
#Polynomial Regression 
if n==2 and __name__=="__main__":
    x_train,x_test,y_train,y_test=preprocess2()
    y_pred,y_pred2=polynomial_regression(x_train,y_train,x_test,y_test)
    Visualize_polynomial_regression(x_train,y_train,x_test,y_test,y_pred,y_pred2)
#-------------------------------------------------------------------------------------
#Lasso
if n==3 and __name__=="__main__":
    x_train,x_test,y_train,y_test=preprocess2()
    y_pred,y_pred2=lasso_regular(x_train,y_train,x_test,y_test)
    Visualize_lasso(x_train,y_train,x_test,y_test,y_pred,y_pred2)      
#------------------------------------------------------------------------------------- 
#Support Vector Regression 
if n==4 and __name__=="__main__":
    x_train,x_test,y_train,y_test=preprocess2()
    y_pred,y_pred2=support_vector_regression(x_train,y_train,x_test,y_test)
    Visualize_SVR(x_train,y_train,x_test,y_test,y_pred,y_pred2)
  
