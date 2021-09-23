import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Support Vector Regression

def support_vector_regression(x_train,y_train,x_test,y_test):
    #1-Learning
    svr=SVR(kernel='rbf', C=100.0, gamma=0.0004, epsilon= 0.01)
    svr.fit(np.array(x_train),np.array(y_train))
    print("SVR is Applied")
    
    #print models intercept and coefficients
    print(f"Model coefficients : {svr.coef_}")
    
    #2-prediction
    y_pred=y_pred=svr.predict(x_test)
    y_pred2=y_pred=svr.predict(x_train)

    eval_df=pd.DataFrame({"Actual":y_test,"Predicted":y_pred})
    print(eval_df)

    #Accuracy
    print("---------------Training--------------------------------------")
    print(f"Accuracy for Training Stage:{r2_score(y_train,y_pred2)}")
    print(f"Mean Squared Error:{mean_squared_error(y_train,y_pred2)}")
    print(f"Mean Absolute Error:{mean_absolute_error(y_train,y_pred2)}")
    
    print("---------------Testing--------------------------------------")
    print(f"Accuracy for Testing Stage:{r2_score(y_test,y_pred)}")
    print(f"Mean Squared Error:{mean_squared_error(y_test,y_pred)}")
    print(f"Mean Absolute Error:{mean_absolute_error(y_test,y_pred)}")
    return y_pred,y_pred2
    
    
######################################################################################################
def Visualize_SVR(x_train,y_train,x_test,y_test,y_pred,y_pred2):
   fig=plt.figure()
   plt.scatter(x_train['Flight Duration'],y_train,color='blue',label='training_data')
   plt.scatter(x_train['Flight Duration'],y_pred2,color='green',label='Predicted_training_data')
   plt.scatter(x_test['Flight Duration'],y_test,color='red',label='testing_data')
   plt.scatter(x_test['Flight Duration'],y_pred,color='pink',label='Predicted_testing_data')
   plt.xlabel('Flight Duration')
   plt.ylabel('Delay')
   plt.xlim(-2, 10)
   plt.ylim(-2, 5)
   plt.legend(loc='best')
   plt.title('Support Vector Regression')


   plt.show()

