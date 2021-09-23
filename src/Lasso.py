import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Regularization using Lasso

def lasso_regular(x_train,y_train,x_test,y_test):
    lasso = Lasso()
    lasso.fit(x_train, y_train)

    #print models coefficients
    print("Lasso model Coefficients:",lasso.coef_)
    """
    x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,test_size=0.125,shuffle=False)

    alphas = [1,2, 3, 4, 5, 6, 7, 8, 9]
    losses = []
    for alpha in alphas:
        lasso=Lasso(alpha)
        lasso.fit(x_train,y_train)
        y_pred=lasso.predict(x_val)
        mse=mean_squared_error(y_pred,y_val)
        losses.append(mse)
    
    plt.plot(alphas, losses)
    plt.title("Lasso alpha value selection")
    plt.xlabel("alpha")
    plt.ylabel("Mean squared error")
    plt.show()

    best_alpha = alphas[np.argmin(losses)]
    print("Best value of alpha:", best_alpha)

    lasso = Lasso(best_alpha)
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    print("MSE on testset:", mean_squared_error(y_test, y_pred))
    """
    #2-prediction
    y_pred=lasso.predict(x_test)
    y_pred2=lasso.predict(x_train)

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
def Visualize_lasso(x_train,y_train,x_test,y_test,y_pred,y_pred2):
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
   plt.title('Lasso')


   plt.show()
