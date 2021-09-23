import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#Ploynomial Regression
"""
def Poly_reg(x_train,y_train,x_test):
    
    #1-Learning
    degrees = [1, 2, 6]
    Accs=[]
    for i in range(len(degrees)):
        poly=PolynomialFeatures(degree=degrees[i])
        x_poly=poly.fit_transform(x_train)
        poly.fit(x_poly,y_train)
        regressor = LinearRegression()
        regressor.fit(x_poly, y_train)
        print(f"Model intercept : {regressor.intercept_}")
        print(f"Model coefficients : {regressor.coef_}")
    
        #2-prediction
        y_pred=regressor.predict(poly.fit_transform(x_test))
    
        #Accuracy
        print(f"Accuracy:{r2_score(y_test,y_pred)}",degrees[i])
        Acc=r2_score(y_test,y_pred)
        Accs.append(Acc)

    SortedAcc=np.sort(np.array(Accs))
    return SortedAcc[-1]


print(f"Acurracy: {Poly_reg(x_train, y_train, x_test)}")

"""
def polynomial_regression(x_train,y_train,x_test,y_test):
    #1-Learning
    #degree=2 the largest r2 score

    poly=PolynomialFeatures(degree=2)
    x_poly=poly.fit_transform(x_train)
    poly.fit(x_poly,y_train)
    regressor = LinearRegression()
    regressor.fit(x_poly, y_train)
    
    #print models intercept and coefficients
    print(f"Model intercept : {regressor.intercept_}")
    print(f"Model coefficients : {regressor.coef_}")
    
    #2-prediction
    y_pred=regressor.predict(poly.fit_transform(x_test))
    y_pred2=regressor.predict(poly.fit_transform(x_train))

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
def Visualize_polynomial_regression(x_train,y_train,x_test,y_test,y_pred,y_pred2):
   fig=plt.figure()
   plt.scatter(x_train['Flight Duration'],y_train,color='blue',label='training_data')
   plt.scatter(x_train['Flight Duration'],y_pred2,color='green',label='Predicted_training_data')
   plt.scatter(x_test['Flight Duration'],y_test,color='red',label='testing_data')
   plt.scatter(x_test['Flight Duration'],y_pred,color='pink',label='Predicted_testing_data')
   plt.xlabel('Flight Duration')
   plt.ylabel('Delay')
   plt.xlim(0,2)
   plt.ylim(0,2)
   plt.legend(loc='best')
   plt.title('Polynomial Regression: Degree 2')


   plt.show()
