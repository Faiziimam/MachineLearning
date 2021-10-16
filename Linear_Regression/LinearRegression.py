import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 


Dataset=pd.read_csv('Linear_Regression/Salary_Data.csv')
x=Dataset.iloc[:,:-1].values
y=Dataset.iloc[:,-1].values

# Spliting Training and Test Set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

# predicting Training Set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
# Test Set
regressor.predict(x_test)
# Visualising Training ;;
plt.scatter(x_train,y_train,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Salary & Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
#Visiualising Test Set
plt.scatter(x_test,y_test,color="red")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title(" T-> Salary & Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()