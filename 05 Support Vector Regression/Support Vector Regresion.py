import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
print(dataset)

X=dataset.iloc[:,  1:2].values
y=dataset.iloc[:,-1].values

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  PolynomialFeatures

#fitting Linear Regression to the dataset
lin_reg = LinearRegression()
lin_reg.fit(X,y)



#Visualising The SVR Model
plt.scatter(X,y,color ='red')
plt.plot(X,lin_reg.predict(X),color='blue')

plt.title ("Truth or Bluff (Linear Regression)")
plt.xlabel('Position level')
plt.ylabel('salary')


#for i in rang(2,5):
#fitting polynomial Regression to the dataset

poly_reg = PolynomialFeatures(degree =i )
X_poly = poly_reg.fit_transform(X)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(X_poly,y)
    
    
    
    
    #Visualising The Polynomial Regresssion Model
plt.scatter(X,y,color ='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)),color='blue')
    
plt.title ("Truth or Bluff (Polynomial Regression)")
plt.xlabel('Position level')
plt.ylabel('salary')
    
plt.show()