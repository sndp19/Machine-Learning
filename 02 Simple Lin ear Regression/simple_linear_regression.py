# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 13:25:01 2021

@author: Sndp
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Importing the dataset
dataset =pd.read_csv('Salary_Data.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,1].values

#Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

#fitting simple Linear Regression to The Training set

from sklearn.linear_model import LinearRegression
regressor  = LinearRegression()
regressor.fit(X_train,y_train)


y_pred = regressor.predict(X_test)

plt.scatter(X_test,y_test,color ='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()