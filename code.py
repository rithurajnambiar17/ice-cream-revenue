# Linear Regression using Normal Equation
# IceCream Revenue v/s Temperature

#Importing libraries
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#Importing data
data = pd.read_csv('./data/IceCreamData.csv')

#Spliting the data into two variables for validation
train = data[:250]
test = data[250:]

#Using normal equation to find the value of theta
x=train.Temperature
y=train.Revenue
x=np.c_[np.ones((250,1)), x]
theta = np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)

#Using theta value that we calculated earlier to predict data from test set
x_new = np.array([[18.578119],[32.334808]]) #Here we used the first two elements from test set above
x_new = np.c_[np.ones((2,1)), x_new]
x_predict = x_new.dot(theta)

#Predicting the entire test set
x_new = np.array(test.Temperature)
x_new = np.c_[np.ones((250,1)), x_new]
x_predict = x_new.dot(theta)
predict = pd.DataFrame(x_predict)
true = pd.DataFrame(test.Revenue)

#Calculation r2 score using r2_score from sklearn
from sklearn.metrics import r2_score
score = r2_score(true, predict)
print("R2 Score: ", score)