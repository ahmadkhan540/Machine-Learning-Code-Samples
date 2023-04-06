import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("F:/UET/SEMESTER 6/ML/Assignment 1/canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]

# Making rank 2 arrays/
X = np.array(X)
Y = np.array(Y)
X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

# Scaling for X
min_X = X.min()
max_X = X.max()
t1 = X - min_X
t2 = max_X - min_X
x_scale = t1 / t2

# Scaling for Y
min_Y = Y.min()
max_Y = Y.max()
t1_y = Y - min_Y
t2_y = max_Y - min_Y
y_scale = t1_y / t2_y


m,col = x_scale.shape
ones = np.ones((m,1))
x_scale = np.hstack((ones,x_scale))

theta = np.zeros((2,1))

iterations = 10000
alpha = 0.01

# Defining Cost function

def Get_cost_J(X,Y,Theta):
    Pridictions = np.dot(X,Theta)
    Error = Pridictions-Y
    SqrError = np.power(Error,2)
    SumSqrError = np.sum(SqrError)
    J  = (1/2*m)*SumSqrError # Where m is tototal number of rows
    return J

#Defining Gradient Decent Algorithm

def Gradient_Decent_Algo(X,Y,Theta,alpha,itrations,m):
    histroy = np.zeros((itrations,1))
    for i in range(itrations):
        temp =(np.dot(X,Theta))-Y
        temp = (np.dot(X.T,temp))*alpha/m
        Theta = Theta - temp
        histroy[i] = Get_cost_J(X, Y, Theta)
       
    return (histroy,Theta)

h,t = Gradient_Decent_Algo(x_scale,y_scale,theta,alpha,iterations,m)

predictions = np.dot(x_scale,t)

plt.scatter(x_scale[:,1],y_scale)
plt.plot(x_scale[:,1],predictions, color="red")
plt.show()

th0=t[0,0]
th1=t[1,0]

t1 = 2020 - min_X
t2 = max_X - min_X
x_scale = t1 / t2

predict_s = th0+(th1*x_scale)

new_predict = predict_s*(max_Y-min_Y)+min_Y
print(new_predict)

