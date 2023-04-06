import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("F:/UET/SEMESTER 6/ML/Assignment 1/hiring.csv")
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0:3]
Y = CSV_Data.iloc[:,3]

# Making rank 2 arrays/
Y = np.array(Y)
Y = Y[:,np.newaxis]

m,col = X.shape
ones = np.ones((m,1))
X = np.hstack((ones,X))

theta = np.zeros((4,1))

iterations = 5000
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

h,t = Gradient_Decent_Algo(X,Y,theta,alpha,iterations,m)


th0 = t[0,0]
th1 = t[1,0]
th2 = t[2,0]
th3 = t[3,0]

# 2 year experience, 9 test score, 6 interview score
prdeiction1 = th0+(th1*2)+(th2*9)+(th3*6)

# 12 year experience, 10 test score, 10 interview score

prdeiction2 = th0+(th1*12)+(th2*10)+(th3*10)


#predictions = np.dot(X,t)

#plt.scatter(X[:,1],y_scale)
#plt.plot(X[:,1],predictions, color="red")
#plt.show()

