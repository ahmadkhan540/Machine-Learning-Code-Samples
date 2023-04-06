import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data 

CSV_Data = pd.read_csv("F:/UET/SEMESTER 6/ML/Assignment 1/canada_per_capita_income.csv", header=None)
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0]
Y = CSV_Data.iloc[:,1]

#Convert in 2 Rank

X = np.array(X)
Y = np.array(Y)
X = X[:,np.newaxis]
Y = Y[:,np.newaxis]

#Scaling_Of_X

min_x = X.min()
max_x = X.max()
scale_x_numerator = X-min_x
scale_x_denominator = max_x-min_x 

scale_x1 = scale_x_numerator/scale_x_denominator
scale_x2 = np.power(scale_x1[:,0],2)
scale_x3 = np.power(scale_x1[:,0],3)
scale_x4 = np.power(scale_x1[:,0],4)
scale_x5 = np.power(scale_x1[:,0],5)
scale_x6 = np.power(scale_x1[:,0],6)
scale_x7 = np.power(scale_x1[:,0],7)
scale_x8 = np.power(scale_x1[:,0],8)
scale_x9 = np.power(scale_x1[:,0],9)
scale_x10 = np.power(scale_x1[:,0],10)
scale_x11 = np.power(scale_x1[:,0],11)
scale_x12 = np.power(scale_x1[:,0],12)
scale_x13 = np.power(scale_x1[:,0],13)
scale_x14 = np.power(scale_x1[:,0],14)
scale_x15 = np.power(scale_x1[:,0],15)
scale_x16 = np.power(scale_x1[:,0],16)
scale_x17 = np.power(scale_x1[:,0],17)
scale_x18 = np.power(scale_x1[:,0],18)
scale_x19 = np.power(scale_x1[:,0],19)
scale_x20 = np.power(scale_x1[:,0],20)
scale_x21 = np.power(scale_x1[:,0],21)
scale_x22 = np.power(scale_x1[:,0],22)
scale_x23 = np.power(scale_x1[:,0],23)
scale_x24 = np.power(scale_x1[:,0],24)
scale_x25 = np.power(scale_x1[:,0],25)
scale_x26 = np.power(scale_x1[:,0],26)

#making it 2 Rank
def rank2(scale_x):
    scale_x = scale_x[:,np.newaxis]
    return scale_x

scale_x2 = rank2(scale_x2)
scale_x3 = rank2(scale_x3)
scale_x4 = rank2(scale_x4)
scale_x5 = rank2(scale_x5)
scale_x6 = rank2(scale_x6)
scale_x7 = rank2(scale_x7)
scale_x8 = rank2(scale_x8)
scale_x9 = rank2(scale_x9)
scale_x10 = rank2(scale_x10)
scale_x11 = rank2(scale_x11)
scale_x12 = rank2(scale_x12)
scale_x13 = rank2(scale_x13)
scale_x14 = rank2(scale_x14)
scale_x15 = rank2(scale_x15)
scale_x16 = rank2(scale_x16)
scale_x17 = rank2(scale_x17)
scale_x18 = rank2(scale_x18)
scale_x19 = rank2(scale_x19)
scale_x20 = rank2(scale_x20)
scale_x21 = rank2(scale_x21)
scale_x22 = rank2(scale_x22)
scale_x23 = rank2(scale_x23)
scale_x24 = rank2(scale_x24)
scale_x25 = rank2(scale_x25)
scale_x26 = rank2(scale_x26)

# Having Extra Column With Values 1 for theta 0

m,col = scale_x1.shape
ones = np.ones((m,1))
theta = np.zeros((27,1))
iterations = 200000
alpha = 0.01

#Concatenation

X_scaling = np.concatenate((ones,scale_x1,scale_x2,scale_x3,scale_x4,scale_x5,scale_x6,scale_x7,scale_x8,scale_x9,scale_x10,scale_x11,scale_x12,scale_x13,scale_x14,scale_x15,
                            scale_x16,scale_x17,scale_x18,scale_x19,scale_x20,scale_x21,scale_x22,scale_x23,scale_x24,scale_x25,scale_x26),axis=1)

#Scaling_Of_Y

min_y = Y.min()
max_y = Y.max()
scale_y_numerator = Y-min_y
scale_y_denominator = max_y-min_y
scale_y = scale_y_numerator/scale_y_denominator

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

h,t = Gradient_Decent_Algo(X_scaling, scale_y, theta, alpha, iterations, m)

final_prediction = np.dot(X_scaling,t)
final_prediction = final_prediction*(max_y-min_y)+min_y

plt.scatter(X,Y)
plt.plot(X,final_prediction,color="red")
plt.show()



