import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("F:/UET/SEMESTER 6/ML/Assignment 1/CarPrice_Assignment.csv")
CSV_Data = CSV_Data.replace(np.NaN,0)
X = CSV_Data.iloc[:,0:6]
Y = CSV_Data.iloc[:,6]

maxofy = Y.max()
minofy = Y.min()

# Making rank 2 arrays/
Y = np.array(Y)
Y = Y[:,np.newaxis]

#Slicing
X_slice = X[:155]
Y_slice = Y[:155]

def scale_down(X):
    min_X = X.min()
    max_X = X.max()
    t1 = X - min_X
    t2 = max_X - min_X
    x_scale = t1 / t2
    
    return x_scale

#Input features scale down
wheel = scale_down(X_slice.iloc[:,0])
carlength = scale_down(X_slice.iloc[:,1])
carwidth = scale_down(X_slice.iloc[:,2])
carheigth = scale_down(X_slice.iloc[:,3])
engine = scale_down(X_slice.iloc[:,4])
stroke = scale_down(X_slice.iloc[:,5])

#output features scale down
price = scale_down(Y_slice)

# Making rank 2 arrays/
wheel = np.array(wheel)
wheel = wheel[:,np.newaxis]

# Making rank 2 arrays/
carlength = np.array(carlength)
carlength = carlength[:,np.newaxis]

# Making rank 2 arrays/
carwidth = np.array(carwidth)
carwidth = carwidth[:,np.newaxis]

# Making rank 2 arrays/
carheigth = np.array(carheigth)
carheigth = carheigth[:,np.newaxis]

# Making rank 2 arrays/
engine = np.array(engine)
engine = engine[:,np.newaxis]

# Making rank 2 arrays/
stroke = np.array(stroke)
stroke = stroke[:,np.newaxis]

#merge  arrays
m,col = X_slice.shape
ones = np.ones((m,1))
X_stack = np.concatenate((ones,wheel,carlength,carwidth,carheigth,engine,stroke),axis=1)

#thetas initialization
theta = np.zeros((7,1))

iterations = 200000
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

h,t = Gradient_Decent_Algo(X_stack,price,theta,alpha,iterations,m)


th0 = t[0,0]
th1 = t[1,0]
th2 = t[2,0]
th3 = t[3,0]
th4 = t[4,0]
th5 = t[5,0]
th6 = t[6,0]


#training data 
X_train = CSV_Data.iloc[154:,0:6]
Y_train = CSV_Data.iloc[154:,6]

# Making rank 2 arrays/
Y_train = np.array(Y_train)
Y_train = Y_train[:,np.newaxis]


#Input features scale down
wheel_train = scale_down(X_train.iloc[:,0])
carlength_train = scale_down(X_train.iloc[:,1])
carwidth_train = scale_down(X_train.iloc[:,2])
carheigth_train = scale_down(X_train.iloc[:,3])
engine_train = scale_down(X_train.iloc[:,4])
stroke_train = scale_down(X_train.iloc[:,5])

#output features scale down
price = scale_down(Y_train)

# Making rank 2 arrays/
wheel_train = np.array(wheel_train)
wheel_train = wheel_train[:,np.newaxis]

carlength_train = np.array(carlength_train)
carlength_train = carlength_train[:,np.newaxis]

carwidth_train = np.array(carwidth_train)
carwidth_train = carwidth_train[:,np.newaxis]

carheigth_train = np.array(carheigth_train)
carheigth_train = carheigth_train[:,np.newaxis]

engine_train = np.array(engine_train)
engine_train = engine_train[:,np.newaxis]

stroke_train = np.array(stroke_train)
stroke_train = stroke_train[:,np.newaxis]

#merge  arrays
m,col = X_train.shape
ones = np.ones((m,1))
X_stack_train = np.concatenate((ones,wheel_train,carlength_train,carwidth_train,carheigth_train,engine_train,stroke_train),axis=1)

predict_values = np.dot(X_stack_train,t)

predict_values_t = np.zeros([50,1])

for i in range(0,50):
    predict_values_t[i,0] = predict_values[i,0]*(maxofy-minofy) + minofy
    
    
    
    
    
    




