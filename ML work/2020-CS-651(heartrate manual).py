import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Reading Files
Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 2/heartrate.csv")

X1 = Data.iloc[0:212,0:1]
X2 = Data.iloc[0:212,1:2]
X3 = Data.iloc[0:212,2:3]
X4 = Data.iloc[0:212,3:4]
X5 = Data.iloc[0:212,4:5]
X6 = Data.iloc[0:212,5:6]
X7 = Data.iloc[0:212,6:7]
Y = Data.iloc[0:212,-1]

Y = np.array(Y[:,np.newaxis])

def scaling_of_X_features(X):
    min_of_x = X.min()
    max_of_x = X.max()
    numerator = X-min_of_x
    denominator = max_of_x-min_of_x
    scaling = numerator/denominator
    return scaling

scale_x1 = scaling_of_X_features(X1)
scale_x2 = scaling_of_X_features(X2)
scale_x3 = scaling_of_X_features(X3)
scale_x4 = scaling_of_X_features(X4)
scale_x5 = scaling_of_X_features(X5)
scale_x6 = scaling_of_X_features(X6)
scale_x7 = scaling_of_X_features(X7)

# Defining important values
m,col = scale_x1.shape
ones = np.ones((m,1))
x_scale_of_features = np.concatenate((ones,scale_x1,scale_x2,scale_x3,scale_x4,scale_x5,scale_x6,scale_x7), axis=1)
theta = np.zeros((8,1))
iterations = 100000
alpha = 0.01

# sigmaoid fucntion

def sigmoid(h):
    g = 1/(1+np.exp(-h))
    return g


# Cost Fucction 

def Get_cost_J(X,Y,Theta,m):
    
    temp1 = np.multiply(Y,np.log(sigmoid(np.dot(X,Theta))))
    temp2 = np.multiply((1-Y),np.log(1-sigmoid(np.dot(X,Theta))))
    
    J  =(-1/m)*np.sum(temp1+temp2)
    return J

# Gradient Decent

def gradient_decent(x,y,theta,alpha,iterations,m):
    history = np.zeros((iterations,1))
    for i in range(iterations):
        z = np.dot(x,theta)
        predictions = sigmoid(z)
        error = predictions-y
        slope = (1/m)*np.dot(x.T,error)
        theta = theta  - (alpha*slope)
        history[i] = Get_cost_J(x, y, theta, m)
    
    return (theta,history)

t,hist = gradient_decent(x_scale_of_features, Y, theta, alpha, iterations, m) 
plt.plot(hist)
plt.show()


#Now For Testing Of 91 Rows

testing_x1 = Data.iloc[212:,0:1]
testing_x2 = Data.iloc[212:,1:2]
testing_x3 = Data.iloc[212:,2:3]
testing_x4 = Data.iloc[212:,3:4]
testing_x5 = Data.iloc[212:,4:5]
testing_x6 = Data.iloc[212:,5:6]
testing_x7 = Data.iloc[212:,6:7]
testing_y = Data.iloc[212:,7:8]

#Scaling Of Testing Features

scale_testing_x1 = scaling_of_X_features(testing_x1)
scale_testing_x2 = scaling_of_X_features(testing_x2)
scale_testing_x3 = scaling_of_X_features(testing_x3)
scale_testing_x4 = scaling_of_X_features(testing_x4)
scale_testing_x5 = scaling_of_X_features(testing_x5)
scale_testing_x6 = scaling_of_X_features(testing_x6)
scale_testing_x7 = scaling_of_X_features(testing_x7)

#Now we find out prediction of testing data

oness = np.ones((91,1))
concatenation_of_features = np.concatenate((oness,scale_testing_x1,scale_testing_x2,scale_testing_x3,scale_testing_x4,scale_testing_x5,scale_testing_x6,scale_testing_x7), axis=1)
p=sigmoid(np.dot(concatenation_of_features,t))


for i in range(91):
    if p[i] > 0.5:
        p[i]=1        
    else:
        p[i]=0

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import metrics

cm = confusion_matrix(testing_y, p)
ac  = accuracy_score(testing_y, p)
cm_report = metrics.classification_report(testing_y,p)
print(cm)
print(np.transpose(cm))
print(ac)
print(cm_report)
cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
cmd.plot(colorbar=False, cmap='Blues')
plt.show()






   