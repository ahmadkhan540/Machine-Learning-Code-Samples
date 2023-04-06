import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage import transform

# Labels

Y = np.array([0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,1,0,1,1,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,0,0,0,0,1,0,1,0,1,1,1,1,1,1,0,0,0,0,0,1,0,0,0,1,0,0,1,0,1,0,1,1,0,0,0,1,1,1,1,1,0,0,0,0,1,0,1,1,1,0,1,1,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,1,0,0,1,1,1,0,0,1,1,0,1,0,1,0,0,0,0,0,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0])
Y = np.array(Y[:,np.newaxis])

# Read Image

pics_data = []
for i in range(209):
    img_data = io.imread('F:/UET/SEMESTER 6/ML/Assignment 5/images/'+str(i)+'.jpg')
    img_data = img_data.reshape([1,12288])
    pics_data.append(img_data)

# Converting to (209,12288)

data = np.array(pics_data)
data = data.reshape((209,12288))
min_vals = data.min(axis=0)
max_vals = data.max(axis=0)
scaled_data = (data - min_vals) / (max_vals - min_vals)

m,c = scaled_data.shape
ones = np.ones((m,1))
after_scaling_of_features = np.concatenate((ones,scaled_data), axis=1)
theta = np.zeros((12289,1))
iterations = 11000
alpha = 0.001

# sigmoid fucntion

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

t,hist = gradient_decent(after_scaling_of_features, Y, theta, alpha, iterations, m) 
plt.plot(hist)
plt.show()

#test the data
img = io.imread('F:/UET/SEMESTER 6/ML/Assignment 5/abc.jpg')
test_img = transform.resize(img, (64, 64))
data = np.array(test_img)
data = data.reshape((1,12288))
rows,cols = data.shape
ones = np.ones((rows,1))
test = np.concatenate((ones,data), axis=1)

p=sigmoid(np.dot(test,t))

if p > 0.5:
    #p[i]=1
    print("It is Cat")
else:
    #p[i]=0
    print("It is not cat")

