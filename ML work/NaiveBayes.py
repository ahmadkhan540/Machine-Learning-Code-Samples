import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
import os
os.chdir(r'E:\BS Teaching\Fall2022\IDS\PyProgs\CSV_Files')

data = pd.read_csv("kNN_Dataset.csv")
print(data.head())
h = np.array(data['Height'])
w = np.array(data['Weight'])
label = np.array(data['Shirt'])

features=list(zip(h,w))
# Get Model
model = GaussianNB()
# Train the model using the training sets
model.fit(features,label)
#Predict Output
predicted= model.predict([[165,62]]) # h, w
print(predicted)
