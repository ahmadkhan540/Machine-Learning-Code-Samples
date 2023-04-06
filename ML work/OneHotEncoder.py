import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import os
os.chdir(r'E:\BS Teaching\Fall2022\IDS\PyProgs\CSV_Files')
data = pd.read_csv("DrugsData.csv")
print(data.head())
f1 = np.array(data['Age'])
f2 = np.array(data['BP'])
label = np.array(data['Drug'])
# LE = LabelEncoder()
# f2 = LE.fit_transform(f2)
#### OneHotEncoding of f2 #######
OHE = pd.get_dummies(data['BP'])
# print(OHE[0:3])
f2 = OHE['HIGH']
f3 = OHE['LOW']
f4 = OHE['NORMAL']
##########
features=list(zip(f1, f2, f3, f4))
print(features[0:5])
# # Get Model
model = KNeighborsClassifier(n_neighbors=3)
# Train the model using the training sets
model.fit(features,label)
# #Predict Output
predicted= model.predict([[45, 0,1,0]]) # f1, f2, f3, f4
print(predicted)
