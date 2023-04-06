import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from sklearn.preprocessing import LabelEncoder
import os
os.chdir(r'E:\BS Teaching\Fall2022\IDS\PyProgs\CSV_Files')
data = pd.read_csv("PlayGolf.csv")
print(data.head())
data_num = data.apply(LabelEncoder().fit_transform)
print(data_num)

features = data_num.drop(["PlayGolf"], axis=1)
label = data_num["PlayGolf"]
model = DecisionTreeClassifier(criterion="entropy")
DT    = model.fit(features,label)

feature_names = data.columns[:4]
target_names = data['PlayGolf'].unique().tolist()

tree.plot_tree(DT, feature_names=feature_names, class_names=target_names, filled=True)
# tree.plot_tree(DT)
plt.show()
# print(DT.predict([[2,0,1,1]]))