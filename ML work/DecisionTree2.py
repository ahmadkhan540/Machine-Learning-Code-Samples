import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import metrics
import os

os.chdir(r'E:\BS Teaching\Fall2022\IDS\PyProgs\CSV_Files')
data = pd.read_csv("DecisionTree_Dataset.csv")
print(data.head())

features = data.iloc[:,:-1]
label = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)

model = DecisionTreeClassifier(criterion="entropy")
DT    = model.fit(X_train, y_train)
y_pred= model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac  = accuracy_score(y_test, y_pred)
cm_report = metrics.classification_report(y_test,y_pred)
print(cm)
print(cm_report)
print(ac)

# print(DT.predict([[137,40,35,168,43.1,2.288,33]]))
# tree.plot_tree(DT)
# plt.show()
