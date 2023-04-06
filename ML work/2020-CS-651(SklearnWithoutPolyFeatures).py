import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Reading and visualizing data using scatter plot
CSV_Data = pd.read_csv("D:/Semester 6/Machine Learning/Assignment 2/heartrate.csv")
CSV_Data.shape
X = CSV_Data.iloc[:,0:7]
Y = CSV_Data.iloc[:,7]

Y = np.array(Y[:,np.newaxis])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
meanX1 = np.mean(X_train)
meanX2 =  np.mean(X_test)
stdX1 = np.std(X_train)
stdX2 = np.std(X_test)

X_train = (X_train - meanX1)/stdX1
X_test = (X_test - meanX2)/stdX2

X_train2 = scale.fit_transform(X_train)
X_test2 = scale.transform(X_test) 


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression()

logr.fit(X_train,y_train)

print(logr.coef_)

y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import metrics

cm = confusion_matrix(y_test, y_pred)
ac  = accuracy_score(y_test, y_pred)
cm_report = metrics.classification_report(y_test,y_pred)
print(cm)
print(np.transpose(cm))
print(ac)
print(cm_report)
cmd = ConfusionMatrixDisplay(cm, display_labels=['0','1'])
cmd.plot(colorbar=False, cmap='Blues')
plt.show()
