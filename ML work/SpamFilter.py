import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import os
######################################
os.chdir(r'E:\BS Teaching\Fall2022\IDS\PyProgs\CSV_Files')
data = pd.read_csv("SpamFilter_Dataset.csv")
print(data.head())
print(data.shape)
# print(data['Label'].value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(data['Contents'], data['Label'], test_size=0.3, random_state=0)
#print(X_train.shape)
#print(X_test.shape)

vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)
#X_train_vectorized.toarray().shape

model = MultinomialNB(alpha=0.1)
model.fit(X_train_vectorized, y_train)

y_pred = model.predict(vectorizer.transform(X_test))
cm = confusion_matrix(y_test, y_pred)
cm = np.transpose(cm)
# ac  = accuracy_score(y_test, y_pred)
cm_report = metrics.classification_report(y_test,y_pred)
# print(cm)
# print(ac)
print(cm_report)
cmd = ConfusionMatrixDisplay(cm, display_labels=['Ham','Spam'])
cmd.plot(colorbar=False, cmap='Blues')
cmd.ax_.set(xlabel='Actual', ylabel='Predicted', title='Confusion Matrix Actual vs Predicted')
plt.show()
#
# y_p = model.predict(vectorizer.transform([
#        "Congratulations! You made it."
#      ]))
# print(y_p)