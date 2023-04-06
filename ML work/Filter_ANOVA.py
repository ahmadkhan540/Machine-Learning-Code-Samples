import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn import metrics
import os
##====================
os.chdir(r'E:\BS Teaching\Fall2022\IDS\PyProgs\CSV_Files')
data = pd.read_csv("Filter_ANOVA_Data.csv")
# print(data.head())
# print(data.shape)

features = data.drop(["Y"], axis=1)
label = data["Y"]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=0)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred= model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm_report = metrics.classification_report(y_test,y_pred)
ac  = accuracy_score(y_test, y_pred)
# print(cm)
print("Accuracy = ", ac)
#print(cm_report)
##### ANOVA Test ########
sel_features = SelectKBest(f_classif, k=5)
sel_features.fit(X_train, y_train)

#### Get Feature Score using sel_features.scores_
for i in range(len(sel_features.scores_)):
	print('Feature %d: %f' % (i, sel_features.scores_[i]))
selected_features = sel_features.get_support(indices=True)
print(selected_features)
# #
X_train_s = X_train.iloc[:,selected_features]
X_test_s = X_test.iloc[:,selected_features]
# #
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train_s, y_train)
y_pred= model.predict(X_test_s)
cm = confusion_matrix(y_test, y_pred)
cm_report = metrics.classification_report(y_test,y_pred)
ac  = accuracy_score(y_test, y_pred)
print("Accuracy After ANOVA = ", ac)