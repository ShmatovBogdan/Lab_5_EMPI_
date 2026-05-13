from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

import pandas as pd

data = load_digits()

print(data.target_names)
print(data.feature_names)

df = pd.DataFrame(data.data, columns=data.feature_names)

print(df.head())

# статистика
print(df.info())
print(df.describe())
3
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.4, random_state=0)

results = []

# Логістична регресія
clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# точність позитивних передбачень
prec = precision_score(y_test, y_pred, average='weighted') 
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nЛогістична регресія")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

results.append(["Logistic Regression", acc, prec, f1])

# Дерево рішень
clf = DecisionTreeClassifier(max_depth=10)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nDecision Tree")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

results.append(["Decision Tree", acc, prec, f1])

# Випадковий ліс
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nRandom Forest")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

results.append(["Random Forest", acc, prec, f1])

# KNN
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nKNN")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

results.append(["KNN", acc, prec, f1])

# SVM
clf = SVC(gamma='scale')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nSVM")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

results.append(["SVM", acc, prec, f1])

# Naive Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print("\nNaive Bayes")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

results.append(["Naive Bayes", acc, prec, f1])

# Підсумкова таблиця
results_df = pd.DataFrame(results, columns=["Method", "Accuracy", "Precision", "F1-score"])

print("\nПідсумкова таблиця")
print(results_df)
