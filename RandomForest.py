# %%
from pandas import read_csv
dataset = read_csv('breast_cancer.csv',skiprows=1, header=None)
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
dataset.head()

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)   
X_train.shape
y_train.shape

# %%
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0, max_features='sqrt',max_depth=10)

# %%
print(classifier.fit(X_train, y_train))

# %%
predictions = classifier.predict(X_test)
print(predictions)

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, predictions)
print(cm)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: {:.2f} %".format(accuracy*100))

# %%
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


