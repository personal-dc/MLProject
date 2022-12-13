from sklearn import svm
import process_data as processor
import numpy as np
from sklearnex import patch_sklearn 
from sklearn.model_selection import cross_val_score

patch_sklearn()

processor.go()

X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test
X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test)

classifier = svm.SVC(kernel = 'rbf', verbose = 1)

scores = cross_val_score(classifier, X, y, cv=5)

print(scores)

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
