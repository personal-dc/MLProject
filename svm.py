from sklearn import svm
import process_data as processor
import numpy as np
from sklearnex import patch_sklearn 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

patch_sklearn()

processor.go()

X_train = processor.X_train_sklearn
y_train = processor.y_train_sklearn
X_test = processor.X_test_sklearn
y_test = processor.y_test_sklearn
X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test)

classifier = svm.SVC()

# param_grid = {'C': [0.1, 1, 10], 
#               'gamma': [1, 0.1, 0.01],
#               'kernel': ['rbf', 'poly', 'linear']} 
  
# grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 5, cv = 3)
  
# # fitting the model for grid search
# grid.fit(X_train, y_train)

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

y_pred = classifier.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()