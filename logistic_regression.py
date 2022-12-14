from sklearn.linear_model import LogisticRegression
import process_data as processor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearnex import patch_sklearn 

processor.go()
patch_sklearn()

X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test
X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test)

param_grid = {
    'C': np.logspace(-4, 4, 50),
    'penalty': ['l1', 'l2']
}

classifier = LogisticRegression(max_iter = 500)

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

grid_search = GridSearchCV(classifier, param_grid, cv=5)
grid_search.fit(X, y)
print(grid_search.best_params_)