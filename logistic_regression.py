from sklearn.linear_model import LogisticRegression
import process_data as processor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearnex import patch_sklearn

processor.go()
patch_sklearn()

X_train = processor.X_train_sklearn
y_train = processor.y_train_sklearn
X_test = processor.X_test_sklearn
y_test = processor.y_test_sklearn
X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test)

param_grid = {
    'C': np.logspace(-4, 4, 50),
    'penalty': ['l1', 'l2']
}

classifier = LogisticRegression(max_iter = 500, C = 494.1713361323828, penalty='l2')

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))

# grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose = 7)
# grid_search.fit(X, y)
# print(grid_search.best_params_)