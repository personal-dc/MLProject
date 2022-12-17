from sklearn.linear_model import LogisticRegression
import process_data as processor
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearnex import patch_sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

processor.go()
patch_sklearn()

X_train = processor.X_train_sklearn
y_train = processor.y_train_sklearn
X_test = processor.X_test_sklearn
y_test = processor.y_test_sklearn
X = np.append(X_train, X_test, axis = 0)
y = np.append(y_train, y_test)

# param_grid = {
#     'C': np.logspace(-4, 4, 50)
# }

classifier = LogisticRegression(max_iter=500)

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
y_pred = classifier.predict(X_test)

# grid_search = GridSearchCV(classifier, param_grid, cv=5, verbose = 1)
# grid_search.fit(X, y)
# print(grid_search.best_params_)
# print("score", grid_search.best_score_)

# mean_test_scores = grid_search.cv_results_['mean_test_score']
# plt.plot(param_grid['C'], mean_test_scores)
# plt.xlabel('C')
# plt.ylabel('Mean Test Score')
# plt.show()

confusion_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()