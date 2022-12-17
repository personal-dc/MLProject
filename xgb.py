import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
import process_data as processor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import seaborn as sns

dataset = pd.read_csv("./shot_logs.csv")
# print(dataset.describe())

processor.go()

X_train = processor.X_train_xgb
y_train = processor.y_train_xgb
X_test = processor.X_test_xgb
y_test = processor.y_test_xgb
X_validation = processor.X_validation_xgb
y_validation = processor.y_validation_xgb


# xgb_model = XGBClassifier().fit(X_train,y_train)

xgb_model = XGBClassifier(max_depth = 3, learning_rate = 1, min_child_weight = 1, n_estimators = 1).fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
actuals = y_test        
precision=precision_score(actuals, predictions)
print(precision)

confusion_matrix = confusion_matrix(y_test, predictions)

sns.heatmap(confusion_matrix, annot=True, fmt='d')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# parameters_for_testing = {
#     'min_child_weight':[1, 0.1, 0.001, 0.0001],
#     'learning_rate': [1, 0.1, 0.001, 0.0001],
#     'max_depth':[3], 
#     'n_estimators' : [1, 2, 3, 4]
# }

# xgb_model = XGBClassifier()

# gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, scoring='precision', verbose = 3)
# gsearch1.fit(X_train, y_train)

# print('best params')
# print (gsearch1.best_params_)
# print('best score')
# print (gsearch1.best_score_)  

# mean_test_scores = gsearch1.cv_results_['mean_test_score']
# plt.scatter(parameters_for_testing['n_estimators'], mean_test_scores)
# plt.xlabel('number of trees')
# plt.ylabel('Mean Test Score')
# plt.show()