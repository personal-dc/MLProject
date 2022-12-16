import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.metrics import precision_score
import process_data as processor
from sklearn.model_selection import GridSearchCV
import numpy as np

dataset = pd.read_csv("./shot_logs.csv")
# print(dataset.describe())

processor.go()

X_train = processor.X_train_xgb
y_train = processor.y_train_xgb
X_test = processor.X_test_xgb
y_test = processor.y_test_xgb
X_validation = processor.X_validation_xgb
y_validation = processor.y_validation_xgb


xgb_model = XGBClassifier(min_child_weight=0.0001,learning_rate=8e-08,
                             n_estimators=1,max_depth=3).fit(X_train,y_train)

# xgb_model = XGBClassifier().fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
actuals = y_test        
precision=precision_score(actuals, predictions)
print(precision)

print(xgb_model.get_booster().get_score(importance_type = 'gain'))

# plot_importance(xgb_model, importance_type='weight')

# pyplot.show()

# parameters_for_testing = {
#     'min_child_weight':[0.0001],
#     'learning_rate':np.linspace(1e-08*8, 1e-08*9, num=10),
#     'max_depth':[3]
# }

# xgb_model = XGBClassifier()

# gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, scoring='precision')
# gsearch1.fit(X_train, y_train)

# print('best params')
# print (gsearch1.best_params_)
# print('best score')
# print (gsearch1.best_score_)  

# mean_test_scores = grid_search.cv_results_['mean_test_score']
# plt.plot(param_grid['C'], mean_test_scores)
# plt.xlabel('C')
# plt.ylabel('Mean Test Score')
# plt.show()

# best params
# {'learning_rate': 1e-07, 'max_depth': 3, 'min_child_weight': 0.0001, 'n_estimators': 1}
# best score
# 0.7942844148539047