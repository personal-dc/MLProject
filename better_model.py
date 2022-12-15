import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBClassifier
from xgboost import plot_importance
from xgboost import plot_tree
import matplotlib.pyplot as plt
from matplotlib import pyplot
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, mean_squared_error,precision_score
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import math

dataset = pd.read_csv("./shot_logs.csv")
# print(dataset.describe())

def update_shot_clock(ds):
    new_col = []
    for index, value in ds['SHOT_CLOCK'].iteritems():
        if pd.isna(value):
            new_col.append(ds['GAME_CLOCK'][index])
        else:
            new_col.append(value)
    ds['SHOT_CLOCK'] = new_col

def shot_clock_map(time):
    if pd.isna(float(time)):
        return 0
    else:
        return time

datasettarget = dataset['FGM']

datasetwithouttarget = dataset[['PTS_TYPE', 'SHOT_NUMBER', 'GAME_CLOCK', 'SHOT_DIST','TOUCH_TIME','FINAL_MARGIN','PERIOD','SHOT_CLOCK','DRIBBLES','CLOSE_DEF_DIST', 'LOCATION', 'W']]
datasetwithouttarget['GAME_CLOCK'] = datasetwithouttarget['GAME_CLOCK'].map(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
datasetwithouttarget['LOCATION'] = datasetwithouttarget['LOCATION'].map(lambda loc : 1 if loc == 'H' else 0)
datasetwithouttarget['W'] = datasetwithouttarget['W'].map(lambda res : 1 if res == 'W' else 0)
datasetwithouttarget['PTS_TYPE'] = datasetwithouttarget['PTS_TYPE'].map(lambda type : 1 if type == 2 else 0)
# datasetwithouttarget['SHOT_CLOCK'] = datasetwithouttarget['SHOT_CLOCK'].map(lambda time : shot_clock_map(time))
update_shot_clock(datasetwithouttarget)

model = XGBClassifier()
model.fit(datasetwithouttarget,datasettarget)
# plot feature importance
plot_importance(model, importance_type='weight')
# pyplot.show()

X_train, X_test, y_train, y_test = train_test_split(datasetwithouttarget , datasettarget, test_size=0.50, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split( X_test, y_test, test_size=0.50, random_state=42)


# xgb_model = XGBClassifier(min_child_weight=0.0001,learning_rate=1e-05,
#                               n_estimators=1,max_depth=3).fit(X_train,y_train)
# predictions = xgb_model.predict(X_test)
# actuals = y_test        
# precision=precision_score(actuals, predictions)
# print(precision)

parameters_for_testing = {
    'min_child_weight':[0.0001,0.001,0.01],
    'learning_rate':[0.00001,0.0001,0.001],
    'n_estimators':[1,3,5,10],
    'max_depth':[3,4]
}

xgb_model = XGBClassifier()

gsearch1 = GridSearchCV(estimator = xgb_model, param_grid = parameters_for_testing, scoring='precision')
gsearch1.fit(X_train, y_train)

print('best params')
print (gsearch1.best_params_)
print('best score')
print (gsearch1.best_score_)  

# best params
# {'learning_rate': 1e-05, 'max_depth': 3, 'min_child_weight': 0.0001, 'n_estimators': 1}