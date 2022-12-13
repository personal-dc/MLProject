from xgboost import XGBClassifier
import process_data as processor

processor.go()

X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

classifier = XGBClassifier()

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))
