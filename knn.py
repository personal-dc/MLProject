from sklearn.neighbors import KNeighborsClassifier
import process_data as processor
from sklearnex import patch_sklearn 

patch_sklearn()


processor.go()

X_train = processor.X_train
y_train = processor.y_train
X_test = processor.X_test
y_test = processor.y_test

classifier = KNeighborsClassifier(n_neighbors = 5000)

classifier.fit(X_train, y_train)
print(classifier.score(X_test, y_test))