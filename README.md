import sys
 
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

features_train, labels_train, features_test, labels_test =  preprocess()



########################## SVM #################################
### we handle the import statement and SVC creation for you here
from sklearn.svm import SVC
clf = SVC(kernel="linear")


#### now your job is to fit the classifier
#### using the training features/labels, and to
#### make a set of predictions on the test data

clf.fit(features_train, labels_train)
SVC()

pred = clf.predict(features_test)

#### store your predictions in a list named pred





from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)

def submitAccuracy():
    return acc
