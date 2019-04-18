from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report

data = pd.read_csv('marks1.txt')

X = data.drop('0',axis=1)
y = data['0']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20)

svc = SVC(kernel='linear')
svc.fit(X_train,y_train)
prediction = svc.predict(X_test)

print(prediction)

print(y_test)

# Матрица неточностей, точность, полнота и мера F1
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


