import pandas as pd
import numpy as np
import math
from preprocessing_data import load_problem
base_dir = "Data/"
filename = "[2, 3, 4]|[5].pickle"
x_train, y_train, x_test,y_test = load_problem(base_dir+filename)
from sklearn.neighbors import NearestNeighbors
label2one = {'B':[1,0,0],'S':[0,1,0],'X':[0,0,1]}
one2label = {0:'B', 1:'S', 2:'X'}
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=120, weights='distance',leaf_size=50)
neigh.fit(x_train, y_train)

def accuracy(pred, y):
    pred = np.argmax(pred,axis=1)
    if y is None:
        return 0
    return np.sum(pred == y) / y.shape[0]
def criterion(pred, y):
    s = 0
    for i in range(pred.shape[0]):
        s -= np.log(pred[i,y[i]])
    return s/y.shape[0]



n = 1000
sum = 0
for i in range(0,n):
    # pred = neigh.predict_proba(x_test[i*1200:(i+1)*1200].A)
    label2one = {'B':0,'S':1,'X':2}
    one2label = {0:'B', 1:'S', 2:'X'}
    vfunc = np.vectorize(lambda x:label2one[x])
    # pred+= 1e-4
    a = neigh.score(x_test[i*50:(i+1)*50].A,y_test[i*50:(i+1)*50])
    print(a)
    sum+=a
    # print(pred, criterion(pred, vfunc(y_test[i*1200:(i+1)*1200])))
print(sum/n)