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
neigh = KNeighborsClassifier(n_neighbors=20, weights='distance',leaf_size=10)
neigh.fit(x_train[:1001186], y_train[:1001186]) 
print(neigh.score(x_test[:1300].A,y_test[:1300]))