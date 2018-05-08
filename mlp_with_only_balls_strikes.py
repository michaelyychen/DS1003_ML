from preprocessing_data import load_problem
# from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
base_dir = "Data/"
filename = "save.pickle"

x_train, y_train, x_test,y_test = load_problem(base_dir+filename)
total_row, n_features = x_train.shape

label2one = {'B':0,'S':1,'X':2}
one2label = {0:'B', 1:'S', 2:'X'}
vfunc = np.vectorize(lambda x:label2one[x])

y_train = vfunc(y_train)
y_test = vfunc(y_test)
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

mlp_clsf = MLPClassifier(hidden_layer_sizes=(30,20 ),
                            activation='relu',
                            solver='adam',
                            alpha=0.0001,
                            batch_size='auto',
                            learning_rate='constant',
                            learning_rate_init=0.001,
                            power_t=0.5, max_iter=20,
                            shuffle=True, random_state=None, 
                            tol=0.0001, verbose=False, warm_start=False,
                            momentum=0.9, nesterovs_momentum=True,
                            early_stopping=False, validation_fraction=0.1,
                            beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# clf = LogisticRegression()
mlp_clsf.fit(x_train, y_train)
prob = mlp_clsf.predict_proba(x_test)
pred = np.argmax(prob,axis=1)
from confusion_matrix import generate_confusion_matrix
generate_confusion_matrix(y_test,pred)
# print(confusion_matrix(y_test, pred))
print("loss: ",criterion(prob, y_test))
print("Accuracy: ",accuracy(prob, y_test))
print("train shape",x_train.shape)
print("test shape",x_test.shape)

# clf.score(logit_prob_y, test_y )