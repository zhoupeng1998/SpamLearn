import numpy as np
import matplotlib.pyplot as plt # use matplotlib for plotting with inline plots %matplotlib inline
import warnings
warnings.filterwarnings('ignore') # for deprecated matplotlib functions
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# load data
spamdata = np.genfromtxt("spambase/spambase.data",delimiter=",")
X, Y = spamdata[:,0:57], spamdata[:,-1]

scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size = 0.25, shuffle = True)

print("Number of features:\t", X.shape[1])
print("Number of training data points:\t", Xtr.shape[0])
print("Number of test data points:\t", Xte.shape[0])

clf = SVC().fit(Xtr, Ytr)
print("Training score:\t", clf.score(Xtr,Ytr))
print("Test score:\t", clf.score(Xte,Yte))

Yhat = clf.predict(Xte)
err = 0
for i in range(len(Xte)):
    if Yhat[i] != Yte[i]:
        err += 1
print("Error rate: ", err/len(Xte))