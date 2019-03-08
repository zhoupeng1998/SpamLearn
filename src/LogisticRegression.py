import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

data = np.genfromtxt("spambase/spambase.data", delimiter=',')
X = data[:,:57]
Y = data[:,-1]
scaler = StandardScaler()
X = scaler.fit_transform(X)

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.05, shuffle=True)

print("Number of features:\t", X.shape[1])
print("Number of training data points:\t", Xtr.shape[0])
print("Number of test data points:\t", Xte.shape[0])

logreg = LogisticRegression(tol=1e-8, C=10.0, random_state=0, solver='lbfgs', multi_class='multinomial')
clf = logreg.fit(Xtr,Ytr)
#clf  = LogisticRegression(tol=1e-8, C=10.0).fit(Xtr, Ytr)

print("training score:\t", clf.score(Xtr,Ytr))
print("test score:\t", clf.score(Xte,Yte))

Yhat = clf.predict(Xte)

err = 0
for i in range(len(Xte)):
    if Yhat[i] != Yte[i]:
        err += 1
print("Error rate: ", err/len(Xte))