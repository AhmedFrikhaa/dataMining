import numpy
from sklearn import datasets
from sklearn import naive_bayes

irisData = datasets.load_iris()

nb = naive_bayes.MultinomialNB(fit_prior=True)
nb.fit(irisData.data[:], irisData.target[:])
P = nb.predict(irisData.data[:])
ea = 0
Y = irisData.target

for i in range(len(irisData.data)):
    if P[i] != Y[i]:
        ea = ea + 1
print(ea / len(irisData.data))
print(ea)
print(numpy.nonzero(P-Y))
print(nb.score(irisData.data,irisData.target))