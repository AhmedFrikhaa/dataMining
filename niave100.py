from sklearn import datasets
from sklearn import naive_bayes
irisData = datasets.load_iris()

nb = naive_bayes.MultinomialNB(fit_prior=True)
nb.fit(irisData.data[:99], irisData.target[:99])
print(nb.predict(irisData.data[100:149]))
