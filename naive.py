from sklearn import datasets
from sklearn import naive_bayes
nb = naive_bayes.MultinomialNB(fit_prior=True)# un algo d'apprentissage
irisData = datasets.load_iris()
nb.fit(irisData.data[:-1], irisData.target[:-1])
p31 = nb.predict([irisData.data[31]])
print (p31)
plast = nb.predict([irisData.data[-1]])
print (plast)
p = nb.predict(irisData.data[:])
print (p)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

# Load the Iris dataset
irisData = datasets.load_iris()

# Split the training dataset into two parts (e.g., 70% for training and 30% for testing)
X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.3, random_state=42)

# Create a Gaussian Naive Bayes classifier
clf = GaussianNB()

# Train the model on the training data
clf.fit(X_train, y_train)

# Evaluate the model on the test data
error_estimee = 1 - clf.score(X_test, y_test)

# Calculate the number of incorrect instances in the test set
incorrect_instances = len(y_test) * error_estimee

print("Erreur estimée (séparation en 2 parties):", error_estimee)
print("Nombre d'instances erronées dans le jeu de test:", int(incorrect_instances))