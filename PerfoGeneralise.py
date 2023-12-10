import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

irisData = datasets.load_iris()

def test_naive_bayes(irisData):
    X_train, X_test, y_train, y_test = train_test_split(irisData.data, irisData.target, test_size=0.3, random_state=42)

    # Create a Gaussian Naive Bayes classifier
    clf = GaussianNB()

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Evaluate the model on the test data
    error_estimee = 1 - clf.score(X_test, y_test)

    return error_estimee


# Run the test procedure 1000 times
total_error = 0
num_iterations = 1000

for i in range(num_iterations):
    total_error += test_naive_bayes(irisData)

average_error = total_error / num_iterations
print("Average Classification Error:", average_error)
