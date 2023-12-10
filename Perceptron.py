import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron

irisData = datasets.load_iris()

def test_perceptron(irisData):
    # Split the data into training and testing sets
    data_train, data_test, target_train, target_test = train_test_split(
        irisData.data, irisData.target, test_size=0.33, random_state=42
    )

    # Create a Perceptron classifier
    perceptron_classifier = Perceptron(random_state=42)

    # Fit the classifier on the training data
    perceptron_classifier.fit(data_train, target_train)

    # Predict the target values for the test data
    predictions = perceptron_classifier.predict(data_test)

    # Calculate the classification error
    classification_error = np.mean(predictions != target_test)

    return classification_error

# Run the test procedure 1000 times
total_error_perceptron = 0
num_iterations = 1000

for i in range(num_iterations):
    total_error_perceptron += test_perceptron(irisData)

average_error_perceptron = total_error_perceptron / num_iterations
print("Average Perceptron Classification Error:", average_error_perceptron)
