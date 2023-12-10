import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

irisData = datasets.load_iris()

def test_decision_tree(irisData):
    # Split the data into training and testing sets
    data_train, data_test, target_train, target_test = train_test_split(
        irisData.data, irisData.target, test_size=0.33, random_state=42
    )

    # Create a Decision Tree classifier with a fixed random state
    dt_classifier = DecisionTreeClassifier(random_state=42)

    # Fit the classifier on the training data
    dt_classifier.fit(data_train, target_train)

    # Predict the target values for the test data
    predictions = dt_classifier.predict(data_test)

    # Calculate the classification error
    classification_error = np.mean(predictions != target_test)

    return classification_error

# Run the test procedure 1000 times
total_error = 0
num_iterations = 1000

for i in range(num_iterations):
    total_error += test_decision_tree(irisData)

average_error = total_error / num_iterations
print("Average Classification Error:", average_error)
