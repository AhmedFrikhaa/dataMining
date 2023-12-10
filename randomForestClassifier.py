import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

irisData = datasets.load_iris()

def test_random_forest(irisData):
    # Split the data into training and testing sets
    data_train, data_test, target_train, target_test = train_test_split(
        irisData.data, irisData.target, test_size=0.33, random_state=42
    )

    # Create a Random Forest classifier with a fixed random state
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the classifier on the training data
    rf_classifier.fit(data_train, target_train)

    # Predict the target values for the test data
    predictions = rf_classifier.predict(data_test)

    # Calculate the accuracy using scikit-learn's accuracy_score
    accuracy = accuracy_score(target_test, predictions)

    # Calculate the classification error
    classification_error = 1 - accuracy

    return classification_error

# Run the test procedure 1000 times
total_error = 0
num_iterations = 1000

for i in range(num_iterations):
    total_error += test_random_forest(irisData)

average_error = total_error / num_iterations
print("Average Classification Error:", average_error)
