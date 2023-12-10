import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

irisData = datasets.load_iris()

def test_bagging_knn(irisData, k=3, num_estimators=10):
    # Split the data into training and testing sets
    data_train, data_test, target_train, target_test = train_test_split(
        irisData.data, irisData.target, test_size=0.33, random_state=42
    )

    # Create a KNN classifier with a fixed random state
    base_classifier = KNeighborsClassifier(n_neighbors=k)

    # Create a BaggingClassifier using KNN as the base classifier
    bagging_classifier = BaggingClassifier(base_classifier, n_estimators=num_estimators, random_state=42)

    # Fit the bagging classifier on the training data
    bagging_classifier.fit(data_train, target_train)

    # Predict the target values for the test data
    predictions = bagging_classifier.predict(data_test)

    # Calculate the classification error
    classification_error = np.mean(predictions != target_test)

    return classification_error

# Run the bagging test procedure 1000 times
total_error_bagging = 0
num_iterations = 1000

for i in range(num_iterations):
    # You can customize the number of neighbors and the number of estimators
    total_error_bagging += test_bagging_knn(irisData, k=5, num_estimators=10)

average_error_bagging = total_error_bagging / num_iterations
print("Average Bagging Classification Error:", average_error_bagging)
