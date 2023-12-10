import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

irisData = datasets.load_iris()

def test_knn(irisData, k=3):
    # Split the data into training and testing sets
    data_train, data_test, target_train, target_test = train_test_split(
        irisData.data, irisData.target, test_size=0.33, random_state=42
    )

    # Create a KNN classifier with a fixed random state
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier on the training data
    knn_classifier.fit(data_train, target_train)

    # Predict the target values for the test data
    predictions = knn_classifier.predict(data_test)

    # Calculate the classification error
    classification_error = np.mean(predictions != target_test)

    return classification_error

# Run the test procedure 1000 times
total_error = 0
num_iterations = 1000

for i in range(num_iterations):
    #we can costumize the number of neighbors
    total_error += test_knn(irisData,5)

average_error = total_error / num_iterations
print("Average Classification Error:", average_error)
