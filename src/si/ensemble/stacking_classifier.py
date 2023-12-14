import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy


class StackingClassifier:
    """
    Class representing a Stack Classifier.

    """

    def __init__(self, models, final_model):
        """
        Initialize the ensemble classifier.

        """
        # parameters
        self.models = models
        self.final_model = final_model



    def fit(self, dataset: Dataset) -> 'StackingClassifier':
        """
        Fit the models according to the given training data.

        """

        # train the initial set of models

        for model in self.models:
            model.fit(dataset)

        # get the predictions of the initial set of models

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()

        # train the final model with the predictions of the initial set of models

        self.final_model.fit(Dataset(dataset.X, predictions))

        return self

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class labels for samples in X.

        """

        # get the predictions of the initial set of models

        predictions = np.array([model.predict(dataset) for model in self.models]).transpose()


        # get the final predictions using the final model and the predictions of the initial set of models

        final_predictions = self.final_model.predict(Dataset(dataset.X, predictions))

        return final_predictions


    def score(self, dataset: Dataset) -> float:
        """
        Returns the mean accuracy on the given test data and labels.

        """
        return accuracy(dataset.y, self.predict(dataset))


if __name__ == '__main__':

    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split
    from si.models.knn_classifier import KNNClassifier
    from si.models.logistic_regression import LogisticRegression
    from si.models.decision_tree_classifier import DecisionTreeClassifier

    data = read_csv('/home/sa_bruno/Documentos/GitHub/si/datasets/breast_bin/breast-bin.csv', sep=',', features=True, label=True)

    train, test = train_test_split(data, test_size=0.33, random_state=42)

    knn = KNNClassifier(5)

    lr = LogisticRegression(0.01, 1000)

    dt = DecisionTreeClassifier(2, 10)

    knn2 = KNNClassifier(5)

    #create a StackingClassifier model using the previous classifiers. The second KNNClassifier model must be used as the final model.
    sc = StackingClassifier([knn, lr, dt], knn2)

    #Train the StackingClassifier model. What is the score of the model on the test set?
    sc.fit(train)
    print(f'Our model score: {sc.score(test)}')

    # sklearn 
    from sklearn.ensemble import StackingClassifier as StackingClassifierSK
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    #create a StackingClassifier model using the previous classifiers. The second KNNClassifier model must be used as the final model.
    sc = StackingClassifierSK(estimators=[('knn', KNeighborsClassifier(5)), ('lr', LogisticRegression()), ('dt', DecisionTreeClassifier())], final_estimator=KNeighborsClassifier(5))
    #Train the StackingClassifier model. What is the score of the model on the test set?
    sc.fit(train.X, train.y)
    print(f'Sklearn model score: {sc.score(test.X, test.y)}')