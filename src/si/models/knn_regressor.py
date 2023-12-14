from typing import Callable, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.metrics.rmse import rmse
from si.statistics.euclidean_distance import euclidean_distance


class KNNRegressor:
    """
    The k-Nearst Neighbors regressor is a machine learning model that predicts the value of new samples based on a similarity measure (e.g., distance functions).
    This algorithm predicts the value of new samples by looking at the values of the k-nearest samples in the training data.  
    The predicted value is the average of the values of the k-nearest samples. 

    Parameters
    ----------
    k: int
        The number of nearest neighbors to use
    distance: Callable
        The distance function to use

    Attributes
    ----------
    dataset: np.ndarray
        The training data
    """

    def __init__(self, k: int = 1, distance: Callable = euclidean_distance):
        """
        Initialize the KNN regressor

        Parameters
        ----------
        k: int
            The number of nearest neighbors to use
        distance: Callable
            The distance function to use
        """
        # parameters
        self.k = k
        self.distance = distance

        # attributes
        self.dataset = None

    def fit(self, dataset: Dataset) -> 'KNNRegressor':
        """
        It fits the model to the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to 

        Returns
        -------
        self: KNNRegressor
            The fitted model
        """
        self.dataset = dataset
        return self

    def _get_closest_label(self, sample: np.ndarray) -> Union[int, str]:
        """
        It returns the closest label of the given sample

        Parameters
        ----------
        sample: np.ndarray
            The sample to get the closest label of

        Returns
        -------
        label: str or int
            The closest label
        """
        # compute the distance between the sample and the dataset
        distances = self.distance(sample, self.dataset.X)

        # get the k nearest neighbors
        k_nearest_neighbors = np.argsort(distances)[:self.k]

        # get the labels of the k nearest neighbors
        k_nearest_neighbors_labels = self.dataset.y[k_nearest_neighbors]

        # get the average of the labels
        return np.mean(k_nearest_neighbors_labels)

    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        It predicts the classes of the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the classes of

        Returns
        -------
        predictions: np.ndarray
            The predictions of the model
        """
        return np.apply_along_axis(self._get_closest_label, axis=1, arr=dataset.X)

    def score(self, dataset: Dataset) -> float:
        """
        It returns the accuracy of the model on the given dataset

        Parameters
        ----------
        dataset: Dataset
            The dataset to evaluate the model on

        Returns
        -------
        accuracy: float
            The accuracy of the model
        """
        predictions = self.predict(dataset)
        return rmse(dataset.y, predictions)


if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.model_selection.split import train_test_split
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.metrics import mean_squared_error
    from si.io.csv_file import read_csv

    dataset_ = read_csv('/home/sa_bruno/Documentos/GitHub/si/datasets/cpu/cpu.csv', features=True, label=True)
    dataset_train, dataset_test = train_test_split(dataset_, test_size=0.2)

    knn = KNNRegressor(k=3)

    knn.fit(dataset_train)

    score = knn.score(dataset_test)
    print(f'The rmse of the model is: {score}')

    # compare with sklearn

    knn_skl = KNeighborsRegressor(n_neighbors=3)

    knn_skl.fit(dataset_train.X, dataset_train.y)

    score_skl = mean_squared_error(dataset_test.y, knn_skl.predict(dataset_test.X), squared=False)
    print(f'The rmse_skl of the model is: {score_skl}')
