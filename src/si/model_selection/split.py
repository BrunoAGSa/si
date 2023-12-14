from typing import Tuple

import numpy as np

from si.data.dataset import Dataset


def train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    np.random.seed(random_state)
    # get dataset size
    n_samples = dataset.shape()[0]
    # get number of samples in the test set
    n_test = int(n_samples * test_size)
    # get the dataset permutations
    permutations = np.random.permutation(n_samples)
    # get samples in the test set
    test_idxs = permutations[:n_test]
    # get samples in the training set
    train_idxs = permutations[n_test:]
    # get the training and testing datasets
    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)
    return train, test

def stratified_train_test_split(dataset: Dataset, test_size: float = 0.2, random_state: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split the dataset into training and testing sets

    Parameters
    ----------
    dataset: Dataset
        The dataset to split
    test_size: float
        The proportion of the dataset to include in the test split
    random_state: int
        The seed of the random number generator

    Returns
    -------
    train: Dataset
        The training dataset
    test: Dataset
        The testing dataset
    """
    # set random state
    
    np.random.seed(random_state)

    # Get unique class labels and their counts;

    labels, counts = np.unique(dataset.y, return_counts=True)

    # Initialize empty lists for train and test indices;

    train_idxs = []
    test_idxs = []

    # Loop through unique labels;

    for label, count in zip(labels, counts):

        # Calculate the number of test samples for the current class;

        n_test = int(count * test_size)
        #ic(n_test)

        # Shuffle and select indices for the current class and add them to the test indices;

        permutations = np.random.permutation(np.where(dataset.y == label)[0])
        #ic(permutations)
        test_idxs.extend(permutations[:n_test])

        # Add the remaining indices to the train indices;

        train_idxs.extend(permutations[n_test:])


    # After the loop, create training and testing datasets;

    train = Dataset(dataset.X[train_idxs], dataset.y[train_idxs], features=dataset.features, label=dataset.label)
    test = Dataset(dataset.X[test_idxs], dataset.y[test_idxs], features=dataset.features, label=dataset.label)

    # Return the training and testing datasets.

    return train, test

if __name__ == '__main__':

    from si.io.csv_file import read_csv

    iris_dataset = read_csv('/home/sa_bruno/Documentos/GitHub/si/datasets/iris/iris.csv', features=True, label=True)

    train, test = train_test_split(iris_dataset, test_size=0.2, random_state=42)    
    train_strat, test_strat = stratified_train_test_split(iris_dataset, test_size=0.2, random_state=42)


    print("Train test split")
    print(f"Iris dataset: {iris_dataset.shape()}")
    print()
    print(f"Training dataset: {train.shape()}")
    print()
    print(f"Testing dataset: {test.shape()}")
    print()
    print()

    print("Stratified train test split")
    print(f"Iris dataset: {iris_dataset.shape()}")
    print()
    print(f"Training dataset: {train_strat.shape()}")
    print()
print(f"Testing dataset: {test_strat.shape()}")