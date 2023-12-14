from typing import Literal, Tuple, Union

import numpy as np

from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy
from si.statistics.impurity import gini_impurity, entropy_impurity
from si.models.decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:

    """ 
    Class representing a Random Forest Classifier.
    """

    def __init__(self, n_estimators: int, max_features: int = None, min_samples_split: int = 2, max_depth: int = 10, mode: Literal['gini', 'entropy'] = 'gini' , seed: int = None):

        """
        Creates a Radom Forest Classifier object.

        Parameters
        ----------
        n_estimators : int
            The number of trees in the forest.
        max_features : int
            The number of features to consider when looking for the best split.
        min_samples_split : int, optional
            The minimum number of samples required to split an internal node, by default 2.
        max_depth : int, optional
            The maximum depth of the tree, by default 10.
        mode : Literal['gini', 'entropy']
            The impurity measure to use, by default 'gini'.
        seed : None, optional
            The seed to use for the random number generator, by default None.

        """
        # parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.mode = mode
        self.seed = seed
    
        #estimared parameters
        self.trees = []
 

    def fit(self, dataset: Dataset) -> 'RandomForestClassifier':
        """_summary_

        Parameters
        ----------
        dataset : Dataset
            The dataset to fit the model to.

        Returns
        -------
        RandomForestClassifier
            The fitted model.
        """

        #sets the random seed

        np.random.seed(self.seed)
        
        # Defines self.max_features to be int(np.sqrt(n_features)) if None
        n_samples, n_features = dataset.shape()

        if self.max_features is None:
            self.max_features = int(np.sqrt(n_features))

        # Create a bootsrap dataset (pick n_samples random samples from the dataset with replacement and self.max_features random features without replacement from the original dataset)

        for _ in range(self.n_estimators):

            boostrap_samples = np.random.choice(n_samples, n_samples, replace=True)
            boostrap_features = np.random.choice(n_features, self.max_features, replace=False)
            boostrap_dataset = Dataset(dataset.X[boostrap_samples][:, boostrap_features], dataset.y[boostrap_samples]) #samples y ?

            # Create and train a decision tree with the bootstrap dataset

            tree = DecisionTreeClassifier(self.min_samples_split, self.max_depth, self.mode)
            tree.fit(boostrap_dataset)

            # Append a tuple containing the features used and the trained tree

            self.trees.append((boostrap_features, tree))
        
        return self
    
    def predict(self, dataset: Dataset) -> np.ndarray:

        # get predictions from each tree using the respective set of features

        predictions = np.array([tree.predict(Dataset(X = dataset.X[:, features], y = dataset.y)) for features, tree in self.trees])
        
        # get the most common prediction for each sample

        return np.array([max(set(c), key=c.count) for c in zip(*predictions)])


    def score(self, dataset: Dataset) -> float:

        predictions = self.predict(dataset)
        return accuracy(dataset.y, predictions)


if __name__ == '__main__':
    from si.io.csv_file import read_csv
    from si.model_selection.split import train_test_split

    data = read_csv('/home/sa_bruno/Documentos/GitHub/si/datasets/iris/iris.csv', sep=',', features=True, label=True)
    train, test = train_test_split(data, test_size=0.33, random_state=42)
    model = RandomForestClassifier(n_estimators=5, min_samples_split=3, max_depth=3, mode='gini')
    model.fit(train)
    print('Our model')
    print(f'Score {model.score(test)}')
    print()


    from sklearn.ensemble import RandomForestClassifier as RFC
    print('Compare with sklearn')
    model = RFC(n_estimators=3, min_samples_split=3, max_depth=3, criterion='gini')
    model.fit(train.X, train.y)
    print(f'Score {model.score(test.X, test.y)}')
