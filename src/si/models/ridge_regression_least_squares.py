from typing import Any
import numpy as np

from si.data.dataset import Dataset
from si.metrics.mse import mse


class RidgeRegressionLeastSquares:

    """ 
    ### Ridge Regression Least Squares


    The RidgeRegressionLeastSquares  is a linear model using the L2 regularization.
    This model solves the linear regression problem using the Least Squares technique.

    Parameters
    ----------
    l2_penalty: float
        The L2 regularization parameter
    scale: bool
        Whether to scale the dataset or not

    Attributes
    ----------
    theta: np.array
        The model parameters, namely the coefficients of the linear model.
        For example, x0 * theta[0] + x1 * theta[1] + ...
    theta_zero: float
        The model parameter, namely the intercept of the linear model.
        For example, theta_zero * 1
    """

    def __init__(self, l2_penalty: float = 1, scale: bool = True):
        """

        Parameters
        ----------
        l2_penalty: float
            The L2 regularization parameter

        scale: bool
            Whether to scale the dataset or not
        """
        # parameters
        self.l2_penalty = l2_penalty
        self.scale = scale

        # attributes
        self.theta = None
        self.theta_zero = None
        self.mean = None
        self.std = None
    
    def fit(self, dataset: Dataset) -> 'RidgeRegressionLeastSquares':

        """
        Fit the model to the dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to fit the model to
        """

        # Scale the data if required

        if self.scale:
            self.mean = np.nanmean(dataset.X, axis=0)
            self.std = np.nanstd(dataset.X, axis=0)
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X

        # Add intercept term to X 

        X = np.c_[np.ones(X.shape[0]), X]

        # Compute the penalty 

        penalty = self.l2_penalty * np.eye(X.shape[1])

        # Change the first position of the penalty matrix to 0 
        
        penalty[0, 0] = 0

        # Compute the model parameters theta_zero and theta 

        self.theta = np.linalg.inv(X.T @ X + penalty) @ X.T @ dataset.y
        self.theta_zero = self.theta[0]
        self.theta = self.theta[1:]

        return self
    
    def predict(self, dataset: Dataset) -> np.array:

        """
        Predict the output of the dataset using the model.

        Parameters
        ----------
        dataset: Dataset
            The dataset to predict the output of

        Returns
        -------
        np.array
            The predicted output
        """

        # Scale the data if required

        if self.scale:
            X = (dataset.X - self.mean) / self.std
        else:
            X = dataset.X
    
        # Add intercept term to X

        X = np.c_[np.ones(X.shape[0]), X]

        # Compute the predicted Y (X * thetas)

        return X.dot(np.r_[self.theta_zero, self.theta])
    

    def score(self, dataset: Dataset) -> float:

        """
        Compute the score of the model on a dataset.

        Parameters
        ----------
        dataset: Dataset
            The dataset to compute the score on

        Returns
        -------
        float
            The score of the model on the dataset
        """

        # Compute the predicted Y

        y_pred = self.predict(dataset)

        # Compute the MSE

        return mse(dataset.y, y_pred)



if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.metrics.mse import mse

    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    dataset_ = Dataset(X=X, y=y)


    print('Our model')
    model = RidgeRegressionLeastSquares()
    model.fit(dataset_)

    print(f"Parameters: {model.theta}")
    print(f"Intercept: {model.theta_zero}")

    score = model.score(dataset_)
    print(f"Score: {score}")

    y_pred_ = model.predict(Dataset(X=np.array([[3, 5]])))
    print(f"Predictions: {y_pred_}")

    print()

    print('Compare with sklearn')

    from sklearn.linear_model import Ridge
    model = Ridge()    
    X = (dataset_.X - np.nanmean(dataset_.X, axis=0)) / np.nanstd(dataset_.X, axis=0)
    model.fit(X, dataset_.y)
    print(f"Parameters: {model.coef_}") # should be the same as theta
    print(f"Intercept: {model.intercept_}") # should be the same as theta_zero
    print(f"Score: {mse(dataset_.y, model.predict(X))}")
    print(f"Predictions: {model.predict(np.array([[3, 5]]))}") # should be the same as y_pred_