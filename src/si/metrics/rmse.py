import numpy as np

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:

    """
    It returns the root mean squared error for the y_pred variable.

    Parameters
    ----------
    y_true: np.ndarray
        The true labels of the dataset

    y_pred: np.ndarray
        The predicted labels of the dataset

    Returns
    -------
    rmse: float
        The root mean squared error of the model
    """

    return np.sqrt(np.sum((y_true - y_pred) ** 2) / len(y_true))


if __name__ == '__main__':
    import numpy as np 

    # compare with sklearn

    from sklearn.metrics import mean_squared_error as mean_squared_error_skl
    from sklearn.metrics import mean_absolute_error as mean_absolute_error_skl
    from sklearn.metrics import r2_score as r2_score_skl

    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1, 2, 3, 6, 20])


    our_rmse = rmse(y_true, y_pred)
    skl_rmse = mean_squared_error_skl(y_true, y_pred, squared=False)

    assert np.allclose(our_rmse, skl_rmse)

    print(f"our rmse = {our_rmse}")
    print(f"sklearn rmse = {skl_rmse}")
