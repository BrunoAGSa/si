from typing import Dict, Tuple, Callable
import numpy as np
from si.model_selection.cross_validation import k_fold_cross_validation
from si.data.dataset import Dataset



def randomized_search_cv(model, dataset: Dataset, hyperparameter_grid: Dict[str, Tuple],scoring: Callable = None, cv: int = 5, n_iter: int = None):
    
    """
    Perform randomized search cross-validation on the given model and dataset.

    Parameters
    ----------
    model
        The model to cross validate.
    dataset: Dataset
        The dataset to cross validate on.
    hyperparameter_grid: Dict[str, Tuple]
        The hyperparameter grid to search over.
    scoring: Callable
        The scoring function to use. 
    cv: int
        The number of cross-validation folds.
    n_iter: int
        The number of hyperparameter settings to try.

    Returns
    -------
    results: Dict[str, float]
        Results of the grid search cross validation. Includes the scores, hyperparameters, best hyperparameters and best score
    """


    #Check if the provided hyperparameters are valid, i.e., if they exist in the model. Hint:use the python function hasattr.

    for hyperparameter in hyperparameter_grid.keys():
        if not hasattr(model, hyperparameter):
            raise ValueError("Invalid hyperparameter: " + hyperparameter)


    results = {"scores": [], "hyperparameters": [], "best_hyperparameters": None, "best_score": None}

    for _ in range(n_iter):

        #Get n_iter hyperparameter combinations. Hint: use np.random.choice to pick a set of random combinations from all possibilities
        #Set the model hyperparameters with the current combination. Hint: use the python function setattr.
        hyperparameters = []
        for hyperparameter in hyperparameter_grid.keys():
            value = np.random.choice(hyperparameter_grid[hyperparameter])
            hyperparameters.append(value)
            setattr(model, hyperparameter, value)


        #Cross validate the model using the k_fold_cross_validation function.

        scores = k_fold_cross_validation(model, dataset, scoring, cv)
        
        #Save the mean of the scores (k scores for k folds) and respective hyperparameters.

        results["scores"].append(np.mean(scores))
        results["hyperparameters"].append(hyperparameters)
                
        #Save the best score and respective hyperparameters.

        if results["best_score"] is None or np.mean(scores) > results["best_score"]:
            results["best_score"] = np.mean(scores)
            results["best_hyperparameters"] = hyperparameters

    #Return a dictionary including all scores and hyperparameters computed and best score and best hyperparameters.
    
    return results

if __name__ == '__main__':
    from si.models.logistic_regression import LogisticRegression
    from si.model_selection.grid_search import grid_search_cv
    from si.io.csv_file import read_csv


    dataset = Dataset(np.random.rand(100, 10), np.random.randint(0, 2, 100))

    model = LogisticRegression()

    hyperparameter_grid = {"l2_penalty" : [0.1, 0.01, 0.001], "alpha": [0.1, 0.01, 0.001],"max_iter": [100, 200, 300]}


    results = randomized_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5, n_iter=10)

    print("Best score:", results["best_score"])
    print("Best hyperparameters:", results["best_hyperparameters"])
    #print("All scores:", results["scores"])
    #print("All hyperparameters:", results["hyperparameters"])
    print("---------------------------------------------------")
    
    results = grid_search_cv(model, dataset, hyperparameter_grid, scoring=None, cv=5)

    print("Best score:", results["best_score"])
    print("Best hyperparameters:", results["best_hyperparameters"])
    #print("All scores:", results["scores"])
    #print("All hyperparameters:", results["hyperparameters"])

    # sklearn ---------------------------------------------------

    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression as skLogisticRegression

    model = skLogisticRegression()

    hyperparameter_grid = {"penalty" : ["l2", "l1"], "C": [0.1, 0.01, 0.001],"max_iter": [100, 200, 300]}

    grid_search = GridSearchCV(model, hyperparameter_grid, cv=5, scoring=None, n_jobs=-1)

    grid_search.fit(dataset.X, dataset.y)

    print("Best score:", grid_search.best_score_)

    print("Best hyperparameters:", grid_search.best_params_)

    print("---------------------------------------------------")



    from si.io.csv_file import read_csv
    from si.models.logistic_regression import LogisticRegression
    from si.model_selection.cross_validation import k_fold_cross_validation

    breast_dataset = read_csv('/home/sa_bruno/Documentos/GitHub/si/datasets/breast_bin/breast-bin.csv', sep=',', features=True, label=True)

    lg = LogisticRegression()

    hyperparameter_grid = {
        'l2_penalty': np.linspace(1, 10, 10),
        'alpha': np.linspace(0.001, 0.0001, 100),
        'max_iter': np.linspace(1000, 2000, 200)
    }

    scores = randomized_search_cv(lg,
                            breast_dataset,
                            hyperparameter_grid=hyperparameter_grid,
                            cv=3,
                            n_iter=10
                            )

    print(f'Best hyperparameters: {scores["best_hyperparameters"]}')
    print(f'Best score: {scores["best_score"]}')







