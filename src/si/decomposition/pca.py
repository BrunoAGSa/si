import numpy as np

class PCA:

    """
    Principal Component Analysis (PCA) is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space. It tries to preserve the essential parts that have more variation of the data and remove the non-essential parts with fewer variation.
    """
    def __init__(self, n_components: int = 2):

        """ 
        PCA implementation using Singular Value Decomposition (SVD) to infer the principal components of a dataset.

        Parameters
        ----------
        n_components : int, optional
            Number of components to keep. If not specified, all components are kept.

        Attributes
        ----------
        n_components : int
            Number of components to keep.

        mean : np.ndarray
            Mean of the dataset.

        components : np.ndarray
            Principal components of the dataset.

        explained_variance : np.ndarray
            Variance explained by each of the selected components.
        
        """

        self.n_components = n_components
        self.mean = None
        self.components = None
        self.explained_variance = None

    def fit(self, X: np.ndarray) -> None:

        """ Fits PCA by inferring the principal components of a dataset.

        Parameters
        ----------
        X : np.ndarray
            Dataset to be fitted.
        """

        # centering the data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # calculate the SVD
        U, S, V = np.linalg.svd(X, full_matrices=False)

        # infer the principal components
        self.components = V[:self.n_components]
       
        # infer the explained variance
        explained_variance = S ** 2 / (len(X) - 1)
        self.explained_variance = explained_variance[:self.n_components]


    def transform(self, X: np.ndarray) -> np.ndarray:

        """ Transforms the dataset by projecting it into a lower-dimensional sub-space.

        Parameters
        ----------
        X : np.ndarray
            Dataset to be transformed.

            
        Returns
        -------
        X_reduced : np.ndarray
            Reduced dataset.
        """ 

        # centering the data
        X = X - self.mean

        # calculate the reduced X
        return np.dot(X, self.components.T)
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:

        """ Fits PCA by inferring the principal components of a dataset and transforms it by projecting it into a lower-dimensional sub-space.

        Parameters
        ----------
        X : np.ndarray
            Dataset to be fitted and transformed.


        Returns
        -------
        X_reduced : np.ndarray
            Reduced dataset.
        """

        self.fit(X)
        return self.transform(X)
    
if __name__ == '__main__':
    from si.data.dataset import Dataset
    from si.io.csv_file import read_csv

    dataset = Dataset(X=np.array([[0, 2, 0, 3],
                                [0, 1, 4, 3],
                                [0, 1, 1, 3]]),
                    y=np.array([0, 1, 0]),
                    features=["f1", "f2", "f3", "f4"],
                    label="y")
            
    pca = PCA(n_components=3)
    pca.fit_transform(dataset.X)

    print(f"Explained variance: {pca.explained_variance}")
    print()
    print(f"Components: {pca.components}")
    print()
    print(f"Mean: {pca.mean}")
    print()
    print(f"Original data: {dataset.X}")
    print()
    print(f"Transformed data: {pca.fit_transform(dataset.X)}")
    print()
    print()
    from sklearn.decomposition import PCA as PCA_skl

    data = read_csv('/home/sa_bruno/Documentos/GitHub/si/datasets/iris/iris.csv', sep=',', features=True, label=True)

    pca = PCA(n_components=3)
    pca.fit_transform(data.X)

    pca_skl = PCA_skl(n_components=3)
    pca_skl.fit_transform(data.X)

    print('Explained Variance')
    print('Our')
    print(pca.explained_variance)
    print()
    print('Sklearn')
    print(pca_skl.explained_variance_)
    print()
    print()

    print('Components')
    print('Our')
    print(pca.components)
    print()
    print('Sklearn')
    print(pca_skl.components_)
    print()
    print()

    print('Mean')
    print('Our')
    print(pca.mean)
    print()
    print('Sklearn')
    print(pca_skl.mean_)    
    print()
    print()