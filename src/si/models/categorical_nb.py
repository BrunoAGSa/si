import numpy as np
from si.data.dataset import Dataset
from si.metrics.accuracy import accuracy

class CategoricalNB:

    def __init__(self, smoothing: float = 1.0):

        """
        Naive Bayes classifier for categorical features.

        Parameters
        ----------
        smoothing : float, default=1.0
            Smoothing parameter (Laplace) (0 for no smoothing).

        Attributes
        ----------
        class_prior : array, shape (n_classes,)
            Probability of each class.
        feature_probs : array, shape (n_classes, n_features)
            Probability of each feature per class.
        """

 
    

        self.smoothing = smoothing
        self.class_prior = None
        self.feature_probs = None

    
    def fit(self, dataset: Dataset):    

        """ Fit the model according to the given training data.

        """


        # Define n_samples, n_features, n_classes

        n_samples, n_features = dataset.shape()
        n_classes = dataset.get_classes().size
    
        # Initialize class_counts (size=n_classes), feature_counts (size=(n_classes, n_features)), and class_prior (size=n_classes) (usenp.zeros)

        class_counts = np.zeros(n_classes)
        feature_counts = np.zeros((n_classes, n_features))
        self.class_prior = np.zeros(n_classes)

        # Compute class_counts (number of samples for each class), 
        
        for i in range(n_classes):
            class_counts[i] = np.sum(dataset.y == i)    

        #feature_counts (sum of each column for each class), 
            
        for i in range(n_classes):
            for j in range(n_features):
                feature_counts[i][j] = np.sum(dataset.X[dataset.y == i, j])

        #and class_prior (class counts for each class divided by n_samples (total))

        for i in range(n_classes):
            self.class_prior[i] = class_counts[i] / n_samples


        # Apply Laplace smoothing to avoid zero probabilities to feature_counts and class_counts

        feature_counts += self.smoothing
        class_counts += self.smoothing

        # Compute feature_probs (feature_counts divided by class_counts for each class)

        self.feature_probs = feature_counts / class_counts[:, np.newaxis]

        return self
    
    def predict(self, dataset: Dataset):

        """ Perform classification on the samples

        """

        # For each sample compute the probability for each class (class_probs[c] = np.prod(sample * self.feature_probs[c] + (1 - sample) * (1 - self.feature_probs[c])) * self.class_prior[c])

        n_samples, _ = dataset.shape()
        n_classes = dataset.get_classes().size
        class_probs = np.zeros(n_classes)
        predictions = np.zeros(n_samples)


        for i in range(n_samples):
            for j in range(n_classes):
                class_probs[j] = np.prod(dataset.X[i] * self.feature_probs[j] + (1 - dataset.X[i]) * (1 - self.feature_probs[j])) * self.class_prior[j]

        # Pick the class with highest probability as the predicted class.
                
            predictions[i] = np.argmax(class_probs)
        
        return predictions
    
    def score (self, dataset: Dataset):

        """ Return the accuracy score of the model on the given test data.
            
        """

        # Get the predictions (y_pred)

        y_pred = self.predict(dataset)

        # Calculate the accuracy between actual values and predictions
        
        return accuracy(dataset.y, y_pred)

            
        

if __name__ == "__main__":

    from si.model_selection.split import train_test_split


    # test CategoricalNB

    dataset = Dataset.from_random(1000, 5, 3)
    train, test = train_test_split(dataset, 0.3)
    # print(dataset.summary())
    # print(dataset.X)
    # print(dataset.y)
    nb = CategoricalNB()
    nb.fit(train)
    nb.predict(test)
    print(nb.score(test))
    

    # test using sklearn 

    from sklearn.naive_bayes import CategoricalNB as CategoricalNB_sklearn
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(dataset.X, dataset.y, test_size=0.3, stratify=dataset.y)
    nb_sklearn = CategoricalNB_sklearn()
    nb_sklearn.fit(X_train, y_train)
    print(nb_sklearn.score(X_test, y_test))


  


    





