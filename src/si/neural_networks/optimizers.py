from abc import abstractmethod

import numpy as np


class Optimizer:

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        raise NotImplementedError


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize the optimizer.

        Parameters
        ----------
        learning_rate: float
            The learning rate to use for updating the weights.
        momentum:
            The momentum to use for updating the weights.
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.retained_gradient = None

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """
        if self.retained_gradient is None:
            self.retained_gradient = np.zeros(np.shape(w))
        self.retained_gradient = self.momentum * self.retained_gradient + (1 - self.momentum) * grad_loss_w
        return w - self.learning_rate * self.retained_gradient


class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initialize the optimizer.

        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0
   

    def update(self, w: np.ndarray, grad_loss_w: np.ndarray) -> np.ndarray:
        """
        Update the weights of the layer.

        Parameters
        ----------
        w: numpy.ndarray
            The current weights of the layer.
        grad_loss_w: numpy.ndarray
            The gradient of the loss function with respect to the weights.

        Returns
        -------
        numpy.ndarray
            The updated weights of the layer.
        """

        #  verify if m and v are initialized, if not initialize them as matrices of zeros
        if self.m is None:
            self.m = np.zeros(np.shape(w))
        if self.v is None:
            self.v = np.zeros(np.shape(w))

        # update t (t+=1)
        self.t += 1

        # compute and update m
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_loss_w

        #compute and update v
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad_loss_w ** 2)

        # compute m_hat
        m_hat = self.m / (1 - self.beta1 ** self.t)

        # compute v_hat
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # compute the moving averages
        w = w - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        # return the updated weights
        return w
    

if __name__ == '__main__':

    print('ADAM')
    print()

    adam = Adam()

    np.random.seed(42)
    X = np.random.randint(0, 5, size=(1, 5))

    np.random.seed(42)
    y = np.random.random(5)

    print(f'Input: {X}')
    print()
    print(f'Target: {y}')
    print()

    print(f'Output: {adam.update(X, y)}')   

