from math import exp
from typing import List
from itertools import cycle
from warnings import warn_explicit

import numpy as np


EPSILON = 1e-15
BATCH_AXIS = 0
FEATURE_AXIS = 1


def batchify(*args, batch_size=32):
    assert len({len(_) for _ in args}) == 1

    length = len(args[0])
    indices = np.array_split(np.arange(length), np.arange(batch_size, length, batch_size))
    for idx in indices:
        yield tuple(_[idx] for _ in args)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Activation:
    def call(self, x) -> float:
        pass

    def derivative(self, x):
        ...


class Loss:
    def call(self, y_true, y_pred):
        ...

    def derivative(self, y_true, y_pred):
        ...


class LogLoss:
    def call(self, y_true, y_pred):
        y_pred = y_pred.clip(EPSILON, 1 - EPSILON)
        return -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def derivative(self, y_true, y_pred):
        y_pred = y_pred.clip(EPSILON, 1 - EPSILON)
        return (y_pred - y_true) / (y_pred * (1 - y_pred))


class Sigmoid(Activation):
    def call(self, x) -> float:
        return sigmoid(x)

    def derivative(self, x):
        return sigmoid(x) * (1 - sigmoid(x))


class Perceptron:
    def __init__(self, activation: Activation, loss: Loss) -> None:
        self.input_dim = None
        self.activation = activation
        self.loss = loss
        self.bias = 0.0
        self.weights = None

    def fit(self, X, y, X_val, y_val, batch_size: int = 32, epochs: int = 1, learning_rate: float = 1e-3, early_stop: int = 0) -> List:
        history = []
        for epoch in range(epochs):
            for X_batch, y_batch in batchify(X, y, batch_size=batch_size):
                self.fit_batch(X_batch, y_batch, learning_rate=learning_rate)
            history.append(
                {
                    "train_loss": self.evaluate(X, y),
                    "val_loss": self.evaluate(X_val, y_val),
                }
            )
            if early_stop > 0 and epoch > early_stop:
                if all(history[-early_stop - 1]["val_loss"] < _["val_loss"] for _ in history[-early_stop:]):
                    print("J'ai plus de patience frère.")
                    break

        return history

    def evaluate(self, X, y_true):
        y_pred = self.forward(X)
        return self.loss.call(y_true, y_pred).mean()

    def fit_batch(self, X, y, learning_rate: float = 1e-3):
        if self.weights is None:
            self._init_weights(X)
        w_grad, b_grad = self.compute_grad(X, y)
        w_grad_to_apply = self.aggregate_weights_grad(w_grad)
        b_grad_to_apply = self.aggregate_bias_grad(b_grad)
        self.update(w_grad_to_apply, b_grad_to_apply, learning_rate=learning_rate)

    def forward(self, X: List[List[float]]) -> List[float]:
        return self.activation.call(self.linear(X))

    def linear(self, X: np.ndarray) -> np.ndarray:
        return (X * self.weights).sum(axis=FEATURE_AXIS) + self.bias

    def compute_grad(self, X, y):
        z = (X * self.weights).sum(axis=FEATURE_AXIS) + self.bias
        p = self.activation.call(z)
        w_grad = (self.loss.derivative(y, p) * self.activation.derivative(z)).reshape(-1, 1) * X
        b_grad = self.loss.derivative(y, p) * self.activation.derivative(z)
        return w_grad, b_grad

    def _init_weights(self, X):
        dim = X.shape[FEATURE_AXIS]
        self.weights = np.zeros(dim)

    def aggregate_weights_grad(self, grad):
        return grad.mean(axis=BATCH_AXIS)

    def aggregate_bias_grad(self, grad):
        return grad.mean(axis=BATCH_AXIS)

    def update(self, w_grad, b_grad, learning_rate):
        print(self.weights, self.bias)
        self.weights -= w_grad * learning_rate
        self.bias -= b_grad * learning_rate


if __name__ == "__main__":
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

    X = df[["Pclass", "Age", "SibSp", "Parch", "Fare"]].fillna(0)
    X = (X - X.mean()) / X.std()
    X = X.values
    y = df["Survived"].fillna(0).values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    lr = LogisticRegression()
    lr.fit(X_train, y_train)

    p = Perceptron(activation=Sigmoid(), loss=LogLoss())
    try:
        history = p.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            batch_size=32,
            epochs=200,
            learning_rate=1e-2,
            early_stop=1,
        )
    except KeyboardInterrupt:
        pass
    finally:
        __import__("IPython").embed()
