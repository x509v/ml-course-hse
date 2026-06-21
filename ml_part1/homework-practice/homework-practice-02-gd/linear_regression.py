import numpy as np
from descents import BaseDescent
from dataclasses import dataclass
from enum import auto, Enum
from typing import Dict, Type, Optional


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()

class LinearRegression:
    def __init__(
        self,
        optimizer: Optional[BaseDescent | str] = None,
        l2_coef: float = 0.0,
        tolerance: float = 1e-6,
        max_iter: int = 1000,
        loss_function: LossFunction = LossFunction.MSE
    ):
        self.optimizer = optimizer
        if isinstance(optimizer, BaseDescent):
            self.optimizer.set_model(self)
        self.l2_coef = l2_coef
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.w = None
        self.X_train = None
        self.y_train = None
        self.loss_history = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w

    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.loss_function is LossFunction.MSE:
            residuals: np.ndarray = np.asarray(X @ self.w - y, dtype=float)
            return (2 / len(y)) * (X.T @ residuals) + 2 * self.l2_coef * self.w
        # elif self.loss_function is ...
        return None

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.loss_function is LossFunction.MSE:
            # TODO: реализовать loss-функцию MSE
            l: int = len(y)
            mse = (np.sum((X @ self.w - y) ** 2)) / l
            return mse + self.l2_coef * np.sum(self.w ** 2)
        # elif self.loss_function is ...
        return 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        # TODO: реализовать обучение модели
        self.X_train, self.y_train = X, y
        self.w = np.zeros(self.X_train.shape[1])
        self.loss_history.append(self.compute_loss(self.X_train, self.y_train))

        if isinstance(self.optimizer, BaseDescent):
            for _ in range(self.max_iter):
                old_w = self.w.copy()
                self.optimizer.step()
                self.loss_history.append(self.compute_loss(self.X_train, self.y_train))
                if np.linalg.norm(self.w - old_w) < self.tolerance:
                    break
        elif self.optimizer is None:
            regularization = self.l2_coef * len(y) * np.eye(X.shape[1])
            self.w = np.linalg.solve(X.T @ X + regularization, X.T @ y)
            self.loss_history.append(self.compute_loss(self.X_train, self.y_train))
        elif self.optimizer == "SVD":
            self.w = np.linalg.pinv(X) @ y
            self.loss_history.append(self.compute_loss(self.X_train, self.y_train))
        else:
            raise ValueError("Invalid optimizer")
        # elif self.optimizer is ...
