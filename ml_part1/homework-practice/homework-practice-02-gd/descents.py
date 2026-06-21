import numpy as np
from abc import ABC, abstractmethod

# ===== Learning Rate Schedules =====
class LearningRateSchedule(ABC):
    @abstractmethod
    def get_lr(self, iteration: int) -> float:
        pass


class ConstantLR(LearningRateSchedule):
    def __init__(self, lr: float):
        self.lr = lr

    def get_lr(self, iteration: int) -> float:
        return self.lr


class TimeDecayLR(LearningRateSchedule):
    def __init__(self, lambda_: float = 1.0):
        self.s0 = 1
        self.p = 0.5
        self.lambda_ = lambda_

    def get_lr(self, iteration: int) -> float:
        return self.lambda_ * ((self.s0 / (self.s0 + iteration)) ** self.p)

# ===== Base Optimizer =====
class BaseDescent(ABC):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        self.lr_schedule = lr_schedule()
        self.iteration = 0
        self.model = None

    def set_model(self, model):
        self.model = model

    @abstractmethod
    def update_weights(self):
        pass

    def step(self):
        self.update_weights()
        self.iteration += 1


# ===== Specific Optimizers =====
class VanillaGradientDescent(BaseDescent):
    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        gradient = self.model.compute_gradients(X_train, y_train)
        update = -self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w = self.model.w + update
        return update


class StochasticGradientDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, batch_size=1):
        super().__init__(lr_schedule)
        self.batch_size = batch_size

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects = len(y_train)
        batch_indices = np.random.randint(num_objects, size=self.batch_size)
        gradient = self.model.compute_gradients(
            X_train[batch_indices], y_train[batch_indices]
        )
        update = -self.lr_schedule.get_lr(self.iteration) * gradient
        self.model.w = self.model.w + update
        return update


class SAGDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR):
        super().__init__(lr_schedule)
        self.grad_memory = None
        self.grad_sum = None

    def update_weights(self):
        X_train = self.model.X_train
        y_train = self.model.y_train
        num_objects, num_features = X_train.shape

        if self.grad_memory is None:
            self.grad_memory = np.zeros((num_objects, num_features))
            self.grad_sum = np.zeros(num_features)

        j = np.random.randint(num_objects)
        grad_old = self.grad_memory[j].copy()
        grad_new = np.asarray(
            self.model.compute_gradients(X_train[j : j + 1], y_train[j : j + 1])
        ).ravel()
        self.grad_sum += (grad_new - grad_old) / num_objects
        self.grad_memory[j] = grad_new

        update = -self.lr_schedule.get_lr(self.iteration) * self.grad_sum
        self.model.w = self.model.w + update
        return update


class MomentumDescent(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta=0.9):
        super().__init__(lr_schedule)
        self.beta = beta
        self.velocity = None

    def update_weights(self):
        if self.velocity is None:
            self.velocity = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train)
        self.velocity = (
            self.beta * self.velocity
            + self.lr_schedule.get_lr(self.iteration) * gradient
        )
        update = -self.velocity
        self.model.w = self.model.w + update
        return update


class Adam(BaseDescent):
    def __init__(self, lr_schedule: LearningRateSchedule = TimeDecayLR, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(lr_schedule)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None

    def update_weights(self):
        if self.m is None:
            self.m = np.zeros_like(self.model.w)
            self.v = np.zeros_like(self.model.w)

        gradient = self.model.compute_gradients(self.model.X_train, self.model.y_train)
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)

        t = self.iteration + 1
        m_hat = self.m / (1 - self.beta1 ** t)
        v_hat = self.v / (1 - self.beta2 ** t)

        update = (
            -self.lr_schedule.get_lr(self.iteration)
            * m_hat
            / (np.sqrt(v_hat) + self.eps)
        )
        self.model.w = self.model.w + update
        return update