"""
This is the loss function: Total Squred Error
"""

import numpy as np

from tensor import Tensor


class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError()

    def grad(self, predicted: Tensor, actual: Tensor) -> Tensor():
        raise NotImplementedError()
