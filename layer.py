"""
We need a linear layer and an activation layer
"""


class Layer:
    def __init__(self) -> None:
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Take these inputs and produce the correspoding outputs
        """
        raise NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        """
        backpropagate this gradient through the layer
        """
        raise NotImplementedError
