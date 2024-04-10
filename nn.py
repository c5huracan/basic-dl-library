"""
A neural network is just a collection of layers.
I behaves a lot like a layer itself. Although
we are not going to make it one. 
"""
from typing import Sequence, Iterator, Tuple
from tensor import Tensor
from layers import Layer

class NeuralNet:
    def __init__(self, layers: Sequence[Layer]) -> None:
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs