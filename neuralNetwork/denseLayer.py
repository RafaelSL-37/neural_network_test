from __future__ import annotations

import numpy as np

from .neuron import Neuron


class DenseLayer:
    def __init__(self, inputSize, outputSize, activationType="relu", randomSeed=0):
        self.neurons = [
            Neuron(inputSize, activationType=activationType, randomSeed=randomSeed + index)
            for index in range(outputSize)
        ]
        self.activationType = activationType

    def forward(self, inputs):
        outputs = np.array([neuron.compute(inputs) for neuron in self.neurons], dtype=float)
        return outputs

    def backward(self, gradOutput, previousActivation, learningRate, applyActivationDerivative=True):
        gradOutput = np.asarray(gradOutput, dtype=float)
        if applyActivationDerivative:
            deltas = gradOutput * np.array(
                [neuron.derivative(neuron.lastValue) for neuron in self.neurons],
                dtype=float,
            )
        else:
            deltas = gradOutput

        weightsMatrix = np.array([neuron.weight for neuron in self.neurons], dtype=float)
        gradWeights = np.outer(deltas, previousActivation)
        gradBiases = deltas.copy()

        for index, neuron in enumerate(self.neurons):
            neuron.weight -= learningRate * gradWeights[index]
            neuron.bias -= learningRate * gradBiases[index]

        return weightsMatrix.T @ deltas
