from __future__ import annotations

from enum import Enum

import numpy as np


class ActivationType(str, Enum):
    SIGMOID = "sigmoid"
    RELU = "relu"
    TANH = "tanh"
    LEAKY_RELU = "leakyRelu"
    LINEAR = "linear"


class Neuron:
    def __init__(self, inputSize, activationType="relu", randomSeed=0):
        rng = np.random.default_rng(randomSeed)
        self.weight = rng.normal(0.0, np.sqrt(2.0 / max(1, inputSize)), size=inputSize)
        self.bias = 0.0
        self.activationType = ActivationType(str(activationType).lower()).value
        self.lastInput = None
        self.lastValue = None

    def activate(self, value):
        if self.activationType == ActivationType.SIGMOID.value:
            return self.sigmoid(value)
        if self.activationType == ActivationType.RELU.value:
            return self.relu(value)
        if self.activationType == ActivationType.TANH.value:
            return self.tanh(value)
        if self.activationType == ActivationType.LEAKY_RELU.value:
            return self.leakyRelu(value)
        if self.activationType == ActivationType.LINEAR.value:
            return value
        raise ValueError(f"Unsupported activation type: {self.activationType}")

    def derivative(self, value):
        if self.activationType == ActivationType.SIGMOID.value:
            activated = self.sigmoid(value)
            return activated * (1.0 - activated)
        if self.activationType == ActivationType.RELU.value:
            return 1.0 if value > 0.0 else 0.0
        if self.activationType == ActivationType.TANH.value:
            return 1.0 - np.tanh(value) ** 2
        if self.activationType == ActivationType.LEAKY_RELU.value:
            return 1.0 if value >= 0.0 else 0.01
        if self.activationType == ActivationType.LINEAR.value:
            return 1.0
        raise ValueError(f"Unsupported activation type: {self.activationType}")

    @staticmethod
    def sigmoid(value):
        value = np.asarray(value, dtype=float)
        return 1.0 / (1.0 + np.exp(-value))

    @staticmethod
    def relu(value):
        return np.maximum(0.0, value)

    @staticmethod
    def tanh(value):
        return np.tanh(value)

    @staticmethod
    def leakyRelu(value, alpha=0.01):
        return np.where(value >= 0.0, value, alpha * value)

    def compute(self, inputs):
        self.lastInput = np.asarray(inputs, dtype=float)
        self.lastValue = np.dot(self.lastInput, self.weight) + self.bias
        return self.activate(self.lastValue)
