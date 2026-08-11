from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score

from .denseLayer import DenseLayer
from .neuron import ActivationType


class NeuralNetwork:
    def __init__(self, layerSizes, activationType="relu", networkType="normal", regionalSize=None, learningRate=0.01, epochs=50, randomState=42):
        self.layerSizes = list(layerSizes)
        self.activationType = str(activationType).lower()
        self.networkType = str(networkType).lower()
        self.regionalSize = regionalSize
        self.learningRate = float(learningRate)
        self.epochs = int(epochs)
        self.randomState = int(randomState)
        self.layers = []

        for index in range(len(self.layerSizes) - 1):
            inputSize = self.layerSizes[index]
            outputSize = self.layerSizes[index + 1]
            currentActivation = self.activationType if index < len(self.layerSizes) - 2 else "linear"
            self.layers.append(DenseLayer(inputSize, outputSize, activationType=currentActivation, randomSeed=self.randomState + index))

    def _softmax(self, logits):
        logits = np.asarray(logits, dtype=float)
        shifted = logits - np.max(logits)
        exps = np.exp(shifted)
        return exps / np.sum(exps)

    def _forwardPass(self, sample):
        activations = [np.asarray(sample, dtype=float)]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        return activations

    def train(self, trainData, trainLabels):
        X = np.asarray(trainData, dtype=float).reshape(len(trainData), -1)
        y = np.asarray(trainLabels, dtype=int)

        for _ in range(self.epochs):
            for sample, label in zip(X, y):
                activations = self._forwardPass(sample)
                logits = activations[-1]
                probabilities = self._softmax(logits)
                target = np.zeros_like(probabilities)
                target[label] = 1.0

                gradOutput = probabilities - target
                nextGrad = gradOutput

                for layerIndex in range(len(self.layers) - 1, -1, -1):
                    layer = self.layers[layerIndex]
                    previousActivation = activations[layerIndex]
                    shouldApplyActivationDerivative = layerIndex < len(self.layers) - 1
                    nextGrad = layer.backward(nextGrad, previousActivation, self.learningRate, applyActivationDerivative=shouldApplyActivationDerivative)

        return self

    def predictProbabilities(self, sample):
        activations = np.asarray(sample, dtype=float)
        for layer in self.layers:
            activations = layer.forward(activations)
        return self._softmax(activations)

    def classify(self, sample):
        probabilities = self.predictProbabilities(sample)
        return int(np.argmax(probabilities))


class DenseNeuralNetwork(NeuralNetwork):
    def __init__(self, layerSizes, activationType="relu", learningRate=0.01, epochs=50, randomState=42):
        super().__init__(
            layerSizes=layerSizes,
            activationType=activationType,
            networkType="normal",
            regionalSize=None,
            learningRate=learningRate,
            epochs=epochs,
            randomState=randomState,
        )


class ConvolutionalNeuralNetwork:
    def __init__(self, inputShape=(28, 28), kernelSize=3, filterCount=4, activationType="relu", learningRate=0.01, epochs=10, randomState=42):
        self.inputShape = tuple(inputShape)
        self.kernelSize = int(kernelSize)
        self.filterCount = int(filterCount)
        self.activationType = str(activationType).lower()
        self.learningRate = float(learningRate)
        self.epochs = int(epochs)
        self.randomState = int(randomState)
        rng = np.random.default_rng(self.randomState)
        self.filters = rng.normal(0.0, 0.1, size=(self.filterCount, self.kernelSize, self.kernelSize))
        self.filterBiases = np.zeros(self.filterCount, dtype=float)
        featureSize = self._featureSize()
        self.classifier = DenseNeuralNetwork(
            layerSizes=[featureSize, 32, 10],
            activationType=self.activationType,
            learningRate=self.learningRate,
            epochs=self.epochs,
            randomState=self.randomState,
        )

    def _featureSize(self):
        convW = self.inputShape[1] - self.kernelSize + 1
        convH = self.inputShape[0] - self.kernelSize + 1
        return self.filterCount * convW * convH

    def _applyActivation(self, value):
        if self.activationType == ActivationType.RELU.value:
            return np.maximum(0.0, value)
        if self.activationType == ActivationType.TANH.value:
            return np.tanh(value)
        if self.activationType == ActivationType.SIGMOID.value:
            return 1.0 / (1.0 + np.exp(-value))
        if self.activationType == ActivationType.LEAKY_RELU.value:
            return np.where(value >= 0.0, value, 0.01 * value)
        return value

    def _extractFeatures(self, image):
        imageArray = np.asarray(image, dtype=float)
        if imageArray.ndim == 1:
            imageArray = imageArray.reshape(self.inputShape)
        elif imageArray.shape != self.inputShape:
            imageArray = imageArray.reshape(self.inputShape)

        convMaps = []
        for filterIndex in range(self.filterCount):
            kernel = self.filters[filterIndex]
            featureMap = np.zeros((self.inputShape[0] - self.kernelSize + 1, self.inputShape[1] - self.kernelSize + 1), dtype=float)
            for row in range(featureMap.shape[0]):
                for col in range(featureMap.shape[1]):
                    patch = imageArray[row:row + self.kernelSize, col:col + self.kernelSize]
                    featureMap[row, col] = np.sum(patch * kernel) + self.filterBiases[filterIndex]
            convMaps.append(self._applyActivation(featureMap))
        return np.concatenate([map_reshape.reshape(-1) for map_reshape in convMaps])

    def train(self, trainData, trainLabels):
        X = np.asarray(trainData, dtype=float)
        y = np.asarray(trainLabels, dtype=int)
        if X.ndim == 1:
            X = X.reshape(1, *self.inputShape)
        elif X.ndim == 2 and X.shape[1] == np.prod(self.inputShape):
            X = X.reshape(len(X), *self.inputShape)

        features = np.array([self._extractFeatures(sample) for sample in X], dtype=float)
        self.classifier.train(features, y)
        return self

    def predictProbabilities(self, sample):
        imageArray = np.asarray(sample, dtype=float)
        if imageArray.ndim == 1:
            imageArray = imageArray.reshape(self.inputShape)
        elif imageArray.shape != self.inputShape:
            imageArray = imageArray.reshape(self.inputShape)
        features = self._extractFeatures(imageArray)
        return self.classifier.predictProbabilities(features)

    def classify(self, sample):
        return int(np.argmax(self.predictProbabilities(sample)))


def analyseResults(testLabels, predictedLabels, predictedProbabilities):
    trueLabels = np.asarray(testLabels, dtype=int)
    predicted = np.asarray(predictedLabels, dtype=int)
    probabilities = np.asarray(predictedProbabilities, dtype=float)

    if probabilities.ndim == 1:
        probabilities = np.column_stack([1.0 - probabilities, probabilities])

    binaryLabels = (trueLabels == 1).astype(int)
    binaryPredictions = (predicted == 1).astype(int)
    positiveClassIndex = 1 if probabilities.shape[1] > 1 else 0
    positiveProbabilities = probabilities[:, positiveClassIndex]

    confusion = confusion_matrix(trueLabels, predicted, labels=np.unique(trueLabels))
    precision = precision_score(trueLabels, predicted, average="macro", zero_division=0)
    recall = recall_score(trueLabels, predicted, average="macro", zero_division=0)
    rocAuc = roc_auc_score(binaryLabels, positiveProbabilities)

    os.makedirs("results", exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(confusion, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    for rowIndex in range(confusion.shape[0]):
        for colIndex in range(confusion.shape[1]):
            plt.text(colIndex, rowIndex, str(confusion[rowIndex, colIndex]), ha="center", va="center", color="black")
    plt.tight_layout()
    plt.savefig("results/confusionMatrix.png")
    plt.close()

    from sklearn.metrics import roc_curve

    falsePositiveRate, truePositiveRate, _ = roc_curve(binaryLabels, positiveProbabilities)
    plt.figure(figsize=(6, 6))
    plt.plot(falsePositiveRate, truePositiveRate, label=f"ROC AUC = {rocAuc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("results/rocCurve.png")
    plt.close()

    return {
        "accuracy": float(accuracy_score(trueLabels, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "rocAuc": float(rocAuc),
        "confusionMatrix": confusion,
        "binaryPredictions": binaryPredictions,
    }


def compareModels(trainData, trainLabels, testData, testLabels):
    denseModel = DenseNeuralNetwork(
        layerSizes=[trainData.shape[1], 32, len(np.unique(trainLabels))],
        activationType="relu",
        learningRate=0.01,
        epochs=5,
        randomState=42,
    )
    denseModel.train(trainData, trainLabels)
    densePredictions = [denseModel.classify(sample) for sample in testData]
    denseProbabilities = [denseModel.predictProbabilities(sample) for sample in testData]

    convModel = ConvolutionalNeuralNetwork(
        inputShape=(28, 28),
        kernelSize=3,
        filterCount=4,
        activationType="relu",
        learningRate=0.01,
        epochs=2,
        randomState=42,
    )
    convModel.train(trainData.reshape(len(trainData), 28, 28), trainLabels)
    convPredictions = [convModel.classify(sample.reshape(28, 28)) for sample in testData]
    convProbabilities = [convModel.predictProbabilities(sample.reshape(28, 28)) for sample in testData]

    denseMetrics = analyseResults(testLabels, densePredictions, denseProbabilities)
    convMetrics = analyseResults(testLabels, convPredictions, convProbabilities)

    return {
        "dense": denseMetrics,
        "convolutional": convMetrics,
    }
