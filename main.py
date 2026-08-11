import os

import numpy as np

from datasets.datasetLoader import loadMnistDataset
from neuralNetwork.neuralNetwork import ConvolutionalNeuralNetwork, DenseNeuralNetwork, analyseResults


def main():
    datasetPath = os.environ.get("DATASET_PATH", os.path.join(os.path.dirname(__file__), "datasets", "digit_recognition"))
    trainData, trainLabels, testData, testLabels = loadMnistDataset(datasetPath)
    print(len(trainData), len(trainLabels), len(testData), len(testLabels))

    flattenedTrainData = trainData.reshape(len(trainData), -1)
    flattenedTestData = testData.reshape(len(testData), -1)

    denseNetwork = DenseNeuralNetwork(
        layerSizes=[flattenedTrainData.shape[1], 64, 32, 10],
        activationType="relu",
        learningRate=0.01,
        epochs=3,
        randomState=42,
    )
    denseNetwork.train(flattenedTrainData, trainLabels)

    densePredictedLabels = [denseNetwork.classify(sample) for sample in flattenedTestData[:200]]
    densePredictedProbabilities = [denseNetwork.predictProbabilities(sample) for sample in flattenedTestData[:200]]
    denseMetrics = analyseResults(testLabels[:200], densePredictedLabels, densePredictedProbabilities)

    convNetwork = ConvolutionalNeuralNetwork(
        inputShape=(28, 28),
        kernelSize=3,
        filterCount=4,
        activationType="relu",
        learningRate=0.01,
        epochs=1,
        randomState=42,
    )

    # amount of values available to train: 60000
    convNetwork.train(trainData[:500], trainLabels[:500])

    # amount of values available to test: 10000
    convPredictedLabels = [convNetwork.classify(sample) for sample in testData[:200]]
    convPredictedProbabilities = [convNetwork.predictProbabilities(sample) for sample in testData[:200]]
    convMetrics = analyseResults(testLabels[:200], convPredictedLabels, convPredictedProbabilities)

    print("Dense network metrics:")
    print(denseMetrics)
    print("\nConvolutional network metrics:")
    print(convMetrics)


if __name__ == "__main__":
    main()
