import numpy as np

from neuralNetwork.neuralNetwork import ConvolutionalNeuralNetwork, NeuralNetwork


def testNeuronUsesSelectedActivation():
    network = NeuralNetwork(layerSizes=[2, 4, 2], activationType="relu", learningRate=0.05, epochs=20)
    assert network.layerSizes == [2, 4, 2]
    assert network.activationType == "relu"


def testTrainAndClassifyReturnValidLabels():
    trainData = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=float)
    labels = np.array([0, 1, 1, 0], dtype=int)

    network = NeuralNetwork(layerSizes=[2, 3, 2], activationType="relu", learningRate=0.05, epochs=50)
    network.train(trainData, labels)
    predictions = [network.classify(sample) for sample in trainData]

    assert len(predictions) == len(trainData)
    assert set(predictions).issubset({0, 1})


def testConvolutionalNetworkAcceptsImageInput():
    images = np.random.default_rng(0).random((12, 4, 4), dtype=float)
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=int)

    network = ConvolutionalNeuralNetwork(inputShape=(4, 4), kernelSize=2, filterCount=2, activationType="relu", learningRate=0.01, epochs=2, randomState=42)
    network.train(images[:10], labels[:10])
    prediction = network.classify(images[10])

    assert isinstance(prediction, (int, np.integer))
    assert prediction in {0, 1}
