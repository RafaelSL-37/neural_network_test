import csv
import os

import numpy as np


def _readIdxImages(filePath):
    with open(filePath, "rb") as fileHandle:
        magicNumber = int.from_bytes(fileHandle.read(4), byteorder="big")
        if magicNumber != 2051:
            raise ValueError(f"Invalid image file: {filePath}")
        imageCount = int.from_bytes(fileHandle.read(4), byteorder="big")
        rows = int.from_bytes(fileHandle.read(4), byteorder="big")
        cols = int.from_bytes(fileHandle.read(4), byteorder="big")
        data = np.frombuffer(fileHandle.read(), dtype=np.uint8)

    imageShape = (imageCount, rows, cols)
    images = data.reshape(imageShape)
    return images.astype(np.float32) / 255.0


def _readIdxLabels(filePath):
    with open(filePath, "rb") as fileHandle:
        magicNumber = int.from_bytes(fileHandle.read(4), byteorder="big")
        if magicNumber != 2049:
            raise ValueError(f"Invalid label file: {filePath}")
        labelCount = int.from_bytes(fileHandle.read(4), byteorder="big")
        labels = np.frombuffer(fileHandle.read(), dtype=np.uint8)

    if len(labels) != labelCount:
        raise ValueError(f"Label count mismatch in {filePath}")
    return labels.astype(int)


def loadMnistDataset(basePath):
    basePath = os.path.abspath(basePath)
    trainImages = _readIdxImages(os.path.join(basePath, "train-images.idx3-ubyte"))
    trainLabels = _readIdxLabels(os.path.join(basePath, "train-labels.idx1-ubyte"))
    testImages = _readIdxImages(os.path.join(basePath, "t10k-images.idx3-ubyte"))
    testLabels = _readIdxLabels(os.path.join(basePath, "t10k-labels.idx1-ubyte"))
    return trainImages, trainLabels, testImages, testLabels


def prepareDataset(filePath, testRatio=0.25):
    datasetPath = str(filePath)
    if not os.path.exists(datasetPath):
        raise FileNotFoundError(f"Dataset file not found: {datasetPath}")

    if datasetPath.lower().endswith(".csv"):
        with open(datasetPath, newline="") as csvFile:
            reader = csv.reader(csvFile)
            rows = list(reader)
        if len(rows) < 2:
            raise ValueError(f"Dataset file {datasetPath} does not contain enough rows.")

        data = np.asarray(rows[1:], dtype=float)
        if data.shape[1] < 2:
            raise ValueError("Dataset requires at least one feature column and one label column.")

        features = data[:, :-1]
        labels = data[:, -1].astype(int)
    elif datasetPath.lower().endswith(".npy"):
        loadedData = np.load(datasetPath, allow_pickle=True)
        if isinstance(loadedData, tuple) and len(loadedData) == 2:
            features, labels = loadedData
        else:
            features = loadedData[:, :-1]
            labels = loadedData[:, -1].astype(int)
    elif os.path.isdir(datasetPath):
        return loadMnistDataset(datasetPath)
    else:
        raise ValueError("Unsupported dataset format. Use a CSV, NumPy file, or MNIST directory.")

    featureCount = len(features)
    if featureCount < 2:
        raise ValueError("Dataset must contain at least two samples for a train/test split.")

    randomState = np.random.default_rng(42)
    indexes = np.arange(featureCount)
    randomState.shuffle(indexes)

    splitIndex = max(1, int(featureCount * (1.0 - testRatio)))
    trainIndexes = indexes[:splitIndex]
    testIndexes = indexes[splitIndex:]

    trainData = features[trainIndexes]
    trainLabels = labels[trainIndexes]
    testData = features[testIndexes]
    testLabels = labels[testIndexes]

    return trainData, trainLabels, testData, testLabels
