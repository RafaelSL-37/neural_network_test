# Neural Network Test

This project builds a compact neural network for supervised classification. It includes a `Neuron` class with configurable activation functions, a `NeuralNetwork` class for training and classification, a dataset loader, and an evaluation step that saves performance visuals.

## Run it

```bash
pip install -r requirements.txt
```

```bash
python3 main.py
```

The default dataset is `sampleData.csv`. To use another file, set the `DATASET_PATH` environment variable or replace the placeholder in `main.py`.

## Included analysis

The program generates result plots in the `results` folder, including:
- confusion matrix
- ROC curve
- precision and recall metrics
