import numpy as np

from MyNeuralNet import BasicNeuralNetwork, Evaluator, MeanSquaredError, MeanSquaredErrorDerivative, DenseLayer, \
    ActivationLayer, ReLU, ReLUDerivative, sigmoid, sigmoid_derivative
from Utility import make_grid_city, InfiniteScenarioGenerator

if __name__ == "__main__":
    width = 10
    height = 10
    num_nodes = width * height
    graph, locations = make_grid_city(width, height)
    generator = InfiniteScenarioGenerator(num_nodes, 0.3, 100000)

    x_train = np.array([sample for sample in generator])

    model = BasicNeuralNetwork(
        evaluator=Evaluator(graph, locations, num_nodes * np.sqrt(2)),
        lossFunc=MeanSquaredError,
        lossDerivativeFunc=MeanSquaredErrorDerivative
    )
    model.addLayer(DenseLayer(num_nodes * 2, 64))
    model.addLayer(ActivationLayer(ReLU, ReLUDerivative))

    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(ReLU, ReLUDerivative))
    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))

    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(ReLU, ReLUDerivative))
    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))

    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(ReLU, ReLUDerivative))
    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))

    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(ReLU, ReLUDerivative))
    model.addLayer(DenseLayer(64, 64))
    model.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))

    model.addLayer(DenseLayer(64, num_nodes))
    model.addLayer(ActivationLayer(sigmoid, sigmoid_derivative))
    model.fit(x_train, 100, 0.01)
