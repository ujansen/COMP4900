from abc import ABC, abstractmethod
import numpy as np
import copy
from Utility import InfiniteScenarioGenerator, make_grid_city
from os import mkdir


def ReLU(values: np.array):
    return np.maximum(0, values)


def ReLUDerivative(values: np.array):
    return np.maximum(0, np.sign(values))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    sigmoid_x = sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def MeanSquaredError(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


def MeanSquaredErrorDerivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# https://towardsdatascience.com/math-neural-network-from-scratch-in-python-d6da9f29ce65
class Layer(ABC):
    @abstractmethod
    def forward(self, inputData: np.array) -> np.array:
        pass

    @abstractmethod
    def backward(self, outputError: np.array, learningRate: float):
        pass

    @abstractmethod
    def mutate(self, rate: float):
        pass


class DenseLayer(Layer):
    def __init__(self, inputSize, outputSize):
        self.weights = np.random.rand(inputSize, outputSize) - 0.5
        self.bias = np.random.rand(1, outputSize) - 0.5
        self.input = None
        self.output = None

    def forward(self, inputData: np.array):
        self.input: np.array = inputData
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, outputError: np.array, learningRate: float):
        input_error = np.dot(outputError, self.weights.T)
        weights_error = np.dot(self.input.T, outputError)
        self.weights -= learningRate * weights_error
        self.bias -= learningRate * outputError
        return input_error

    def mutate(self, rate: float):
        self.weights += (np.random.rand(self.weights.shape[0], self.weights.shape[1]) - 0.5) * rate
        self.bias += (np.random.rand(self.bias.shape[0], self.bias.shape[1]) - 0.5) * rate


class ActivationLayer(Layer):
    def __init__(self, activation, derivative):
        self.activation = activation
        self.derivative = derivative
        self.input = None
        self.output = None

    def forward(self, inputData):
        self.input = inputData
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_error, learningRate):
        return self.derivative(self.input) * output_error

    def mutate(self, rate: float):
        pass

    def copy(self):
        return self


class Evaluator:
    def __init__(self, graph: np.array, locations: np.array, divisor: float):
        self.graph: np.array = graph
        self.locations: np.array = locations
        self.divisor = divisor

    def evaluate(self, sample, agentPosition):
        order = np.argsort(sample) + 1
        order = np.insert(order, 0, 0)

        tempLocations = self.locations.copy()
        tempLocations = np.vstack((self.locations[agentPosition], tempLocations))

        differences = np.diff(tempLocations[order], axis=0)
        distances = np.linalg.norm(differences, axis=1)
        y_true = np.cumsum(distances) / self.divisor

        return y_true


class DistanceEvaluator:
    def __init__(self, graph: np.array, locations: np.array):
        self.graph: np.array = graph
        self.locations: np.array = locations

    def evaluate(self, sample, agentPosition, inputs):
        order = np.argsort(sample)
        reordered_goals = inputs[order[0]]
        target_indices = np.argwhere(reordered_goals == 1)
        last_goal = np.max(target_indices)

        order += 1
        order = np.insert(order, 0, 0)

        tempLocations = self.locations.copy()
        tempLocations = np.vstack((self.locations[agentPosition], tempLocations))

        differences = np.diff(tempLocations[order[:last_goal + 1]], axis=0)
        distances = np.linalg.norm(differences, axis=1)

        return np.sum(distances)


class BasicNeuralNetwork:
    def __init__(self, evaluator, lossFunc: callable, lossDerivativeFunc: callable):
        self.layers: list = []
        self.evaluator: Evaluator = evaluator
        self.lossFunc: callable = lossFunc
        self.lossDerivativeFunc: callable = lossDerivativeFunc

    def addLayer(self, layer: Layer):
        self.layers.append(layer)

    def fit(self, data, epochs: int, learning_rate: float):
        samples = data.shape[0]

        for i in range(epochs):
            avgError = 0
            for j in range(samples):
                output = data[j].reshape(1, data[j].shape[0])
                for layer in self.layers:
                    output = layer.forward(output)

                agentPosition = np.where(data[j][:data.shape[1] // 2] == 1)[0][0]
                trueValues = self.evaluator.evaluate(output, agentPosition)

                error = self.lossDerivativeFunc(trueValues, output)
                avgError += self.lossFunc(trueValues * self.evaluator.divisor, output)

                for layer in reversed(self.layers):
                    error = layer.backward(error, learning_rate)
            print(avgError / samples)


class PredefinedNeuralNetwork:
    def __init__(self, evaluator, input_size, hidden_layer_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.layers: list = []
        self.layers.append(DenseLayer(input_size, hidden_layer_size))
        self.layers.append(ActivationLayer(ReLU, None))
        for i in range(hidden_layers):
            self.layers.append(DenseLayer(hidden_layer_size, hidden_layer_size))
            self.layers.append(ActivationLayer(ReLU, None))
        self.layers.append(DenseLayer(hidden_layer_size, output_size))
        self.evaluator: Evaluator = evaluator

    def evaluate(self, samples: np.array):
        costs = np.zeros(samples.shape[0])
        for j in range(samples.shape[0]):
            output = samples[j]
            for layer in self.layers:
                output = layer.forward(output)
            agentPosition = np.where(samples[j][:samples.shape[1] // 2] == 1)[0][0]
            cost = self.evaluator.evaluate(output, agentPosition, samples[j][samples.shape[1] // 2:])
            costs[j] = cost
        return np.average(costs)

    def get_child(self, mutation_degree):
        new_network = copy.deepcopy(self)
        new_network.mutate(mutation_degree)
        return new_network

    def mutate(self, rate: float):
        for layer in self.layers:
            layer.mutate(rate)

    def save(self, id, path):
        path = path + str(id)
        mkdir(path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                np.savez(path + "/" + str(i), weights=layer.weights, biases=layer.bias)


class PopulationManager:
    def __init__(self, population_size):
        self.width = 10
        self.height = 10
        self.num_nodes = self.width * self.height
        graph, locations = make_grid_city(self.width, self.height)
        self.evaluator = DistanceEvaluator(graph, locations)
        self.population_size = population_size
        self.networks = np.array([
            PredefinedNeuralNetwork(
                self.evaluator,
                self.num_nodes * 2,
                32,
                6,
                self.num_nodes
            )
            for _ in range(population_size)
        ])

    def run_epoch(self, mutation_rate):
        generator = InfiniteScenarioGenerator(self.num_nodes, 0.9, 300)
        samples = np.array([sample for sample in generator])

        costs = np.zeros(self.population_size)
        for i in range(self.population_size):
            # print(f"Network {i}", end='')
            cost = self.networks[i].evaluate(samples)
            costs[i] = cost
            # print(f", Cost: {cost}")

        order = np.argsort(costs)
        probabilities = np.linspace(0, 1, self.population_size)
        rolls = np.random.rand(self.population_size)
        keep = np.where(rolls > probabilities)
        network_index = order[keep]
        new_list = self.networks[network_index]
        backfill = np.array([
            np.random.choice(new_list).get_child(mutation_rate)
            for _ in range(self.population_size - new_list.shape[0])
        ])
        self.networks = np.concatenate((new_list, backfill))
        print(f"Average: {np.average(costs)}, Best: {np.min(costs)}")

    def save(self, path):
        mkdir(path)
        for i in range(self.population_size):
            self.networks[i].save(i, path + "/")


if __name__ == "__main__":
    population = PopulationManager(1000)
    for i in range(500):
        population.run_epoch(0.03)
        if i % 20 == 0:
            population.save(f"./epoch_{i}")
    population.save(f"./final")
