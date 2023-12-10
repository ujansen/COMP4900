import numpy as np


def make_grid_city(dim_x, dim_y):
    graph = np.zeros((dim_x * dim_y, 5), dtype=int)
    locations = np.zeros((dim_x * dim_y, 2), dtype=float)
    offset_per_x = 1.0 / dim_x
    offset_per_y = 1.0 / dim_y
    for x in range(dim_x):
        for y in range(dim_y):
            index = y * dim_x + x
            locations[index][0] = x * offset_per_x + (offset_per_x / 2)
            locations[index][1] = y * offset_per_y + (offset_per_y / 2)

            if x > 0:
                graph[index][0] = y * dim_x + (x - 1)
            else:
                graph[index][0] = index

            if y > 0:
                graph[index][1] = (y - 1) * dim_x + x
            else:
                graph[index][1] = index

            if x < dim_x - 1:
                graph[index][2] = y * dim_x + (x + 1)
            else:
                graph[index][2] = index

            if y < dim_y - 1:
                graph[index][3] = (y + 1) * dim_x + x
            else:
                graph[index][3] = index

            graph[index][4] = index

    return graph, locations


# https://wiki.python.org/moin/Generators
class InfiniteScenarioGenerator(object):
    def __init__(self, n_nodes, targetProbability, numSamples):
        self.n_nodes = n_nodes
        self.numSamples = numSamples
        self.samplesProduced = 0
        self.targetProbability = targetProbability

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.samplesProduced >= self.numSamples:
            raise StopIteration()
        targets = np.random.rand(self.n_nodes)
        targets = np.where(targets < self.targetProbability, 0, 1)
        if np.sum(targets) == 0:
            targets[np.random.randint(self.n_nodes)] = 1
        agent = np.zeros(self.n_nodes)
        agent[np.random.randint(self.n_nodes)] = 1
        combined = np.concatenate((agent, targets))
        self.samplesProduced += 1
        return combined
