import numpy as np
import random
import scipy.spatial.distance as dist

class TSPProblem:
    def __init__(self, name="custom", nodes=None):
        self.name = name
        self.nodes = nodes
        self.n_cities = len(nodes) if nodes is not None else 0
        self.dist_matrix = None
        
        if nodes is not None:
            self.dist_matrix = dist.cdist(nodes, nodes, 'euclidean')

    def initialize_random_solution(self):
        return np.random.permutation(self.n_cities)

    def evaluate(self, route):
        total_dist = sum(
            self.dist_matrix[route[i], route[i+1]] 
            for i in range(self.n_cities - 1)
        )
        total_dist += self.dist_matrix[route[-1], route[0]] 
        return total_dist

    def crossover(self, p1, p2):
        size = self.n_cities
        a, b = sorted(random.sample(range(size), 2))
        
        def generate_child(parent1, parent2):
            child = [None] * size
            child[a:b] = parent1[a:b]
            p2_filtered = [item for item in parent2 if item not in child[a:b]]
            curr_idx = b % size
            for item in p2_filtered:
                while child[curr_idx] is not None:
                    curr_idx = (curr_idx + 1) % size
                child[curr_idx] = item
            return np.array(child)

        return generate_child(p1, p2), generate_child(p2, p1)

    def mutate(self, route, rate):
        new_route = route.copy()
        if random.random() < rate:
            idx1, idx2 = random.sample(range(self.n_cities), 2)
            new_route[idx1], new_route[idx2] = new_route[idx2], new_route[idx1]
        return new_route

    def apply_2opt(self, route):
        best_route = route.copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, self.n_cities - 2):
                for j in range(i + 1, self.n_cities):
                    if j - i == 1: 
                        continue
                    new_route = best_route.copy()
                    new_route[i:j] = best_route[i:j][::-1]
                    if self.evaluate(new_route) < self.evaluate(best_route):
                        best_route = new_route
                        improved = True
        return best_route