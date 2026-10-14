import numpy as np
import random
import operator
from .base import CIMethod


class GeneticAlgorithm(CIMethod):


    def __init__(self, params: dict = None):
        defaults = {
            "population_size": 100,
            "generations": 500,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1,
            "selection": "tournament",
            "tournament_size": 3,
            "elitism": 2
        }
        if params:
            defaults.update(params)

        super().__init__(defaults)

    def fit(self, data, targets=None):

        problem = data
        self.validate_parameters([
            "population_size",
            "generations",
            "crossover_rate",
            "mutation_rate",
            "elitism"
        ])

        pop_size = self.params["population_size"]

        population = [
            problem.initialize_random_solution()
            for _ in range(pop_size)
        ]

        for generation in range(self.params["generations"]):

            fitness_scores = [
                problem.evaluate(individual)
                for individual in population
            ]

            sorted_indices = np.argsort(fitness_scores)

            elite_count = self.params["elitism"]
            new_population = [
                population[i].copy()
                for i in sorted_indices[:elite_count]
            ]
            while len(new_population) < pop_size:

                if self.params["selection"] == "tournament":
                    parent1 = self._tournament(population, fitness_scores)
                    parent2 = self._tournament(population, fitness_scores)
                else:
                    raise NotImplementedError(
                        "Only tournament selection is supported."
                    )
                if random.random() < self.params["crossover_rate"]:
                    child1, child2 = problem.crossover(parent1, parent2)
                else:
                    child1 = parent1.copy()
                    child2 = parent2.copy()
                child1 = problem.mutate(child1, self.params["mutation_rate"])
                child2 = problem.mutate(child2, self.params["mutation_rate"])

                new_population.extend([child1, child2])

            population = new_population[:pop_size]

            current_fitness = [
                problem.evaluate(individual)
                for individual in population
            ]

            best_fitness = min(current_fitness)
            self.convergence_history.append(best_fitness)

            self.logs.append({
                "generation": generation,
                "best_fitness": best_fitness
            })

        final_fitness = [
            problem.evaluate(individual)
            for individual in population
        ]
        self.best_solution = population[np.argmin(final_fitness)]

        return self

    def _tournament(self, population, fitness_scores):
        indices = random.sample(
            range(len(population)),
            self.params["tournament_size"]
        )
        best_index = indices[
            np.argmin([fitness_scores[i] for i in indices])
        ]

        return population[best_index]

    def predict(self, data=None):
        return self.best_solution

class PSO(CIMethod):

    def __init__(self, params: dict = None):
        defaults = {
            "n_particles": 50,
            "max_iterations": 500,
            "w": 0.7,   # Inertia weight
            "c1": 1.5,  # Cognitive coefficient
            "c2": 1.5,  # Social coefficient
            "v_max": 0.5
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def fit(self, data, targets=None):


        problem = data

        self.validate_parameters([
            "n_particles",
            "max_iterations",
            "w",
            "c1",
            "c2",
            "v_max"
        ])

        dim = problem.dim
        bounds = problem.bounds
        n_particles = self.params["n_particles"]


        pos = np.random.uniform(bounds[0], bounds[1], (n_particles, dim))
        vel = np.random.uniform(-self.params["v_max"], self.params["v_max"], (n_particles, dim))

        p_best = pos.copy()
        p_best_fit = np.array([problem.evaluate(p) for p in pos])

    
        g_best_index = np.argmin(p_best_fit)
        g_best = p_best[g_best_index].copy()
        g_best_fit = p_best_fit[g_best_index]

    
        for iteration in range(self.params["max_iterations"]):
            r1, r2 = np.random.rand(), np.random.rand()

            vel = (
                self.params["w"] * vel +
                self.params["c1"] * r1 * (p_best - pos) +
                self.params["c2"] * r2 * (g_best - pos)
            )

            pos += vel
            pos = np.clip(pos, bounds[0], bounds[1]) 

            for j in range(n_particles):
                fit = problem.evaluate(pos[j])
                if fit < p_best_fit[j]:
                    p_best[j] = pos[j].copy()
                    p_best_fit[j] = fit
            current_g_best_index = np.argmin(p_best_fit)
            current_g_best_fit = p_best_fit[current_g_best_index]
            if current_g_best_fit < g_best_fit:
                g_best_fit = current_g_best_fit
                g_best = p_best[current_g_best_index].copy()

            self.convergence_history.append(g_best_fit)
            self.logs.append({
                "iteration": iteration,
                "global_best_fitness": g_best_fit
            })


        self.best_solution = g_best
        return self

    def predict(self, data=None):

        return self.best_solution
    
    

class ACO(CIMethod):

    def __init__(self, params: dict = None):
        defaults = {
            "n_ants": 30,
            "max_iterations": 200,
            "alpha": 1.0,          # Pheromone importance
            "beta": 2.0,           # Heuristic importance
            "evaporation_rate": 0.5,
            "Q": 100               # Pheromone deposit factor
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def fit(self, data, targets=None):
   

        problem = data

        self.validate_parameters([
            "n_ants",
            "max_iterations",
            "alpha",
            "beta",
            "evaporation_rate",
            "Q"
        ])

        n_cities = problem.dist_matrix.shape[0]

        pheromones = np.ones((n_cities, n_cities))

        best_distance = float('inf')
        self.best_solution = None

        for iteration in range(self.params["max_iterations"]):

            all_tours = []
            all_distances = []

        
            for ant in range(self.params["n_ants"]):
                tour = self._build_tour(problem, pheromones)
                distance = problem.evaluate(tour)
                all_tours.append(tour)
                all_distances.append(distance)

        
                if distance < best_distance:
                    best_distance = distance
                    self.best_solution = tour.copy()

            pheromones *= (1 - self.params["evaporation_rate"])

            for tour, distance in zip(all_tours, all_distances):
                for i in range(n_cities - 1):
                    pheromones[tour[i], tour[i+1]] += self.params["Q"] / distance

            self.convergence_history.append(best_distance)
            self.logs.append({
                "iteration": iteration,
                "best_distance": best_distance
            })

        return self

    def _build_tour(self, problem, pheromones):

        n = problem.dist_matrix.shape[0]
        unvisited = list(range(n))
        tour = [random.choice(unvisited)]
        unvisited.remove(tour[0])

        while unvisited:
            current_city = tour[-1]
            probs = []
            for next_city in unvisited:
                tau = pheromones[current_city, next_city] ** self.params["alpha"]
                eta = (1.0 / problem.dist_matrix[current_city, next_city]) ** self.params["beta"]
                probs.append(tau * eta)

            probs = np.array(probs)
            probs /= probs.sum()

            next_city = np.random.choice(unvisited, p=probs)
            tour.append(next_city)
            unvisited.remove(next_city)

        return tour

    def predict(self, data=None):

        return self.best_solution

import random
import operator
import numpy as np
from .base import CIMethod

class Node:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

class GeneticProgramming(CIMethod):
    def __init__(self, params: dict = None):
        defaults = {
            "population_size": 200,
            "generations": 50,
            "max_depth": 6,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "function_set": ["+", "-", "*", "/"],
            "terminal_set": ["x", "constants"],
            "parsimony_coefficient": 0.001
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)
        self.best_tree = None
        self.ops = {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": lambda l, r: l / r if r != 0 else 1
        }

    def _generate_tree(self, depth):
        if depth == 0 or (depth < self.params["max_depth"] and random.random() < 0.3):
            return Node("x") if random.random() < 0.5 else Node(random.uniform(-5, 5))
        op = random.choice(self.params["function_set"])
        return Node(op, self._generate_tree(depth-1), self._generate_tree(depth-1))

    def _evaluate_tree(self, node, x):
        if isinstance(node.value, (int, float)):
            return node.value
        if node.value == "x":
            return x
        left_val = self._evaluate_tree(node.left, x)
        right_val = self._evaluate_tree(node.right, x)
        return self.ops[node.value](left_val, right_val)

    def fit(self, data, targets=None):
        self.validate_parameters(["population_size", "generations"])
        problem = data
        pop = [self._generate_tree(self.params["max_depth"] // 2) for _ in range(self.params["population_size"])]
        self.logs.append(f"Initialized GP with {self.params['population_size']} trees.")
        for gen in range(self.params["generations"]):
            fitness_scores = [problem.evaluate(tree, self._evaluate_tree) for tree in pop]
            best_idx = np.argmin(fitness_scores)
            self.convergence_history.append(fitness_scores[best_idx])
            if gen % 5 == 0:
                self.logs.append(f"Gen {gen}: Best Fitness {fitness_scores[best_idx]:.4f}")
            new_pop = [pop[best_idx]]
            while len(new_pop) < self.params["population_size"]:
                parent = pop[random.randint(0, len(pop)-1)]
                new_pop.append(parent)
            pop = new_pop
        self.best_tree = pop[0]
        return self

    def predict(self, x):
        return self._evaluate_tree(self.best_tree, x)