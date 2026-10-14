import numpy as np

class OptimizationProblem:
    def __init__(self, function_name="rastrigin", dimensions=10):
        self.name = function_name.lower()
        self.dim = dimensions
        self.bounds = self._get_bounds()

    def _get_bounds(self):
        if self.name == "rastrigin":
            return (-5.12, 5.12)
        if self.name == "ackley":
            return (-5.0, 5.0)
        if self.name == "rosenbrock":
            return (-5.0, 10.0)
        return (-5.0, 5.0)

    def initialize_random_solution(self):
        """Generate a random candidate solution within bounds"""
        low, high = self.bounds
        return np.random.uniform(low, high, self.dim)

    def evaluate(self, x):
        x = np.array(x)
        if self.name == "rastrigin":
            return 10 * self.dim + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        
        elif self.name == "ackley":
            a, b, c = 20, 0.2, 2 * np.pi
            sum1 = np.sum(x**2)
            sum2 = np.sum(np.cos(c * x))
            return -a * np.exp(-b * np.sqrt(sum1 / self.dim)) - np.exp(sum2 / self.dim) + a + np.exp(1)
            
        elif self.name == "rosenbrock":
            return np.sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1.0 - x[:-1])**2.0)
        
        return np.sum(x**2) 