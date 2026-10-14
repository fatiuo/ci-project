import numpy as np
from .base import CIMethod

class FuzzyController(CIMethod):
    def __init__(self, params: dict = None):
        defaults = {
            "n_membership_functions": 3,
            "membership_type": "triangular",
            "defuzzification": "centroid",
            "rules": "standard"
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

    def _triangular(self, x, abc):
        a, b, c = abc
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))

    def fit(self, data, targets=None):
        self.universe = np.linspace(np.min(data), np.max(data), 100)
        u_min, u_max = np.min(data), np.max(data)
        mid = (u_min + u_max) / 2
        self.mfs = {
            "low": [u_min, u_min, mid],
            "medium": [u_min, mid, u_max],
            "high": [mid, u_max, u_max]
        }
        self.logs.append(f"Fuzzy sets initialized over range [{u_min}, {u_max}]")
        return self

    def fuzzify(self, value):
        memberships = {}
        for label, params in self.mfs.items():
            memberships[label] = self._triangular(value, params)
        return memberships

    def predict(self, input_value):
        fuzzy_vals = self.fuzzify(input_value)
        numerator = 0
        denominator = 0
        for label, activation in fuzzy_vals.items():
            center = self.mfs[label][1]
            numerator += activation * center
            denominator += activation
        if denominator == 0:
            return np.mean(self.universe)
        return numerator / denominator

    def get_metrics(self):
        metrics = super().get_metrics()
        metrics["n_sets"] = self.params["n_membership_functions"]
        return metrics