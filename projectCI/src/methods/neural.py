import numpy as np
from .base import CIMethod

class Perceptron(CIMethod):
    def __init__(self, params: dict = None):
        default_params = {
            "learning_rate": 0.01,
            "max_epochs": 100,
            "bias": True
        }
        if params:
            default_params.update(params)
        super().__init__(default_params)
        self.weights = None
        self.bias_weight = 0.0

    def fit(self, X, y, targets=None):
        self.validate_parameters(["learning_rate", "max_epochs"])
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias_weight = 0.0

        for epoch in range(self.params["max_epochs"]):
            errors = 0
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias_weight
                y_pred = 1 if linear_output >= 0 else 0
                update = self.params["learning_rate"] * (y[i] - y_pred)
                self.weights += update * X[i]
                if self.params["bias"]:
                    self.bias_weight += update
                errors += int(update != 0.0)

            self.convergence_history.append(errors)
            if errors == 0:
                self.logs.append(f"Converged at epoch {epoch}")
                break

        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias_weight
        return np.where(linear_output >= 0, 1, 0)
    
from sklearn.neural_network import MLPClassifier
from .base import CIMethod

class MLP(CIMethod):
    def __init__(self, params: dict = None):
        defaults = {
            "hidden_layers": [64, 32],
            "activation": "relu",
            "learning_rate": 0.001,
            "max_epochs": 500,
            "optimizer": "adam"
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)

        self.model = MLPClassifier(
            hidden_layer_sizes=self.params["hidden_layers"],
            activation=self.params["activation"],
            learning_rate_init=self.params["learning_rate"],
            max_iter=self.params["max_epochs"],
            solver=self.params["optimizer"],
            random_state=42
        )

    def fit(self, X, y, targets=None):
        self.model.fit(X, y)
        self.convergence_history = self.model.loss_curve_
        self.logs.append(f"MLP trained for {self.model.n_iter_} iterations.")
        return self

    def predict(self, X):
        return self.model.predict(X)
    

class SOM(CIMethod):
    def __init__(self, params: dict = None):
        defaults = {
            "map_size": (10, 10),
            "learning_rate_initial": 0.5,
            "max_epochs": 1000,
            "topology": "rectangular"
        }
        if params:
            if "map_size" in params and isinstance(params["map_size"], str):
                map_str = params["map_size"]
                if 'x' in map_str:
                    rows, cols = map(int, map_str.split('x'))
                    params["map_size"] = (rows, cols)
                elif ',' in map_str:
                    rows, cols = map(int, map_str.split(','))
                    params["map_size"] = (rows, cols)
            
            defaults.update(params)
        super().__init__(defaults)
        self.weights = None

    def fit(self, X, y=None, targets=None):
        n_samples, n_features = X.shape
        rows, cols = self.params["map_size"]
        self.weights = np.random.random((rows, cols, n_features))

        for epoch in range(self.params["max_epochs"]):
            lr = self.params["learning_rate_initial"] * (1 - epoch / self.params["max_epochs"])
            sample = X[np.random.randint(0, n_samples)]
            dists = np.linalg.norm(self.weights - sample, axis=2)
            bmu_idx = np.unravel_index(np.argmin(dists), (rows, cols))
            self.weights[bmu_idx] += lr * (sample - self.weights[bmu_idx])

            if epoch % 100 == 0:
                self.convergence_history.append(np.mean(dists))

        self.logs.append("SOM Training Complete")
        return self

    def predict(self, X):
        rows, cols = self.params["map_size"]
        predictions = []
        
        for x in X:
            dists = np.linalg.norm(self.weights - x, axis=2)
            bmu_idx = np.unravel_index(np.argmin(dists), (rows, cols))
            label = bmu_idx[0] * cols + bmu_idx[1]
            predictions.append(label)
            
        return np.array(predictions)
    def predict_2d(self, X):
        """Return 2D coordinates on the map"""
        rows, cols = self.params["map_size"]
        predictions = []
        
        for x in X:
            dists = np.linalg.norm(self.weights - x, axis=2)
            bmu_idx = np.unravel_index(np.argmin(dists), (rows, cols))
            predictions.append(bmu_idx)
            
        return np.array(predictions)

class HopfieldNetwork(CIMethod):
    def __init__(self, params: dict = None):
        defaults = {
            "max_iterations": 100,
            "async_update": True
        }
        if params:
            defaults.update(params)
        super().__init__(defaults)
        self.weights = None

    def fit(self, patterns, y=None, targets=None):
        num_patterns, n_features = patterns.shape
        self.weights = np.zeros((n_features, n_features))

        for p in patterns:
            self.weights += np.outer(p, p)

        np.fill_diagonal(self.weights, 0)
        self.weights /= n_features
        self.logs.append(f"Stored {num_patterns} patterns in Hopfield Network")
        return self

    def predict(self, X):
        state = np.copy(X)
        for iteration in range(self.params["max_iterations"]):
            prev_state = np.copy(state)
            if self.params["async_update"]:
                for j in range(len(state)):
                    state[j] = 1 if np.dot(self.weights[j], state) >= 0 else -1
            else:
                state = np.where(np.dot(self.weights, state) >= 0, 1, -1)

            if np.array_equal(state, prev_state):
                break

            self.convergence_history.append(np.mean(state != prev_state))

        return state