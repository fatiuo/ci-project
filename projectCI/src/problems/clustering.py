import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score, 
    adjusted_rand_score
)

class ClusteringProblem:
    def __init__(self, dataset_type="synthetic"):
        self.dataset_type = dataset_type
        self.X = None
        self.y_true = None

    def load_data(self, n_samples=500, n_features=2):
        if self.dataset_type == "iris":
            print("here")
            iris = datasets.load_iris()
            self.X = iris.data
            self.y_true = iris.target
            
        elif self.dataset_type == "mall":
            try:
                df = pd.read_csv(r'D:\CI_Project\CI_Project\data\Mall_Customers.csv')
                self.X = df.iloc[:, [2, 3, 4]].values
            except FileNotFoundError:
                self.X = np.random.rand(200, 3)
            self.y_true = None
            
        else:
            self.X, self.y_true = datasets.make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=5,
                cluster_std=1.0,
                random_state=42
            )

        self.X = StandardScaler().fit_transform(self.X)

    def evaluate(self, labels):
        results = {
            "silhouette": silhouette_score(self.X, labels),
            "davies_bouldin": davies_bouldin_score(self.X, labels),
            "calinski_harabasz": calinski_harabasz_score(self.X, labels)
        }

        if self.y_true is not None:
            results["adjusted_rand_index"] = adjusted_rand_score(self.y_true, labels)

        return results