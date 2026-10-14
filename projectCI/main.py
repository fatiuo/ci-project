import numpy as np

from src.orchestrator.agent import LLMOrchestrator
from src.methods.evolutionary import (
    GeneticAlgorithm, PSO, ACO, GeneticProgramming
)
from src.methods.neural import (
    MLP, Perceptron, SOM, HopfieldNetwork
)
from src.problems.tsp import TSPProblem
from src.problems.optimization import OptimizationProblem
from src.problems.classification import TitanicProblem
from src.problems.clustering import ClusteringProblem


def normalize_method_name(name: str):
    """Normalize method name returned by LLM."""
    name = name.lower()
    if "genetic programming" in name or name == "gp":
        return "GP"
    if "genetic" in name or name == "ga":
        return "GA"
    if "pso" in name:
        return "PSO"
    if "ant" in name or "aco" in name:
        return "ACO"
    if "mlp" in name:
        return "MLP"
    if "perceptron" in name:
        return "Perceptron"
    if "som" in name:
        return "SOM"
    if "hopfield" in name:
        return "Hopfield"
    return name


def normalize_problem_type(ptype: str):
    """Convert various LLM outputs into known problem types."""
    p = ptype.lower()
    if "cluster" in p or "segmentation" in p:
        return "clustering"
    if "class" in p or "predict" in p:
        return "classification"
    if "combinatorial" in p or "tsp" in p:
        return "combinatorial_optimization"
    if "numerical" in p or "optimization" in p:
        return "numerical_optimization"
    return None


def main():

    user_problem = input("\nEnter problem name: ").strip().lower()
    

    if user_problem == "mall":
        user_input = """
        Perform customer segmentation on the Mall Customers dataset.
        Dataset: 200 customers with features: Age, Annual Income (k$), Spending Score (1-100).
        Goal: Discover natural customer segments for targeted marketing campaigns.
        Requirements:
          - Identify 4-6 meaningful clusters
          - Maximize cluster separation (high silhouette score)
          - Interpret clusters in business terms (e.g., "Young High Spenders")
        Constraints: Unsupervised learning (no labels available).
        Priority: Cluster interpretability over pure metric optimization.
        """
        
    elif user_problem == "titanic":
        user_input = """
        Predict passenger survival on the Titanic disaster dataset.
        Dataset: 891 training samples with features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
        Goal: Build a classifier to predict survival (binary: 0=No, 1=Yes).
        Requirements:
          - Maximize F1-score (balance precision and recall)
          - Handle missing values in Age and Embarked features
          - Encode categorical variables (Sex, Embarked)
          - Normalize numerical features
        Constraints: Must handle class imbalance (~38% survived).
        Priority: Model interpretability and generalization over raw accuracy.
        """
        
    elif user_problem == "tsp":
        user_input = """
        Solve the Traveling Salesman Problem for 30 cities.
        Problem: Find shortest tour visiting each city exactly once and returning to start.
        Input: Euclidean distance matrix for 30 randomly generated cities in 2D plane.
        Goal: Minimize total tour distance.
        Requirements:
          - Solution quality within 5% of optimal
          - Computation time under 60 seconds
          - Support permutation-based representation
        Constraints: Combinatorial optimization with O(n!) search space.
        Priority: Solution quality over speed (accuracy-critical application).
        """
        
    elif user_problem == "iris":
        user_input = """
        Cluster Iris flowers into species groups using only feature measurements.
        Dataset: 150 samples with 4 features (sepal/petal length/width), 3 true species labels.
        Goal: Discover natural groupings without using species labels during training.
        Requirements:
          - Achieve high Adjusted Rand Index (>0.8) against true labels
          - Identify exactly 3 clusters matching biological species
          - Visualize cluster separation in 2D feature space
        Constraints: Unsupervised learning (labels only for validation).
        Priority: Topological preservation of feature relationships.
        """
        
    elif user_problem == "ackley":
        user_input = """
        Optimize the Ackley function in 10 dimensions.
        Function: f(x) = -20*exp(-0.2*sqrt(1/n*Σx_i²)) - exp(1/n*Σcos(2πx_i)) + 20 + e
        Domain: x_i ∈ [-5, 5] for all dimensions.
        Global minimum: f(0,0,...,0) = 0.
        Characteristics: Large flat region surrounding a deep global optimum (hard to locate).
        Goal: Find solution with error < 1e-4 from global optimum.
        Requirements:
          - Escape flat regions effectively
          - Avoid premature convergence to local minima
          - Complete within 500 iterations
        Constraints: Continuous optimization, highly multimodal landscape.
        Priority: Solution accuracy over speed (scientific application).
        """
        
    elif user_problem == "rastrigin":
        user_input = """
        Optimize the Rastrigin function in 20 dimensions.
        Function: f(x) = 10n + Σ[x_i² - 10*cos(2πx_i)]
        Domain: x_i ∈ [-5.12, 5.12] for all dimensions.
        Global minimum: f(0,0,...,0) = 0.
        Characteristics: Highly multimodal with regular "egg-carton" landscape (many local minima).
        Goal: Find solution with error < 1e-3 from global optimum.
        Requirements:
          - Navigate through numerous local optima
          - Maintain population diversity during search
          - Complete within 1000 function evaluations
        Constraints: Continuous optimization, deceptive landscape.
        Priority: Robustness across multiple runs (low variance in results).
        """
        
    elif user_problem == "rosenbrock":
        user_input = """
        Optimize the Rosenbrock function in 30 dimensions.
        Function: f(x) = Σ[100*(x_{i+1} - x_i²)² + (1 - x_i)²]
        Domain: x_i ∈ [-5, 10] for all dimensions.
        Global minimum: f(1,1,...,1) = 0.
        Characteristics: Narrow curved valley leading to global optimum (hard convergence).
        Goal: Find solution with error < 0.1 from global optimum.
        Requirements:
          - Navigate narrow valley without oscillation
          - Maintain search direction stability
          - Complete within 2000 iterations
        Constraints: Continuous optimization, ill-conditioned landscape.
        Priority: Convergence reliability over raw speed.
        """
        
    else:
        print(f"\n⚠ Unknown problem '{user_problem}'. Using default: Mall Customers clustering.")
        user_input = """
        Perform customer segmentation on the Mall Customers dataset.
        Dataset: 200 customers with features: Age, Annual Income (k$), Spending Score (1-100).
        Goal: Discover natural customer segments for targeted marketing campaigns.
        Requirements:
          - Identify 4-6 meaningful clusters
          - Maximize cluster separation (high silhouette score)
          - Interpret clusters in business terms
        Constraints: Unsupervised learning (no labels available).
        Priority: Cluster interpretability over pure metric optimization.
        """
        user_problem = "mall"
    
    orchestrator = LLMOrchestrator()
    config = orchestrator.analyze_problem(user_input)

    raw_method = config.get("selected_method")
    method_name = normalize_method_name(raw_method)
    params = config.get("parameters", {})  
    raw_problem_type = config.get("problem_type", "")
    problem_type = normalize_problem_type(raw_problem_type)
    reasoning = config.get("reasoning", "No reasoning provided")

    print(f"\nSelected Method: {raw_method} → {method_name}")
    print(f"Problem Type: {raw_problem_type} → {problem_type}")
    print(f"Reasoning: {reasoning}\n")


    if problem_type == "combinatorial_optimization":
        problem = TSPProblem(nodes=np.random.rand(30, 2))

    elif problem_type == "numerical_optimization":
        f_name="rastrigin"
        if user_problem == 'rastrigin':
            f_name="rastrigin"
        elif user_problem =='rosenbrock':
            f_name= "rosenbrock"
        else:
            f_name="ackley"
        problem = OptimizationProblem(
            function_name=f_name,
            dimensions=10
        )

    elif problem_type == "classification":
        problem = TitanicProblem()
        problem.load_and_preprocess()

    elif problem_type == "clustering":
        if user_problem == 'mall':
            problem = ClusteringProblem(dataset_type="mall")
        else:
            problem = ClusteringProblem(dataset_type="iris")
        problem.load_data()

    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

    methods_map = {
        "GA": GeneticAlgorithm,
        "PSO": PSO,
        "ACO": ACO,
        "GP": GeneticProgramming,
        "MLP": MLP,
        "Perceptron": Perceptron,
        "SOM": SOM,
        "Hopfield": HopfieldNetwork
    }

    if method_name not in methods_map:
        print(f"⚠ Unknown method '{method_name}', falling back to GA")
        method_name = "GA"


    method = methods_map[method_name](params)

    print("Executing method...\n")

    if method_name in ["GA", "PSO", "ACO"]:
        method.fit(problem)
        best_result = method.predict()
        iterations = len(method.convergence_history)

    elif method_name == "GP":
        method.fit(problem)
        best_result = method.predict(0.5)
        iterations = params.get("generations")

    elif method_name in ["MLP", "Perceptron"]:
        method.fit(problem.X_train, problem.y_train)
        y_pred = method.predict(problem.X_test)
        best_result = problem.evaluate(problem.y_test, y_pred)
        iterations = None

    elif method_name == "SOM":
        method.fit(problem.X)
        labels = method.predict(problem.X)
        best_result = problem.evaluate(labels)
        iterations = params.get("max_epochs")

    elif method_name == "Hopfield":
        method.fit(problem.X_train[:10])
        best_result = method.predict(problem.X_train[0])
        iterations = params.get("max_iterations")

    else:
        best_result = None
        iterations = None
    print("## Results")
    print("Best Result:", best_result)
    if iterations:
        print("Iterations:", iterations)
    feedback = orchestrator.interpret_results({
        "method": method_name,
        "problem_type": problem_type,
        "result": best_result,
        "iterations": iterations
    })

    print("\n## MetaMind Feedback")
    print(feedback)


if __name__ == "__main__":
    main()