from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class CIMethod(ABC):
    """
    Abstract base class enforcing the Standardized API.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Initialize with parameters selected by LLM or defaults.
        Logs initialization.
        """
        self.params = params
        self.logs = []
        self.convergence_history = []
        
    @abstractmethod
    def fit(self, data: Any, targets: Optional[Any] = None) -> 'CIMethod':
        """
        Execute the method (train/optimize). 
        Must implement Progress Callback and Logging.
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> Any:
        """
        Generate predictions or return the best solution found.
        """
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """
        Return execution metrics (time, iterations, error)[cite: 44].
        """
        return {
            "convergence_history": self.convergence_history,
            "logs": self.logs
        }

    def validate_parameters(self, required_params: List[str]) -> bool:
        """
        Validate parameters before execution.
        """
        for param in required_params:
            if param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")
        return True