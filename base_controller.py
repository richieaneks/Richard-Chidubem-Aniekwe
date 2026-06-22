from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """All controllers share this interface"""
    
    @abstractmethod
    def compute(self, obs: np.ndarray) -> float:
        """obs -> action (PWM/voltage)"""
        pass
    
    def reset(self):
        """Reset internal state between episodes (override if needed)"""
        pass

    def __call__(self, obs: np.ndarray) -> float:
        return self.compute(obs)