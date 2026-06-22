from abc import ABC, abstractmethod
import numpy as np
from simple_pid import PID

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

class PIDController(BaseController):
    def __init__(self, Kp, Ki, Kd, setpoint=0, output_limits=(-12, 12)):
        self.pid = PID(Kp, Ki, Kd, setpoint=setpoint, output_limits=output_limits)

    def compute(self, obs):
        # obs = [error, omega, current]
        error = obs[0]
        return self.pid(error)           # simple_pid takes measurement, not error
                                         # so pass -error or restructure as needed

    def set_target(self, target):
        self.pid.setpoint = target

    def reset(self):
        self.pid.reset()