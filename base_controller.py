from abc import ABC, abstractmethod
import numpy as np
from simple_pid import PID
import control as ct

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

class LQRController(BaseController):
    def __init__(self, A, B, Q, R_lqr, output_limits=(-12, 12)):
        self.K, _, _ = ct.lqr(A, B, Q, R_lqr)
        self.output_limits = output_limits
        self.target = 0.0

    def compute(self, obs):
        # obs = [error, omega, current]  — use full state [omega, current]
        omega, current = obs[1], obs[2]
        x = np.array([[omega - self.target], [current]])
        u = -self.K @ x
        return float(np.clip(u, *self.output_limits))

    def set_target(self, target):
        self.target = target

    def reset(self):
        pass