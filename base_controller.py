from abc import ABC, abstractmethod
import numpy as np
from simple_pid import PID
import control as ct

class BaseController(ABC):
    """Universal controller interface - all controllers implement compute()."""

    @abstractmethod
    def compute(self, obs: np.ndarray) -> float:
        """Map observation [error, omega, current] -> scalar action (V)."""
        pass

    def reset(self):
        """Reset internal state between episodes. Override when needed."""
        pass

    def set_target(self, target: float):
        """Update the speed setpoint. Override when needed."""
        pass

    def __call__(self, obs: np.ndarray) -> float:
        return self.compute(obs)

class PIDController(BaseController):
    """
    PID speed controller.
    obs = [error, omega, current],  error = target - omega  (computed by env)

    simple_pid works with a *measurement* compared to an internal setpoint.
    Fix: setpoint=0, feed -error as measurement so:
         (measurement - setpoint) = -error  ->  correct sign.
    """
    def __init__(self, Kp, Ki, Kd, setpoint=0.0,
                 output_limits=(-12.0, 12.0)):
        self._Kp = Kp; self._Ki = Ki; self._Kd = Kd
        self._output_limits = output_limits
        self._build_pid()

    def _build_pid(self):
        # pass -error as measurement to simple_pid
        self.pid = PID(self._Kp, self._Ki, self._Kd,
                       setpoint=0.0,
                       output_limits=self._output_limits)

    def compute(self, obs: np.ndarray) -> float:
        error = float(obs[0])          # error = target - omega  (from env)
        return float(self.pid(-error)) # meas = -error  -> (meas - sp=0) ✓

    def set_target(self, target: float):
        pass    # target is already embedded in obs[0] by the env

    def reset(self):
        self._build_pid()


class LQRController(BaseController):
    """
    LQR state-feedback controller.
    obs = [error, omega, current]

    State-space is built directly from motor physics so the state vector
    [omega, current] maps directly onto obs[1] and obs[2]:

        ẋ = A.x + B.u
        A = [[-b/J,  K/J],     B = [[  0  ],
             [-K/L, -R/L]]          [1/L  ]]
        x = [omega, current]

    IMPORTANT: call update_matrices() after every env.reset() so the gain K
    always matches the current randomised plant.
    """
    def __init__(self, params: dict, Q, R_lqr,
                 output_limits=(-12.0, 12.0)):
        self.Q   = np.asarray(Q,     dtype=float)
        self.R_lqr = np.asarray(R_lqr, dtype=float)
        self.output_limits = output_limits
        self.target = 0.0
        self._compute_gain(params)

    @staticmethod
    def _build_AB(params: dict):
        """Physics-based state-space: states = [omega, current]."""
        J = params['J']; b = params['b']; K = params['K']
        R = params['R']; L = params['L']
        A = np.array([[-b/J,  K/J],
                      [-K/L, -R/L]])
        B = np.array([[0.0 ],
                      [1.0/L]])
        return A, B

    def _compute_gain(self, params: dict):
        A, B = self._build_AB(params)
        self.K_lqr, _, _ = ct.lqr(A, B, self.Q, self.R_lqr)  # shape (1,2)

    def update_matrices(self, params: dict):
        """Re-compute LQR gain for a new (randomised) plant - call after reset."""
        self._compute_gain(params)

    def compute(self, obs: np.ndarray) -> float:
        omega   = float(obs[1])
        current = float(obs[2])
        x = np.array([[omega   - self.target],
                      [current - 0.0        ]])   # current reference = 0
        u = -(self.K_lqr @ x).item()              # .item() -> scalar
        return float(np.clip(u, *self.output_limits))

    def set_target(self, target: float):
        self.target = target

    def reset(self):
        pass

class BangBangController(BaseController):
    """On/off baseline - useful sanity check and imitation-learning lower bound."""
    def __init__(self, high=12.0, low=0.0):
        self.high = high
        self.low  = low

    def compute(self, obs: np.ndarray) -> float:
        return self.high if obs[0] > 0 else self.low