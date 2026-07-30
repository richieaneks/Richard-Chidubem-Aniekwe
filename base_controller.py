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
                 output_limits=(-12.0, 12.0), sample_time=None):
        self._Kp = Kp; self._Ki = Ki; self._Kd = Kd
        self._output_limits = output_limits
        self._sample_time = sample_time
        self._build_pid()

    def _build_pid(self):
        # pass -error as measurement to simple_pid
        self.pid = PID(self._Kp, self._Ki, self._Kd,
                       setpoint=0.0,
                       output_limits=self._output_limits,
                       sample_time=self._sample_time)

    def compute(self, obs: np.ndarray) -> float:
        error = float(obs[0])          # error = target - omega  (from env)
        # Pass the simulated dt explicitly - simple_pid defaults to wall-clock
        # time.monotonic(), which is unrelated to env.dt and jitters wildly
        # relative to it, corrupting the integral/derivative terms.
        dt = self._sample_time
        return float(self.pid(-error, dt=dt)) # meas = -error  -> (meas - sp=0) ✓

    def set_target(self, target: float):
        pass    # target is already embedded in obs[0] by the env

    def reset(self):
        self._build_pid()

    def set_sample_time(self, dt: float):
        """Set controller sample time (s). Useful to synchronise with env.dt."""
        self._sample_time = float(dt)
        if hasattr(self, 'pid') and self.pid is not None:
            self.pid.sample_time = self._sample_time


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
                 output_limits=(-12.0, 12.0), dt=0.01):
        self.Q   = np.asarray(Q,     dtype=float)
        self.R_lqr = np.asarray(R_lqr, dtype=float)
        self.output_limits = output_limits
        self.target = 0.0
        self.dt = float(dt)
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
        # The controller only actually acts once per env.dt (zero-order hold
        # at ~100 Hz), while this plant's electrical/mechanical resonance
        # sits at ~25 Hz (zeta ~0.07). A gain from continuous-time ct.lqr()
        # assumes continuous actuation and, applied via ZOH at this dt, is
        # unstable (closed-loop discrete pole magnitude >> 1). Design on the
        # dt-discretized plant instead so the gain matches how it's applied.
        sysd = ct.c2d(ct.ss(A, B, np.eye(2), np.zeros((2, 1))), self.dt)
        Ad, Bd = sysd.A, sysd.B
        self.K_lqr, _, _ = ct.dlqr(Ad, Bd, self.Q, self.R_lqr)  # shape (1,2)

        # Reference feedforward (Nbar): driving x -> [target, 0] has no
        # reason to settle at omega=target, since [target, 0] generally
        # isn't an equilibrium of the plant - holding a nonzero speed
        # against damping needs a nonzero steady-state current, not 0.
        # Solve for the true (x_ss, u_ss) that makes omega_ss = target and
        # feed forward through Nbar so the regulator has zero steady-state
        # error instead of quietly settling wherever K happens to point.
        C = np.array([[1.0, 0.0]])
        n = Ad.shape[0]
        M = np.block([[np.eye(n) - Ad, -Bd],
                      [C, np.zeros((1, 1))]])
        rhs = np.zeros((n + 1, 1)); rhs[-1, 0] = 1.0
        x_ss, u_ss = np.split(np.linalg.solve(M, rhs), [n])
        self.Nbar = float((self.K_lqr @ x_ss + u_ss).item())

    def update_matrices(self, params: dict):
        """Re-compute LQR gain for a new (randomised) plant - call after reset."""
        self._compute_gain(params)

    def set_sample_time(self, dt: float):
        """Sync controller dt with env.dt (called by run_episode); redesigns
        the discrete gain since it depends on dt."""
        self.dt = float(dt)

    def compute(self, obs: np.ndarray) -> float:
        omega   = float(obs[1])
        current = float(obs[2])
        x = np.array([[omega], [current]])
        u = (-self.K_lqr @ x + self.Nbar * self.target).item()
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