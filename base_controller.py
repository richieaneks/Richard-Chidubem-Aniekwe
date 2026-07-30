from abc import ABC, abstractmethod
import numpy as np
from simple_pid import PID
from scipy.optimize import minimize
import control as ct


def _motor_AB(params: dict):
    """Physics-based continuous state-space: states = [omega, current]."""
    J = params['J']; b = params['b']; K = params['K']
    R = params['R']; L = params['L']
    A = np.array([[-b/J,  K/J],
                  [-K/L, -R/L]])
    B = np.array([[0.0 ],
                  [1.0/L]])
    return A, B


def _discretize(A, B, dt):
    """Zero-order-hold discretization at sample time dt (control acts once
    per dt, so gains/models designed on the continuous plant and applied via
    ZOH can be unstable - see LQRController)."""
    sysd = ct.c2d(ct.ss(A, B, np.eye(2), np.zeros((2, 1))), dt)
    return sysd.A, sysd.B


def _tracking_equilibrium(Ad, Bd):
    """
    Unit-reference equilibrium (x_ss, u_ss) solving x_ss = Ad x_ss + Bd u_ss,
    omega_ss = 1. Holding a nonzero speed against damping needs a nonzero
    steady-state current, not 0 - scaling this by the actual target gives
    the (state, input) the plant must sit at, used as feedforward so
    state-feedback controllers have zero steady-state error.
    """
    C = np.array([[1.0, 0.0]])
    n = Ad.shape[0]
    M = np.block([[np.eye(n) - Ad, -Bd],
                  [C, np.zeros((1, 1))]])
    rhs = np.zeros((n + 1, 1)); rhs[-1, 0] = 1.0
    x_ss, u_ss = np.split(np.linalg.solve(M, rhs), [n])
    return x_ss, u_ss


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

    def _compute_gain(self, params: dict):
        A, B = _motor_AB(params)
        # The controller only actually acts once per env.dt (zero-order hold
        # at ~100 Hz), while this plant's electrical/mechanical resonance
        # sits at ~25 Hz (zeta ~0.07). A gain from continuous-time ct.lqr()
        # assumes continuous actuation and, applied via ZOH at this dt, is
        # unstable (closed-loop discrete pole magnitude >> 1). Design on the
        # dt-discretized plant instead so the gain matches how it's applied.
        Ad, Bd = _discretize(A, B, self.dt)
        self.K_lqr, _, _ = ct.dlqr(Ad, Bd, self.Q, self.R_lqr)  # shape (1,2)

        # Reference feedforward (Nbar): driving x -> [target, 0] has no
        # reason to settle at omega=target, since [target, 0] generally
        # isn't an equilibrium of the plant. Feed forward through Nbar so
        # the regulator has zero steady-state error instead of quietly
        # settling wherever K happens to point.
        x_ss, u_ss = _tracking_equilibrium(Ad, Bd)
        self.Nbar = float((self.K_lqr @ x_ss + u_ss).item())

    def update_matrices(self, params: dict):
        """Re-compute LQR gain for a new (randomised) plant - call after reset."""
        self._compute_gain(params)

    def set_sample_time(self, dt: float):
        """Sync controller dt with env.dt (called by run_episode). Only
        takes effect on the next update_matrices() call, since the gain
        depends on dt - run_episode calls this before update_matrices()."""
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


class MPCController(BaseController):
    """
    Linear MPC speed controller.
    obs = [error, omega, current]

    Like LQRController, re-derives the discrete plant model each episode
    (via update_matrices) so it always matches the current randomised
    plant. Unlike LQR/PID, the actuator limits are enforced *inside* the
    optimization over a receding horizon rather than clipped after the
    fact, which is what lets it settle with almost no overshoot even
    though this plant is a near-undamped resonance (zeta ~0.07) that
    saturates the actuator from a cold start.

    Each step solves:
        min_{u_0..u_{N-1}}  sum_k (x_k - x_ref)' Q (x_k - x_ref) + R u_k^2
        s.t. x_{k+1} = Ad x_k + Bd u_k,   u_k in output_limits
    as a small dense QP (state has no constraints, so this reduces to a
    box-constrained QP - solved with scipy instead of pulling in a QP
    library like cvxpy), applies u_0, and re-solves next step (receding
    horizon).
    """
    def __init__(self, params: dict, Q, R_mpc, horizon=15,
                 output_limits=(-12.0, 12.0), dt=0.01):
        self.Q = np.asarray(Q, dtype=float)
        self.R_mpc = float(np.asarray(R_mpc).reshape(()))
        self.horizon = int(horizon)
        self.output_limits = output_limits
        self.target = 0.0
        self.dt = float(dt)
        self._u_prev = None
        self._build_model(params)

    def _build_model(self, params: dict):
        A, B = _motor_AB(params)
        Ad, Bd = _discretize(A, B, self.dt)
        self.Ad, self.Bd = Ad, Bd
        self._x_ss_unit, _ = _tracking_equilibrium(Ad, Bd)  # per unit target

        n, N = 2, self.horizon
        # Prediction matrices: X = Sx.x0 + Su.U, stacking x_1..x_N and
        # u_0..u_{N-1}. Built once per episode (params fixed within an
        # episode), not per step - only compute() runs every step.
        Sx = np.zeros((n * N, n))
        Su = np.zeros((n * N, N))
        for row in range(N):
            Sx[row*n:(row+1)*n, :] = np.linalg.matrix_power(Ad, row + 1)
            for col in range(row + 1):
                Su[row*n:(row+1)*n, col:col+1] = np.linalg.matrix_power(Ad, row - col) @ Bd

        Qbar = np.kron(np.eye(N), self.Q)
        Rbar = np.eye(N) * self.R_mpc
        self._Sx = Sx
        self._QbarSu = Qbar @ Su            # reused each step to build f
        self._H = 2.0 * (Su.T @ Qbar @ Su + Rbar)   # constant within an episode

    def update_matrices(self, params: dict):
        """Re-derive the discrete model/QP for a new (randomised) plant -
        call after reset(), same contract as LQRController."""
        self._build_model(params)

    def set_sample_time(self, dt: float):
        self.dt = float(dt)

    def set_target(self, target: float):
        self.target = float(target)

    def compute(self, obs: np.ndarray) -> float:
        omega, current = float(obs[1]), float(obs[2])
        x0 = np.array([[omega], [current]])
        x_ref = self._x_ss_unit * self.target
        x_ref = np.tile(x_ref, (self.horizon, 1))

        e0 = self._Sx @ x0 - x_ref
        f = 2.0 * (self._QbarSu.T @ e0).flatten()

        lo, hi = self.output_limits
        u_init = self._u_prev if self._u_prev is not None \
            else np.zeros(self.horizon)

        def cost_and_grad(U):
            return 0.5 * U @ self._H @ U + f @ U, self._H @ U + f

        result = minimize(cost_and_grad, u_init, jac=True,
                          method='L-BFGS-B', bounds=[(lo, hi)] * self.horizon)
        U = result.x
        # Warm-start next step from the tail of this step's plan (receding
        # horizon: shift left, repeat the last predicted input).
        self._u_prev = np.concatenate([U[1:], U[-1:]])
        return float(np.clip(U[0], lo, hi))

    def reset(self):
        self._u_prev = None


class BangBangController(BaseController):
    """On/off baseline - useful sanity check and imitation-learning lower bound."""
    def __init__(self, high=12.0, low=0.0):
        self.high = high
        self.low  = low

    def compute(self, obs: np.ndarray) -> float:
        return self.high if obs[0] > 0 else self.low