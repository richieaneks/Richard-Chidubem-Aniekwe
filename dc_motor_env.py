import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces

#  Nominal motor parameters (small brushed DC motor, RC-car scale) 
# DC gain ≈ 1.25 rad/s/V  ->  max speed at 12 V ≈ 15 rad/s (≈ 143 RPM)
NOMINAL_PARAMS = {
    'J': 0.0005,   # rotor inertia        (kg·m²)
    'b': 0.001,    # viscous damping       (N·m·s)
    'K': 0.8,      # motor constant Ke=Kt  (V·s/rad  or  N·m/A)
    'R': 1.0,      # armature resistance   (Ω)
    'L': 0.05,     # armature inductance   (H)
}


def motor_dynamics(t, x, V, load_torque=0.0, params=None):
    """
    DC motor electrical + mechanical ODE.
    State  x     = [omega (rad/s),  i (A)]
    Input  V     = applied voltage  (V)
    """
    # avoid mutable default dict argument
    if params is None:
        params = NOMINAL_PARAMS
    J = params['J']; b = params['b']; K = params['K']
    R = params['R']; L = params['L']

    omega, i = x
    domega = (K * i  -  b * omega  -  load_torque) / J
    di     = (V  -  K * omega  -  R * i) / L
    return [domega, di]


class DCMotorEnv(gym.Env):
    """
    DC Motor speed-control Gymnasium environment.

    Observation : [error (rad/s),  omega (rad/s),  current (A)]
                   error = target_speed - omega
    Action      : voltage  ∈ [-12, 12] V   (maps to PWM duty cycle)
    Reward      : -error² - 0.01·V²         (penalise tracking error + effort)
    """
    metadata = {'render_modes': []}

    def __init__(self, seed: int = 42):
        super().__init__()
        self.seed_val = seed
        self.rng      = np.random.default_rng(seed=self.seed_val)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-12.0, high=12.0, shape=(1,), dtype=np.float32)

        self.dt     = 0.01          # control timestep (s)
        self.state  = np.zeros(2)   # [omega, i]
        self.target = 0.0           # target speed (rad/s)

        self._domain_randomization()

    #  Core Gym interface 
    def step(self, action):
        action = np.asarray(action, dtype=float).flatten()
        V = float(np.clip(action[0], -12.0, 12.0))

        # apply PWM deadband (was randomised but never enforced)
        if abs(V) < self.deadband:
            V = 0.0

        sol = solve_ivp(
            motor_dynamics, [0.0, self.dt], self.state,
            args=(V, self.load_torque, self.params),
            dense_output=False)

        self.state = sol.y[:, -1]
        omega, current = self.state

        error  = self.target - omega
        reward = -(error ** 2) - 0.01 * (V ** 2)

        obs = np.array([error, omega, current], dtype=np.float32)
        return obs, float(reward), False, False, {}

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed=seed)

        self._domain_randomization()
        self.state  = np.zeros(2)

        # target within plant capability: 0–12 rad/s  (≈ 0–115 RPM)
        self.target = float(self.rng.uniform(2.0, 12.0))

        # obs[0] = error = target - 0 = target (not target itself)
        obs = np.array([self.target, 0.0, 0.0], dtype=np.float32)
        return obs, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    #  Helpers 
    def _domain_randomization(self):
        """Sample randomised motor parameters and disturbances each episode."""
        self.J = float(self.rng.uniform(0.0004, 0.0007))    # ±~20 %
        self.b = float(self.rng.uniform(0.0007, 0.0015))
        self.K = float(self.rng.uniform(0.65,   0.95))
        self.R = float(self.rng.uniform(0.8,    1.2))
        self.L = float(self.rng.uniform(0.04,   0.06))
        self.params = {'J': self.J, 'b': self.b, 'K': self.K,
                       'R': self.R, 'L': self.L}

        self.load_torque = float(self.rng.uniform(0.0,  0.002))
        self.deadband    = float(self.rng.uniform(0.0,  0.3))

    def get_state_space_matrices(self):
        """
        Return (A, B) linearised state-space matrices for the *current*
        randomised parameters.  Call this after every env.reset() so that
        the LQR controller recomputes its gain K for the active plant.
        """
        import control as ct
        s = ct.TransferFunction.s
        K, J, b, R, L = self.K, self.J, self.b, self.R, self.L
        G = K / ((J * s + b) * (L * s + R) + K ** 2)
        A, B, _, _ = ct.ssdata(ct.tf2ss(G))
        return A, B