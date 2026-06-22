import numpy as np
from scipy.integrate import solve_ivp
import gymnasium as gym
from gymnasium import spaces

# Motor parameters (typical small DC motor)
J = 0.01    # rotor inertia (kg.m^2)
b = 0.1     # damping coefficient (N.m.s)
K = 0.01    # motor constant (Ke = Kt = K)
R = 1.0     # armature resistance (Ohm)
L = 0.5     # armature inductance (H)

def motor_dynamics(t, x, V, load_torque=0):
    omega, i = x          # state: angular velocity, current
    domega = (K*i - b*omega - load_torque) / J
    di = (V - K*omega - R*i) / L
    return [domega, di]


class DCMotorEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))  # [error, omega, current]
        self.action_space = spaces.Box(low=-12, high=12, shape=(1,))  # voltage PWM equiv
        self.dt = 0.01
        self.state = np.zeros(2)
    def step(self, action):
        sol = solve_ivp(motor_dynamics, [0, self.dt], self.state, args=(action[0],))
        self.state = sol.y[:, -1]
        error = self.target - self.state[0]
        reward = -error**2 - 0.01*action[0]**2  # penalize error + control effort
        obs = np.array([error, self.state[0], self.state[1]])
        return obs, reward, False, False, {}
    
    def reset(self, seed=None):
        self.state = np.zeros(2)
        self.target = np.random.uniform(50, 150)  # random target speed
        return np.array([self.target, 0, 0]), {}
    def render(self, mode='human'):
        pass

    def close(self):
        pass