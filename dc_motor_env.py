import numpy as np
from scipy.integrate import solve_ivp

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

