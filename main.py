import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dc_motor_env import motor_dynamics


def simulate_step_response():
    # Simulate step response
    sol = solve_ivp(motor_dynamics, [0, 2], [0, 0], args=(12,), 
                    t_eval=np.linspace(0, 2, 500))

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(sol.t, sol.y[0])
    plt.ylabel('Angular velocity (rad/s)')
    plt.xlabel('Time (s)')
    plt.title('Step Response of DC Motor')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(sol.t, sol.y[1])
    plt.ylabel('Current (A)')
    plt.xlabel('Time (s)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def run_episode(env, controller, max_steps=500, render=False):
    obs, info   = env.reset()
    controller.reset()

    # sync controller target with env's sampled target
    if hasattr(controller, 'set_target'):
        controller.set_target(env.target)

    log = dict(obs=[], actions=[], rewards=[], omega=[])
    total_reward = 0

    for _ in range(max_steps):
        act             = controller(obs)
        obs, reward, terminated, truncated, _ = env.step([act])

        log['obs'].append(obs)
        log['actions'].append(act)
        log['rewards'].append(reward)
        log['omega'].append(obs[1])

        total_reward += reward
        if terminated or truncated:
            break

    return total_reward, log

if __name__ == "__main__":
    simulate_step_response()