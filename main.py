import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dc_motor_env import motor_dynamics, DCMotorEnv
from base_controller import PIDController, LQRController
import control as ct


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
    # simulate_step_response()
    env     = DCMotorEnv()
    # PID parameters
    Kp = 1.2
    Ki = 0.5
    Kd = 0.01
    # Motor parameters
    J, b, K, R, L = env.J, env.b, env.K, env.R, env.L

    # LQR parameters
    Q = np.diag([10,1])
    R_lqr=[[0.1]]

    s = ct.TransferFunction.s
    G = K / ((J*s + b)*(L*s + R) + K**2)
    t, y = ct.step_response(G)

    # LQR
    A, B, C, D = ct.ssdata(ct.tf2ss(G))

    controllers = {
    "PID"      : PIDController(Kp=Kp, Ki=Ki, Kd=Kd),
    "LQR"      : LQRController(A, B, Q, R_lqr), # there might be need to update 
                                                # the A & B because of domain randomization
}

    results = {}

    for name, ctrl in controllers.items():
        reward, log = run_episode(env, ctrl)
        results[name] = log
        print(f"{name:12s} | Total reward: {reward:.2f}")

    #  plot speed tracking
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for name, log in results.items():
        t = np.arange(len(log['omega'])) * env.dt
        axes[0].plot(t, log['omega'],   label=name)
        axes[1].plot(t, log['actions'], label=name)

    axes[0].axhline(env.target, color='k', linestyle='--', label='Setpoint')
    axes[0].set_ylabel('Speed (RPM)')
    axes[0].legend()
    axes[0].set_title('Speed Tracking')
    axes[1].set_ylabel('Action (V)')
    axes[1].legend()
    axes[1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
