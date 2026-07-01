import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dc_motor_env import motor_dynamics, DCMotorEnv, NOMINAL_PARAMS
from base_controller import PIDController, LQRController, BangBangController
import control as ct


# Open-loop step response 
def simulate_step_response():
    """Visualise plant behaviour before closing the loop."""
    sol = solve_ivp(motor_dynamics, [0, 2], [0, 0],
                    args=(12.0, 0.0, NOMINAL_PARAMS),
                    t_eval=np.linspace(0, 2, 500))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(sol.t, sol.y[0], color='steelblue', linewidth=2)
    axes[0].set_ylabel('Angular velocity  (rad/s)')
    axes[0].set_title('Open-Loop Step Response  (V = 12 V, nominal params)')
    axes[0].grid(True)

    axes[1].plot(sol.t, sol.y[1], color='darkorange', linewidth=2)
    axes[1].set_ylabel('Current  (A)')
    axes[1].set_xlabel('Time  (s)')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('outputs/step_response.png', dpi=150)
    plt.close()
    print("Saved: step_response.png")


# Episode runner 
def run_episode(env: DCMotorEnv, controller, max_steps: int = 500):
    """
    Controller-agnostic episode loop.
    The ONLY coupling point is:   act = controller(obs)
    Everything else (env reset, logging) is shared.
    """
    obs, _ = env.reset()
    controller.reset()

    # Sync setpoint
    if hasattr(controller, 'set_target'):
        controller.set_target(env.target)

    # update LQR gain for the current randomised plant params
    if isinstance(controller, LQRController):
        controller.update_matrices(env.params)   # env.params = current episode params

    log = dict(omega=[], actions=[], rewards=[])
    total_reward = 0.0

    for _ in range(max_steps):
        act = controller(obs)                                    # ← swappable
        obs, reward, terminated, truncated, _ = env.step([act])

        log['omega'].append(float(obs[1]))
        log['actions'].append(float(act))
        log['rewards'].append(float(reward))
        total_reward += reward

        if terminated or truncated:
            break

    return total_reward, log


# Benchmark 
def benchmark(controllers: dict, env: DCMotorEnv, max_steps: int = 500):
    """
    Each controller faces the *same* plant and setpoint (same RNG seed).
    """
    results = {}
    for name, ctrl in controllers.items():
        env.rng = np.random.default_rng(seed=env.seed_val)   # rewind RNG
        reward, log = run_episode(env, ctrl, max_steps=max_steps)
        results[name] = log
        print(f"  {name:<12s}| reward {reward:>12.1f} "
              f"| final ω {log['omega'][-1]:5.2f} rad/s "
              f"| target  {env.target:.2f} rad/s")
    return results


# Plotting 
def plot_results(results: dict, env: DCMotorEnv):
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = plt.cm.tab10.colors

    for idx, (name, log) in enumerate(results.items()):
        t = np.arange(len(log['omega'])) * env.dt
        axes[0].plot(t, log['omega'],   label=name,
                     color=colors[idx], linewidth=1.8)
        axes[1].plot(t, log['actions'], label=name,
                     color=colors[idx], linewidth=1.8)

    axes[0].axhline(env.target, color='k', linestyle='--', linewidth=1.5,
                    label=f'Setpoint  ({env.target:.2f} rad/s)')
    axes[0].set_ylabel('Angular velocity  (rad/s)')
    axes[0].set_title('Speed Tracking — Controller Comparison')
    axes[0].legend(loc='lower right')
    axes[0].grid(True)

    axes[1].set_ylabel('Action / Voltage  (V)')
    axes[1].set_xlabel('Time  (s)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('outputs/controller_comparison.png', dpi=150)
    plt.close()
    print("Saved: controller_comparison.png")


#  Entry point 
if __name__ == "__main__":
    os.mkdir("outputs", exist_ok=True)

    # Open-loop step response (nominal plant)
    simulate_step_response()

    # Environment
    env = DCMotorEnv(seed=42)

    # Controllers — all share the BaseController interface
    #    LQR receives nominal params for initialisation;
    #    gains are recomputed per-episode in run_episode()
    controllers = {
        "PID"      : PIDController(Kp=2.0, Ki=5.0, Kd=0.005,
                                   output_limits=(-12.0, 12.0)),
        "LQR"      : LQRController(NOMINAL_PARAMS,
                                   Q=np.diag([100.0, 1.0]),
                                   R_lqr=[[0.1]],
                                   output_limits=(-12.0, 12.0)),
        "BangBang" : BangBangController(high=12.0, low=0.0),
    }

    # Benchmark
    print("\n Benchmark (same seed -> same plant & target per controller) ")
    results = benchmark(controllers, env, max_steps=500)

    # Plot
    plot_results(results, env)