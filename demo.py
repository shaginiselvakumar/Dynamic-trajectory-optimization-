"""
COMPREHENSIVE DEMO
Demonstrates all features of the trajectory optimization system
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

from config import Config
from environment import SimulationEnvironment
from realtime_system import RealtimeNavigationSystem
from multi_agent import MultiAgentSystem
from cost_function import CostFunction, MultiAgentCostFunction
from sensor_perception import SensorPerception
from terrain_predictor import TerrainPredictor, generate_terrain_training_data

# =========================
# Project-local Output Path
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams['font.family'] = 'DejaVu Sans'


def savefig(path: Path):
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()


def demo_1_basic_navigation():
    """
    Demo 1: Basic single-agent navigation with Douglas-Rachford
    """
    print("\n" + "="*70)
    print("DEMO 1: BASIC NAVIGATION WITH DOUGLAS-RACHFORD OPTIMIZATION")
    print("="*70)

    start = np.array([10, 10])
    goal = np.array([85, 85])

    system = RealtimeNavigationSystem(start, goal, use_terrain_predictor=False)
    system.run_simulation(max_time=40.0, verbose=True)
    system.visualize_results(save_path=str(OUT_DIR / "demo1_basic.png"))

    print("\nDemo 1 complete! Results saved to:", OUT_DIR / "demo1_basic.png")


def demo_2_with_terrain_prediction():
    """
    Demo 2: Navigation with AI terrain cost prediction
    """
    print("\n" + "="*70)
    print("DEMO 2: AI-ENHANCED TERRAIN PREDICTION")
    print("="*70)

    start = np.array([15, 15])
    goal = np.array([80, 80])

    system = RealtimeNavigationSystem(start, goal, use_terrain_predictor=True)
    system.run_simulation(max_time=40.0, verbose=True)
    system.visualize_results(save_path=str(OUT_DIR / "demo2_ai_terrain.png"))

    print("\nDemo 2 complete! Results saved to:", OUT_DIR / "demo2_ai_terrain.png")


def demo_3_energy_constrained():
    """
    Demo 3: Energy-constrained mission with recovery zones
    """
    print("\n" + "="*70)
    print("DEMO 3: ENERGY-CONSTRAINED NAVIGATION")
    print("="*70)

    start = np.array([10, 10])
    goal = np.array([90, 90])

    system = RealtimeNavigationSystem(start, goal, use_terrain_predictor=True)
    system.energy.reset(battery_level=60.0)

    print("\nStarting with reduced battery (60%)")
    print("Robot must use recovery zones strategically!")

    system.run_simulation(max_time=60.0, verbose=True)
    system.visualize_results(save_path=str(OUT_DIR / "demo3_energy.png"))

    print("\nDemo 3 complete! Results saved to:", OUT_DIR / "demo3_energy.png")


def demo_4_multi_agent():
    """
    Demo 4: Multi-agent coordinated navigation
    """
    print("\n" + "="*70)
    print("DEMO 4: MULTI-AGENT COORDINATION")
    print("="*70)

    env = SimulationEnvironment()
    multi_agent = MultiAgentSystem(env, num_agents=3)

    starts = [
        np.array([10, 10]),
        np.array([10, 90]),
        np.array([90, 50])
    ]

    goals = [
        np.array([90, 90]),
        np.array([90, 10]),
        np.array([10, 50])
    ]

    multi_agent.initialize_agents(starts, goals)

    cost_fns = []
    sensors = []
    for agent in multi_agent.agents:
        cost_fn = MultiAgentCostFunction(env, agent['energy'],
                                         num_agents=multi_agent.num_agents)
        sensor = SensorPerception(env)
        cost_fns.append(cost_fn)
        sensors.append(sensor)

    print("\nOptimizing trajectories for 3 agents...")
    trajectories, info = multi_agent.optimize_all_trajectories(
        cost_fns, sensors, verbose=True)

    fig, ax = plt.subplots(figsize=(14, 14))
    env.visualize(ax=ax)

    for i, (agent, traj) in enumerate(zip(multi_agent.agents, trajectories)):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=agent['color'],
                linewidth=4, label=f'Agent {i}', alpha=0.8)
        ax.plot(starts[i][0], starts[i][1], 'o', color=agent['color'],
                markersize=15, markeredgecolor='black', markeredgewidth=2)
        ax.plot(goals[i][0], goals[i][1], '*', color=agent['color'],
                markersize=20, markeredgecolor='black', markeredgewidth=2)

        for j in range(i + 1, multi_agent.num_agents):
            min_dist = min(
                np.linalg.norm(traj[t] - trajectories[j][t])
                for t in range(len(traj))
            )
            print(f"Min distance between Agent {i} and Agent {j}: {min_dist:.2f} m")

    ax.legend(fontsize=12)
    ax.set_title('Multi-Agent Coordinated Trajectories\n(Collision Avoidance Active)',
                 fontsize=14, fontweight='bold')

    savefig(OUT_DIR / "demo4_multi_agent.png")
    print("\nDemo 4 complete! Results saved to:", OUT_DIR / "demo4_multi_agent.png")


def demo_5_robustness_test():
    """
    Demo 5: Robustness under disturbances and noise
    """
    print("\n" + "="*70)
    print("DEMO 5: ROBUSTNESS UNDER DISTURBANCES")
    print("="*70)

    start = np.array([20, 20])
    goal = np.array([80, 80])

    original_config = Config()
    config = Config()
    config.WIND_STRENGTH = 1.5
    config.EXTERNAL_FORCE_STD = 0.8
    config.SENSOR_NOISE_STD = 0.5

    print("\nTesting with high disturbances:")
    print(f"Wind strength: {config.WIND_STRENGTH} (was {original_config.WIND_STRENGTH})")
    print(f"Sensor noise:  {config.SENSOR_NOISE_STD} (was {original_config.SENSOR_NOISE_STD})")

    system = RealtimeNavigationSystem(start, goal, use_terrain_predictor=True)
    system.config = config
    system.environment.config = config
    system.sensor.config = config

    system.run_simulation(max_time=50.0, verbose=True)
    system.visualize_results(save_path=str(OUT_DIR / "demo5_robustness.png"))

    print("\nDemo 5 complete! Results saved to:", OUT_DIR / "demo5_robustness.png")


def demo_6_comparison():
    """
    Demo 6: Comparison of different scenarios
    """
    print("\n" + "="*70)
    print("DEMO 6: COMPARATIVE ANALYSIS")
    print("="*70)

    start = np.array([15, 15])
    goal = np.array([85, 85])

    scenarios = {
        'Without AI Terrain': False,
        'With AI Terrain': True
    }

    results = {}

    for scenario_name, use_ai in scenarios.items():
        print(f"\n--- Running: {scenario_name} ---")
        system = RealtimeNavigationSystem(start, goal, use_terrain_predictor=use_ai)
        system.run_simulation(max_time=40.0, verbose=False)

        results[scenario_name] = {
            'path_length': system.metrics['path_length'],
            'energy_consumed': system.energy.total_consumed,
            'replans': system.metrics['total_replans'],
            'avg_opt_time': np.mean(system.metrics['optimization_times']) if system.metrics['optimization_times'] else 0,
            'mission_complete': system.mission_complete
        }

    print("\n" + "="*70)
    print("COMPARATIVE RESULTS")
    print("="*70)

    print(f"\n{'Metric':<25} {'Without AI':<15} {'With AI':<15} {'Improvement':<15}")
    print("-" * 70)

    for metric in ['path_length', 'energy_consumed', 'replans', 'avg_opt_time']:
        val1 = results['Without AI Terrain'][metric]
        val2 = results['With AI Terrain'][metric]

        if metric == 'avg_opt_time':
            improvement = ((val1 - val2) / val1 * 100) if val1 > 0 else 0
            print(f"{metric:<25} {val1*1000:>10.2f} ms {val2*1000:>10.2f} ms {improvement:>10.1f}%")
        else:
            improvement = ((val1 - val2) / val1 * 100) if val1 > 0 else 0
            print(f"{metric:<25} {val1:>14.2f} {val2:>14.2f} {improvement:>10.1f}%")

    print("\nDemo 6 complete!")


def run_all_demos():
    """
    Run all demonstration scenarios
    """
    print("\n" + "="*80)
    print(" " * 15 + "DOUGLAS-RACHFORD TRAJECTORY OPTIMIZATION")
    print(" " * 20 + "COMPREHENSIVE DEMONSTRATION")
    print("="*80)

    demos = [
        ("Basic Navigation", demo_1_basic_navigation),
        ("AI Terrain Prediction", demo_2_with_terrain_prediction),
        ("Energy Constraints", demo_3_energy_constrained),
        ("Multi-Agent", demo_4_multi_agent),
        ("Robustness Test", demo_5_robustness_test),
        ("Comparative Analysis", demo_6_comparison)
    ]

    for i, (name, demo_func) in enumerate(demos, 1):
        print(f"\n\n{'#'*80}")
        print(f"# Running Demo {i}/{len(demos)}: {name}")
        print(f"{'#'*80}")

        try:
            demo_func()
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*80)
    print("ALL DEMOS COMPLETE!")
    print("="*80)
    print("\nGenerated files in:", OUT_DIR)


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        demo_map = {
            '1': demo_1_basic_navigation,
            '2': demo_2_with_terrain_prediction,
            '3': demo_3_energy_constrained,
            '4': demo_4_multi_agent,
            '5': demo_5_robustness_test,
            '6': demo_6_comparison,
            'all': run_all_demos
        }

        if demo_num in demo_map:
            demo_map[demo_num]()
        else:
            print(f"Unknown demo: {demo_num}")
            print("Available: 1, 2, 3, 4, 5, 6, all")
    else:
        run_all_demos()
