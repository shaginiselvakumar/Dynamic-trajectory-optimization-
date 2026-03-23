# Real-Time Trajectory Optimization using Douglas–Rachford Splitting

## Project Overview

This project implements a **real-time autonomous navigation system** that computes optimal trajectories using the **Douglas–Rachford splitting method** under:
- ✅ Energy constraints
- ✅ Dynamic obstacles
- ✅ Disturbances (wind, sensor noise)
- ✅ Multi-agent coordination

### Key Features

1. **Douglas–Rachford Optimization** 
2. - Novel application of DR splitting to trajectory planning
2. **Energy-Aware Planning** - Battery consumption with recovery zones
3. **AI Terrain Prediction** - Neural network learns terrain traversal costs
4. **Multi-Agent Coordination** - Simultaneous optimization with collision avoidance
5. **Robustness** - Handles disturbances, sensor noise, and dynamic obstacles
6. **Real-Time Execution** - Continuous replanning during mission

---

## Installation

### Requirements
- Python 3.8+
- NumPy
- Matplotlib

### Quick Start

```bash
# Install dependencies
pip install numpy matplotlib

# Run basic demo
python demo.py 1

# Run all demos
python demo.py all
```

---

## Project Structure

```
├── config.py                 # Configuration parameters
├── environment.py            # Simulation environment (Module 1)
├── motion_engine.py          # Robot kinematics (Module 2)
├── sensor_perception.py      # Sensor simulation (Module 3)
├── energy_model.py           # Battery dynamics (Module 6)
├── cost_function.py          # Multi-objective cost (Module 5)
├── douglas_rachford.py       # DR optimizer (Module 7) ⭐ CORE
├── terrain_predictor.py      # AI terrain model (Feature 1)
├── multi_agent.py            # Multi-agent system (Feature 3)
├── realtime_system.py        # Main simulation loop
└── demo.py                   # Demonstration scenarios
```

---

## Core Modules

### Module 1: Simulation Environment
Creates virtual world with:
- Static/dynamic obstacles
- Terrain zones (rough, smooth, normal)
- Energy recovery zones
- Wind field (disturbances)

```python
from environment import SimulationEnvironment

env = SimulationEnvironment()
env.visualize()
```

### Module 2: Robot Motion Engine
Handles robot physics:
- Position/velocity/acceleration
- Kinematic constraints
- Trajectory prediction

```python
from motion_engine import RobotMotionEngine

robot = RobotMotionEngine(initial_position=[10, 10])
robot.apply_control(desired_velocity=[2.0, 1.5], dt=0.1)
```

### Module 3: Sensor & Perception
Realistic sensor simulation:
- Limited range detection
- Gaussian noise
- Occlusion modeling
- Occupancy grid mapping

```python
from sensor_perception import SensorPerception

sensor = SensorPerception(environment)
detected_obstacles = sensor.sense_obstacles(robot_position)
```

### Module 6: Energy Model
Battery dynamics:
- Motion-based consumption
- Terrain cost multipliers
- Recovery zones
- Feasibility checking

```python
from energy_model import EnergyModel

energy = EnergyModel(initial_battery=100.0)
energy.update(velocity, acceleration, dt, in_recovery_zone=True)
```

### Module 5: Cost Function
Multi-objective optimization:
- J(X) = w₁·Length + w₂·Energy + w₃·Smoothness + w₄·Obstacle + w₅·Goal

```python
from cost_function import CostFunction

cost_fn = CostFunction(environment, energy_model)
total_cost = cost_fn.compute_total_cost(trajectory, goal_position)
```

### Module 7: Douglas–Rachford Optimizer ⭐
**CORE CONTRIBUTION**

Solves: `min f(X) + g(X)`
- f(X) = smooth cost function
- g(X) = non-smooth constraints

**Algorithm:**
```
1. x_{k+1} = prox_g(z_k)                    # Project onto constraints
2. y_{k+1} = prox_f(2x_{k+1} - z_k)         # Minimize cost
3. z_{k+1} = z_k + λ(y_{k+1} - x_{k+1})     # Update dual variable
```

```python
from douglas_rachford import DouglasRachfordOptimizer

optimizer = DouglasRachfordOptimizer(cost_fn, environment, energy_model)
optimal_trajectory, info = optimizer.optimize(initial_trajectory, goal)
```

---

## Advanced Features

### Feature 1: AI Terrain Prediction
Neural network predicts terrain traversal costs from local features.

```python
from terrain_predictor import TerrainPredictor, generate_terrain_training_data

# Train predictor
predictor = TerrainPredictor()
X_train, y_train = generate_terrain_training_data(environment, num_samples=2000)
predictor.train(X_train, y_train, epochs=100)

# Use for prediction
terrain_cost = predictor.predict(features)
```

### Feature 2: Energy Recovery Zones
Special regions where robots can recharge batteries.
- Influences trajectory selection
- Trade-off: longer path vs. energy gain

### Feature 3: Multi-Agent Coordination
Multiple robots optimize simultaneously with collision avoidance.

```python
from multi_agent import MultiAgentSystem

multi_agent = MultiAgentSystem(environment, num_agents=3)
multi_agent.initialize_agents(start_positions, goal_positions)
trajectories, info = multi_agent.optimize_all_trajectories(cost_fns, sensors)
```

### Feature 4: Disturbance & Noise Modeling
Tests robustness:
- Wind fields (spatial varying)
- Random external forces
- Sensor measurement noise
- Dynamic obstacle uncertainty

---

## Usage Examples

### Example 1: Basic Navigation

```python
from realtime_system import RealtimeNavigationSystem
import numpy as np

start = np.array([10, 10])
goal = np.array([85, 85])

system = RealtimeNavigationSystem(start, goal)
system.run_simulation(max_time=40.0, verbose=True)
system.visualize_results(save_path='results.png')
```

### Example 2: Energy-Constrained Mission

```python
system = RealtimeNavigationSystem(start, goal)
system.energy.reset(battery_level=60.0)  # Low battery
system.run_simulation(max_time=60.0)
```

### Example 3: Multi-Agent Scenario

```python
from multi_agent import MultiAgentSystem

env = SimulationEnvironment()
multi_agent = MultiAgentSystem(env, num_agents=3)

starts = [np.array([10, 10]), np.array([10, 90]), np.array([90, 50])]
goals = [np.array([90, 90]), np.array([90, 10]), np.array([10, 50])]

multi_agent.initialize_agents(starts, goals)
# ... optimize and execute
```

---

## Demonstration Scenarios

Run the comprehensive demos:

```bash
# Demo 1: Basic navigation with DR optimization
python demo.py 1

# Demo 2: AI-enhanced terrain prediction
python demo.py 2

# Demo 3: Energy-constrained navigation
python demo.py 3

# Demo 4: Multi-agent coordination
python demo.py 4

# Demo 5: Robustness under disturbances
python demo.py 5

# Demo 6: Comparative analysis
python demo.py 6

# Run all demos
python demo.py all
```

Each demo generates visualization plots saved as PNG files.

---

## Configuration

Modify `config.py` to customize:

```python
# Environment
WORKSPACE_SIZE = (100, 100)  # meters
NUM_STATIC_OBSTACLES = 8
NUM_DYNAMIC_OBSTACLES = 3

# Robot
MAX_VELOCITY = 5.0  # m/s
MAX_ACCELERATION = 2.0  # m/s^2

# Douglas-Rachford
DR_MAX_ITERATIONS = 100
DR_TOLERANCE = 1e-4
DR_GAMMA = 0.5

# Energy
INITIAL_BATTERY = 100.0  # percent
RECOVERY_RATE = 2.0  # %/second

# Cost weights
WEIGHT_LENGTH = 1.0
WEIGHT_ENERGY = 2.0
WEIGHT_SMOOTHNESS = 1.5
```

---

## Performance Metrics

The system tracks:
- **Energy consumption** - Total battery used
- **Path optimality** - Length compared to straight-line
- **Convergence rate** - DR algorithm iterations
- **Computation time** - Real-time performance
- **Collision avoidance** - Success rate
- **Robustness** - Performance under noise

---

## Research Contributions

This project's novelty lies in combining:
1. Douglas–Rachford splitting for real-time trajectory optimization
2. Energy-aware path planning with recovery zones
3. AI-based terrain cost prediction
4. Multi-agent coordinated navigation
5. Robustness to disturbances and sensor noise

In a **unified framework** suitable for publication.

---

## Experimental Scenarios

Test various conditions:
- Static vs dynamic obstacles
- High vs low battery
- With vs without recovery zones
- Single vs multi-agent
- No disturbance vs high disturbance
- With vs without AI terrain prediction

---

## Mathematical Formulation

**Optimization Problem:**
```
minimize   f(X) + g(X)
  X

where:
  X = [x₁, x₂, ..., xₙ] ∈ ℝⁿˣ²     (trajectory)
  
  f(X) = w₁·L(X) + w₂·E(X) + w₃·S(X)  (cost function)
    L(X) = path length
    E(X) = energy consumption
    S(X) = smoothness (curvature)
  
  g(X) = indicator function for constraints:
    - Collision-free
    - Kinematic limits
    - Energy feasibility
    - Workspace bounds
```

**Douglas–Rachford Splitting:**
```
Proximal operators:
  prox_g(z) = argmin_x { ½‖x - z‖² : x satisfies constraints }
  prox_f(z) = argmin_x { f(x) + ½‖x - z‖² }

Iteration:
  xₖ₊₁ = prox_g(zₖ)
  yₖ₊₁ = prox_f(2xₖ₊₁ - zₖ)
  zₖ₊₁ = zₖ + λ(yₖ₊₁ - xₖ₊₁)

Convergence: ‖zₖ - zₖ₋₁‖ < ε
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{douglas_rachford_trajectory_optimization,
  title = {Real-Time Trajectory Optimization using Douglas–Rachford Splitting},
  author = {Your Name},
  year = {2025},
  description = {Multi-agent trajectory planning with energy constraints and disturbances}
}
```

---

## License

This project is provided as-is for research and educational purposes.

---

## Contact

For questions or collaborations, please open an issue on the repository.

---

## Acknowledgments

- Douglas–Rachford splitting algorithm
- Python scientific computing ecosystem (NumPy, Matplotlib)
- Robotics and optimization research community

---

**Ready for Publication!** 🚀

This implementation provides:
✅ Complete working code
✅ Multiple demonstration scenarios
✅ Performance metrics and analysis
✅ Comprehensive documentation
✅ Research-quality results

Perfect for submission to robotics conferences (ICRA, IROS, etc.) or journals!
