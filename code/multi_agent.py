"""
Advanced Feature 3: Multi-Agent Coordination
Multiple robots optimize simultaneously with collision avoidance
"""

import numpy as np
from config import Config
from motion_engine import RobotMotionEngine
from energy_model import EnergyModel
from douglas_rachford import DouglasRachfordOptimizer

class MultiAgentSystem:
    """
    Coordinated multi-agent trajectory optimization
    with inter-agent collision avoidance
    """
    
    def __init__(self, environment, num_agents=None):
        self.config = Config()
        self.environment = environment
        self.num_agents = num_agents if num_agents else self.config.NUM_AGENTS
        
        # Create agents
        self.agents = []
        self.agent_colors = plt.cm.Set3(np.linspace(0, 1, self.num_agents))
        
    def initialize_agents(self, start_positions, goal_positions, 
                         initial_velocities=None):
        """
        Initialize all agents
        
        Args:
            start_positions: List of start positions
            goal_positions: List of goal positions
            initial_velocities: Optional list of initial velocities
        """
        self.agents = []
        
        for i in range(self.num_agents):
            init_vel = (initial_velocities[i] if initial_velocities 
                       else np.zeros(2))
            
            agent = {
                'id': i,
                'robot': RobotMotionEngine(start_positions[i], init_vel),
                'energy': EnergyModel(),
                'goal': goal_positions[i],
                'trajectory': [],
                'planned_trajectory': None,
                'color': self.agent_colors[i]
            }
            
            self.agents.append(agent)
    
    def optimize_all_trajectories(self, cost_fn_list, sensor_list, verbose=False):
        """
        Jointly optimize trajectories for all agents
        Uses alternating optimization with collision constraints
        
        Args:
            cost_fn_list: List of cost functions (one per agent)
            sensor_list: List of sensor objects
            verbose: Print progress
            
        Returns:
            all_trajectories: List of optimized trajectories
            convergence_info: Optimization statistics
        """
        # Initialize trajectories
        all_trajectories = []
        for agent in self.agents:
            start = agent['robot'].position
            goal = agent['goal']
            N = self.config.HORIZON_STEPS
            
            # Straight line initialization
            init_traj = np.linspace(start, goal, N)
            all_trajectories.append(init_traj)
        
        # Alternating optimization
        max_outer_iterations = 10
        convergence_history = []
        
        for outer_iter in range(max_outer_iterations):
            if verbose:
                print(f"\nOuter iteration {outer_iter + 1}/{max_outer_iterations}")
            
            total_cost_change = 0.0
            
            # Optimize each agent's trajectory
            for i, agent in enumerate(self.agents):
                # Get other agents' trajectories
                other_trajectories = [all_trajectories[j] 
                                     for j in range(self.num_agents) if j != i]
                
                # Create optimizer with inter-agent constraints
                optimizer = MultiAgentDROptimizer(
                    cost_fn_list[i],
                    self.environment,
                    agent['energy'],
                    other_trajectories,
                    agent_id=i
                )
                
                # Optimize
                optimal_traj, info = optimizer.optimize(
                    all_trajectories[i],
                    agent['goal'],
                    agent['robot'].velocity,
                    verbose=False
                )
                
                # Update
                cost_change = abs(info['final_cost'] - 
                                cost_fn_list[i].compute_total_cost(
                                    all_trajectories[i], agent['goal']))
                total_cost_change += cost_change
                
                all_trajectories[i] = optimal_traj
                
                if verbose:
                    print(f"  Agent {i}: cost change = {cost_change:.4f}")
            
            convergence_history.append(total_cost_change)
            
            # Check convergence
            if total_cost_change < 0.01:
                if verbose:
                    print(f"Multi-agent optimization converged at iteration {outer_iter + 1}")
                break
        
        # Update agent planned trajectories
        for i, agent in enumerate(self.agents):
            agent['planned_trajectory'] = all_trajectories[i]
        
        convergence_info = {
            'outer_iterations': outer_iter + 1,
            'convergence_history': convergence_history,
            'final_cost_change': total_cost_change
        }
        
        return all_trajectories, convergence_info
    
    def execute_step(self, dt):
        """
        Execute one time step for all agents
        
        Args:
            dt: Time step
        """
        for agent in self.agents:
            # Get control from planned trajectory
            if agent['planned_trajectory'] is not None and len(agent['planned_trajectory']) > 1:
                # Desired position (next waypoint)
                target_pos = agent['planned_trajectory'][1]
                
                # Compute desired velocity
                desired_vel = (target_pos - agent['robot'].position) / dt
                
                # Get external forces (wind)
                wind_force = self.environment.get_wind_force(agent['robot'].position)
                
                # Apply control
                agent['robot'].apply_control(desired_vel, dt, wind_force)
                
                # Update energy
                in_recovery, rate = self.environment.in_recovery_zone(
                    agent['robot'].position)
                terrain_cost = self.environment.get_terrain_cost(
                    agent['robot'].position)
                
                agent['energy'].update(
                    agent['robot'].velocity,
                    agent['robot'].acceleration,
                    dt,
                    in_recovery,
                    rate,
                    terrain_cost
                )
                
                # Store trajectory point
                agent['trajectory'].append(agent['robot'].position.copy())
                
                # Shift planned trajectory
                agent['planned_trajectory'] = agent['planned_trajectory'][1:]
    
    def check_inter_agent_collisions(self):
        """Check for collisions between agents"""
        min_distance = self.config.INTER_AGENT_DISTANCE
        collisions = []
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pos_i = self.agents[i]['robot'].position
                pos_j = self.agents[j]['robot'].position
                
                distance = np.linalg.norm(pos_i - pos_j)
                
                if distance < min_distance:
                    collisions.append((i, j, distance))
        
        return collisions
    
    def get_statistics(self):
        """Get statistics for all agents"""
        stats = []
        
        for agent in self.agents:
            agent_stats = {
                'id': agent['id'],
                'position': agent['robot'].position.copy(),
                'velocity': agent['robot'].velocity.copy(),
                'battery': agent['energy'].battery_level,
                'path_length': agent['robot'].compute_path_length(),
                'goal_distance': np.linalg.norm(
                    agent['robot'].position - agent['goal']
                ),
                'energy_stats': agent['energy'].get_statistics()
            }
            stats.append(agent_stats)
        
        return stats


class MultiAgentDROptimizer(DouglasRachfordOptimizer):
    """
    Extended Douglas-Rachford optimizer for multi-agent scenarios
    Adds inter-agent collision avoidance constraints
    """
    
    def __init__(self, cost_function, environment, energy_model, 
                 other_trajectories, agent_id=0):
        super().__init__(cost_function, environment, energy_model)
        self.other_trajectories = other_trajectories
        self.agent_id = agent_id
    
    def _prox_constraints(self, trajectory, goal_position):
        """
        Extended constraint projection with inter-agent avoidance
        """
        # First apply standard constraints
        projected = super()._prox_constraints(trajectory, goal_position)
        
        # Then enforce inter-agent separation
        min_distance = self.config.INTER_AGENT_DISTANCE
        
        # Project away from other agents
        for other_traj in self.other_trajectories:
            if len(other_traj) != len(projected):
                continue
            
            for i in range(len(projected)):
                # Distance to other agent at same time step
                vec = projected[i] - other_traj[i]
                dist = np.linalg.norm(vec)
                
                if dist < min_distance:
                    # Push away
                    push_dist = min_distance - dist
                    if dist > 1e-6:
                        direction = vec / dist
                    else:
                        # Random direction if exactly overlapping
                        direction = np.random.randn(2)
                        direction /= np.linalg.norm(direction)
                    
                    projected[i] = projected[i] + direction * push_dist * 0.5
        
        return projected


# Import matplotlib here to avoid circular dependency
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Test multi-agent system
    from environment import SimulationEnvironment
    from cost_function import CostFunction
    from sensor_perception import SensorPerception
    
    # Setup
    env = SimulationEnvironment()
    multi_agent = MultiAgentSystem(env, num_agents=3)
    
    # Define starts and goals
    starts = [
        np.array([10, 10]),
        np.array([10, 90]),
        np.array([90, 10])
    ]
    
    goals = [
        np.array([90, 90]),
        np.array([90, 10]),
        np.array([10, 90])
    ]
    
    # Initialize
    multi_agent.initialize_agents(starts, goals)
    
    # Create cost functions and sensors
    cost_fns = []
    sensors = []
    for agent in multi_agent.agents:
        cost_fn = CostFunction(env, agent['energy'])
        sensor = SensorPerception(env)
        cost_fns.append(cost_fn)
        sensors.append(sensor)
    
    # Optimize
    print("Optimizing multi-agent trajectories...")
    trajectories, info = multi_agent.optimize_all_trajectories(
        cost_fns, sensors, verbose=True)
    
    print(f"\nMulti-agent optimization complete!")
    print(f"Outer iterations: {info['outer_iterations']}")
    print(f"Final cost change: {info['final_cost_change']:.6f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(12, 12))
    env.visualize(ax=ax)
    
    # Plot trajectories
    for i, (agent, traj) in enumerate(zip(multi_agent.agents, trajectories)):
        ax.plot(traj[:, 0], traj[:, 1], '-', color=agent['color'], 
               linewidth=3, label=f'Agent {i}', alpha=0.8)
        ax.plot(starts[i][0], starts[i][1], 'o', color=agent['color'], 
               markersize=12)
        ax.plot(goals[i][0], goals[i][1], '*', color=agent['color'], 
               markersize=15)
    
    ax.legend()
    ax.set_title('Multi-Agent Coordinated Trajectories')
    
    plt.tight_layout()
    plt.savefig('/home/claude/multi_agent_test.png', dpi=150, bbox_inches='tight')
    print("Multi-agent test complete!")
