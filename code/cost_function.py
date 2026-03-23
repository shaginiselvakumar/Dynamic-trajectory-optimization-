"""
Module 5: Cost Function Modeling
Defines trajectory optimality criteria
"""

import numpy as np
from config import Config

class CostFunction:
    """Multi-objective cost function for trajectory optimization"""
    
    def __init__(self, environment, energy_model):
        self.config = Config()
        self.environment = environment
        self.energy_model = energy_model
        
        # Weights
        self.w_length = self.config.WEIGHT_LENGTH
        self.w_energy = self.config.WEIGHT_ENERGY
        self.w_smoothness = self.config.WEIGHT_SMOOTHNESS
        self.w_obstacle = self.config.WEIGHT_OBSTACLE
        self.w_goal = self.config.WEIGHT_GOAL
    
    def compute_total_cost(self, trajectory, goal_position, 
                          velocity_profile=None, return_components=False):
        """
        Compute total trajectory cost
        
        Args:
            trajectory: Array of positions [N x 2]
            goal_position: Goal position [x, y]
            velocity_profile: Optional velocity sequence [N x 2]
            return_components: If True, return cost breakdown
            
        Returns:
            total_cost: Scalar cost value
            components: (optional) Dict of individual cost components
        """
        N = len(trajectory)
        
        # 1. Path length cost
        length_cost = self._compute_path_length(trajectory)
        
        # 2. Energy cost
        if velocity_profile is not None:
            energy_cost = self._compute_energy_cost(trajectory, velocity_profile)
        else:
            energy_cost = 0.0
        
        # 3. Smoothness cost (acceleration/jerk)
        smoothness_cost = self._compute_smoothness_cost(trajectory)
        
        # 4. Obstacle proximity cost
        obstacle_cost = self._compute_obstacle_cost(trajectory)
        
        # 5. Goal proximity cost
        goal_cost = self._compute_goal_cost(trajectory[-1], goal_position)
        
        # Total weighted cost
        total_cost = (self.w_length * length_cost + 
                     self.w_energy * energy_cost +
                     self.w_smoothness * smoothness_cost +
                     self.w_obstacle * obstacle_cost +
                     self.w_goal * goal_cost)
        
        if return_components:
            components = {
                'length': length_cost,
                'energy': energy_cost,
                'smoothness': smoothness_cost,
                'obstacle': obstacle_cost,
                'goal': goal_cost,
                'total': total_cost
            }
            return total_cost, components
        
        return total_cost
    
    def _compute_path_length(self, trajectory):
        """Compute total path length"""
        if len(trajectory) < 2:
            return 0.0
        
        diffs = np.diff(trajectory, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)
    
    def _compute_energy_cost(self, trajectory, velocity_profile):
        """Compute energy consumption cost"""
        N = len(trajectory)
        dt = self.config.DT
        
        # Get terrain costs for each position
        terrain_costs = np.array([self.environment.get_terrain_cost(pos) 
                                 for pos in trajectory])
        
        # Predict energy consumption
        total_consumption, _ = self.energy_model.predict_energy_consumption(
            velocity_profile, dt, terrain_costs)
        
        # Normalize to 0-1 range (assume max consumption is 100%)
        normalized_cost = total_consumption / 100.0
        
        return normalized_cost
    
    def _compute_smoothness_cost(self, trajectory):
        """
        Compute trajectory smoothness (penalize sharp turns)
        Based on sum of squared accelerations
        """
        if len(trajectory) < 3:
            return 0.0
        
        # Second-order differences (approximate acceleration)
        velocities = np.diff(trajectory, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Sum of squared acceleration magnitudes
        accel_magnitudes = np.linalg.norm(accelerations, axis=1)
        smoothness_cost = np.sum(accel_magnitudes ** 2)
        
        # Normalize
        normalized_cost = smoothness_cost / len(trajectory)
        
        return normalized_cost
    
    def _compute_obstacle_cost(self, trajectory):
        """
        Compute cost based on proximity to obstacles
        Uses soft constraints (exponential penalty)
        """
        total_cost = 0.0
        safety_margin = self.config.OBSTACLE_SAFETY_MARGIN
        
        for position in trajectory:
            min_distance = float('inf')
            
            # Check static obstacles
            for obs in self.environment.static_obstacles:
                if obs['type'] == 'circle':
                    dist = np.linalg.norm(position - obs['center']) - obs['radius']
                    min_distance = min(min_distance, dist)
                
                elif obs['type'] == 'rectangle':
                    # Distance to rectangle (simplified)
                    corner = obs['corner']
                    center = corner + np.array([obs['width'], obs['height']]) / 2
                    
                    # Approximate as distance to center minus half-diagonal
                    half_diag = np.sqrt(obs['width']**2 + obs['height']**2) / 2
                    dist = np.linalg.norm(position - center) - half_diag
                    min_distance = min(min_distance, dist)
            
            # Check dynamic obstacles
            for obs in self.environment.dynamic_obstacles:
                dist = np.linalg.norm(position - obs['position']) - obs['radius']
                min_distance = min(min_distance, dist)
            
            # Exponential penalty if too close
            if min_distance < safety_margin:
                penalty = np.exp(-min_distance / safety_margin)
                total_cost += penalty
        
        return total_cost
    
    def _compute_goal_cost(self, final_position, goal_position):
        """Compute cost based on distance to goal"""
        distance = np.linalg.norm(final_position - goal_position)
        return distance ** 2  # Quadratic penalty
    
    def compute_gradient(self, trajectory, goal_position, velocity_profile=None):
        """
        Compute gradient of cost function w.r.t trajectory
        Uses finite differences
        
        Args:
            trajectory: Current trajectory [N x 2]
            goal_position: Goal position
            velocity_profile: Optional velocities
            
        Returns:
            gradient: Gradient array [N x 2]
        """
        N = len(trajectory)
        gradient = np.zeros_like(trajectory)
        epsilon = 1e-5
        
        # Compute gradient for each point
        for i in range(N):
            for dim in range(2):
                # Perturb in positive direction
                traj_plus = trajectory.copy()
                traj_plus[i, dim] += epsilon
                cost_plus = self.compute_total_cost(traj_plus, goal_position, 
                                                   velocity_profile)
                
                # Perturb in negative direction
                traj_minus = trajectory.copy()
                traj_minus[i, dim] -= epsilon
                cost_minus = self.compute_total_cost(traj_minus, goal_position, 
                                                    velocity_profile)
                
                # Central difference
                gradient[i, dim] = (cost_plus - cost_minus) / (2 * epsilon)
        
        return gradient
    
    def is_collision_free(self, trajectory):
        """Check if entire trajectory is collision-free"""
        for position in trajectory:
            if self.environment.check_collision(position):
                return False
        return True
    
    def compute_terrain_aware_cost(self, trajectory):
        """Compute cost considering terrain difficulty"""
        total_cost = 0.0
        
        for i in range(len(trajectory) - 1):
            segment_length = np.linalg.norm(trajectory[i+1] - trajectory[i])
            terrain_cost = self.environment.get_terrain_cost(trajectory[i])
            total_cost += segment_length * terrain_cost
        
        return total_cost
    
    def set_weights(self, w_length=None, w_energy=None, w_smoothness=None,
                   w_obstacle=None, w_goal=None):
        """Update cost function weights"""
        if w_length is not None:
            self.w_length = w_length
        if w_energy is not None:
            self.w_energy = w_energy
        if w_smoothness is not None:
            self.w_smoothness = w_smoothness
        if w_obstacle is not None:
            self.w_obstacle = w_obstacle
        if w_goal is not None:
            self.w_goal = w_goal


class MultiAgentCostFunction(CostFunction):
    """Extended cost function for multi-agent scenarios"""
    
    def __init__(self, environment, energy_model, num_agents):
        super().__init__(environment, energy_model)
        self.num_agents = num_agents
        self.w_separation = 50.0  # Weight for inter-agent separation
    
    def compute_multi_agent_cost(self, trajectories, goal_positions):
        """
        Compute cost for multiple agents
        
        Args:
            trajectories: List of trajectories (one per agent)
            goal_positions: List of goal positions
            
        Returns:
            total_cost: Combined cost
            individual_costs: List of costs per agent
        """
        individual_costs = []
        
        # Individual costs
        for i, (traj, goal) in enumerate(zip(trajectories, goal_positions)):
            cost = self.compute_total_cost(traj, goal)
            individual_costs.append(cost)
        
        # Inter-agent collision avoidance cost
        separation_cost = self._compute_separation_cost(trajectories)
        
        # Total cost
        total_cost = sum(individual_costs) + self.w_separation * separation_cost
        
        return total_cost, individual_costs
    
    def _compute_separation_cost(self, trajectories):
        """Penalize agents getting too close"""
        min_distance = self.config.INTER_AGENT_DISTANCE
        total_cost = 0.0
        
        N = len(trajectories[0])  # trajectory length
        
        for t in range(N):
            # Check all pairs of agents
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    pos_i = trajectories[i][t]
                    pos_j = trajectories[j][t]
                    
                    distance = np.linalg.norm(pos_i - pos_j)
                    
                    # Penalty if too close
                    if distance < min_distance:
                        penalty = (min_distance - distance) ** 2
                        total_cost += penalty
        
        return total_cost


if __name__ == '__main__':
    # Test cost function
    from environment import SimulationEnvironment
    from energy_model import EnergyModel
    import matplotlib.pyplot as plt
    
    env = SimulationEnvironment()
    energy = EnergyModel()
    cost_fn = CostFunction(env, energy)
    
    # Create test trajectory
    N = 30
    t = np.linspace(0, 2*np.pi, N)
    trajectory = np.column_stack([
        20 + 15 * np.cos(t),
        20 + 15 * np.sin(t)
    ])
    
    goal = np.array([80, 80])
    
    # Compute costs
    total_cost, components = cost_fn.compute_total_cost(
        trajectory, goal, return_components=True)
    
    print("Cost Components:")
    for key, value in components.items():
        print(f"  {key}: {value:.4f}")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(10, 10))
    env.visualize(ax=ax)
    
    # Plot trajectory with cost-based coloring
    colors = plt.cm.viridis(np.linspace(0, 1, N))
    for i in range(N-1):
        ax.plot(trajectory[i:i+2, 0], trajectory[i:i+2, 1], 
               color=colors[i], linewidth=3)
    
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', 
           markersize=15, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    
    ax.set_title(f'Test Trajectory (Total Cost: {total_cost:.2f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('/home/cost_test.png', dpi=150, bbox_inches='tight')
    print("Cost function test complete!")
