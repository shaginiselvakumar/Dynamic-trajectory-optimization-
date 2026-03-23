"""
Module 7: Douglas-Rachford Optimization Solver
CORE CONTRIBUTION: Real-time trajectory optimization using DR splitting

Solves: min f(X) + g(X)
where:
  f(X) = cost function (smooth)
  g(X) = constraints (non-smooth: collision avoidance, bounds, energy)

Douglas-Rachford Algorithm:
  1. z_{k+1} = prox_{γg}(z_k)
  2. y_{k+1} = prox_{γf}(2z_{k+1} - z_k)  
  3. z_{k+1} = z_k + λ(y_{k+1} - z_{k+1})
"""

import numpy as np
from config import Config
import time

class DouglasRachfordOptimizer:
    """
    Douglas-Rachford splitting for trajectory optimization
    with energy, obstacle, and smoothness constraints
    """
    
    def __init__(self, cost_function, environment, energy_model):
        self.config = Config()
        self.cost_fn = cost_function
        self.environment = environment
        self.energy_model = energy_model
        
        # DR parameters
        self.gamma = self.config.DR_GAMMA
        self.lambda_param = self.config.DR_LAMBDA
        self.max_iterations = self.config.DR_MAX_ITERATIONS
        self.tolerance = self.config.DR_TOLERANCE
        
        # Statistics
        self.iteration_count = 0
        self.convergence_history = []
        self.cost_history = []
        self.computation_time = 0.0
    
    def optimize(self, initial_trajectory, goal_position, current_velocity=None,
                verbose=False):
        """
        Optimize trajectory using Douglas-Rachford splitting
        
        Args:
            initial_trajectory: Initial guess [N x 2]
            goal_position: Goal position [x, y]
            current_velocity: Current robot velocity [vx, vy]
            verbose: Print optimization progress
            
        Returns:
            optimal_trajectory: Optimized trajectory [N x 2]
            convergence_info: Dict with optimization statistics
        """
        start_time = time.time()
        
        N = len(initial_trajectory)
        
        # Initialize
        z = initial_trajectory.copy()  # Dual variable
        residuals = []
        costs = []
        
        if verbose:
            print(f"Starting Douglas-Rachford optimization for {N}-step trajectory")
        
        for iteration in range(self.max_iterations):
            # Store previous iterate
            z_prev = z.copy()
            
            # Step 1: Proximal operator for constraints g(X)
            # This projects onto feasible set
            x = self._prox_constraints(z, goal_position)
            
            # Step 2: Proximal operator for cost function f(X)
            # This minimizes the cost
            y = self._prox_cost(2 * x - z, goal_position, current_velocity)
            
            # Step 3: Update dual variable
            z = z + self.lambda_param * (y - x)
            
            # Compute residual (convergence criterion)
            residual = np.linalg.norm(z - z_prev) / (np.linalg.norm(z_prev) + 1e-10)
            residuals.append(residual)
            
            # Compute cost
            cost = self.cost_fn.compute_total_cost(x, goal_position)
            costs.append(cost)
            
            if verbose and iteration % 10 == 0:
                print(f"  Iter {iteration}: residual={residual:.6f}, cost={cost:.4f}")
            
            # Check convergence
            if residual < self.tolerance:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
        
        self.computation_time = time.time() - start_time
        self.iteration_count = iteration + 1
        self.convergence_history = residuals
        self.cost_history = costs
        
        # Final trajectory (primal variable)
        optimal_trajectory = x
        
        # Convergence info
        convergence_info = {
            'iterations': self.iteration_count,
            'final_residual': residuals[-1] if residuals else 0.0,
            'final_cost': costs[-1] if costs else 0.0,
            'computation_time': self.computation_time,
            'converged': residual < self.tolerance if residual else False,
            'residual_history': residuals,
            'cost_history': costs
        }
        
        return optimal_trajectory, convergence_info
    
    def _prox_constraints(self, trajectory, goal_position):
        """
        Proximal operator for constraints g(X)
        Projects trajectory onto feasible set:
          - Collision-free
          - Within workspace bounds
          - Kinematic constraints
          - Energy feasible
        """
        N = len(trajectory)
        projected = trajectory.copy()
        
        # 1. Workspace bounds
        projected[:, 0] = np.clip(projected[:, 0], 
                                 self.config.ROBOT_RADIUS,
                                 self.environment.width - self.config.ROBOT_RADIUS)
        projected[:, 1] = np.clip(projected[:, 1], 
                                 self.config.ROBOT_RADIUS,
                                 self.environment.height - self.config.ROBOT_RADIUS)
        
        # 2. Obstacle avoidance (iterative projection)
        for iteration in range(3):  # Few iterations of projection
            for i in range(N):
                projected[i] = self._project_away_from_obstacles(projected[i])
        
        # 3. Kinematic feasibility (velocity and acceleration limits)
        dt = self.config.HORIZON_TIME / N
        projected = self._enforce_kinematic_limits(projected, dt)
        
        # 4. Energy constraint (soft enforcement - reduce velocity if needed)
        projected = self._enforce_energy_feasibility(projected, dt)
        
        return projected
    
    def _project_away_from_obstacles(self, position):
        """Project a single point away from obstacles"""
        safety_margin = self.config.OBSTACLE_SAFETY_MARGIN
        
        # Check all obstacles
        min_penetration = 0.0
        push_direction = np.zeros(2)
        
        # Static obstacles
        for obs in self.environment.static_obstacles:
            if obs['type'] == 'circle':
                vec = position - obs['center']
                dist = np.linalg.norm(vec)
                required_dist = obs['radius'] + safety_margin
                
                if dist < required_dist:
                    penetration = required_dist - dist
                    if penetration > min_penetration:
                        min_penetration = penetration
                        push_direction = vec / (dist + 1e-10)
            
            elif obs['type'] == 'rectangle':
                # Simplified: push away from rectangle center
                corner = obs['corner']
                center = corner + np.array([obs['width'], obs['height']]) / 2
                
                # Check if inside or close
                if (corner[0] - safety_margin <= position[0] <= corner[0] + obs['width'] + safety_margin and
                    corner[1] - safety_margin <= position[1] <= corner[1] + obs['height'] + safety_margin):
                    
                    vec = position - center
                    dist = np.linalg.norm(vec)
                    if dist > 0:
                        push_direction = vec / dist
                        min_penetration = safety_margin
        
        # Dynamic obstacles
        for obs in self.environment.dynamic_obstacles:
            vec = position - obs['position']
            dist = np.linalg.norm(vec)
            required_dist = obs['radius'] + safety_margin
            
            if dist < required_dist:
                penetration = required_dist - dist
                if penetration > min_penetration:
                    min_penetration = penetration
                    push_direction = vec / (dist + 1e-10)
        
        # Push away if needed
        if min_penetration > 0:
            position = position + push_direction * min_penetration
        
        return position
    
    def _enforce_kinematic_limits(self, trajectory, dt):
        """Enforce velocity and acceleration limits"""
        N = len(trajectory)
        adjusted = trajectory.copy()
        
        for i in range(1, N):
            # Velocity constraint
            displacement = adjusted[i] - adjusted[i-1]
            velocity = displacement / dt
            vel_mag = np.linalg.norm(velocity)
            
            if vel_mag > self.config.MAX_VELOCITY:
                # Scale down displacement
                scale = self.config.MAX_VELOCITY / vel_mag
                adjusted[i] = adjusted[i-1] + displacement * scale
        
        # Acceleration constraint (second pass)
        for i in range(2, N):
            vel_prev = (adjusted[i-1] - adjusted[i-2]) / dt
            vel_curr = (adjusted[i] - adjusted[i-1]) / dt
            acceleration = (vel_curr - vel_prev) / dt
            accel_mag = np.linalg.norm(acceleration)
            
            if accel_mag > self.config.MAX_ACCELERATION:
                # Limit acceleration
                scale = self.config.MAX_ACCELERATION / accel_mag
                vel_curr_limited = vel_prev + (vel_curr - vel_prev) * scale
                adjusted[i] = adjusted[i-1] + vel_curr_limited * dt
        
        return adjusted
    
    def _enforce_energy_feasibility(self, trajectory, dt):
        """Ensure trajectory doesn't deplete energy"""
        N = len(trajectory)
        
        # Compute velocities
        velocities = np.diff(trajectory, axis=0) / dt
        
        # Predict energy consumption
        terrain_costs = np.array([self.environment.get_terrain_cost(pos) 
                                 for pos in trajectory[:-1]])
        
        total_consumption, battery_traj = self.energy_model.predict_energy_consumption(
            velocities, dt, terrain_costs)
        
        # If energy insufficient, scale down velocities
        if battery_traj[-1] < self.config.MIN_BATTERY_THRESHOLD:
            # Reduce speed to make it feasible
            scale_factor = 0.8
            adjusted = trajectory.copy()
            
            for i in range(1, N):
                displacement = trajectory[i] - trajectory[i-1]
                adjusted[i] = adjusted[i-1] + displacement * scale_factor
            
            return adjusted
        
        return trajectory
    
    def _prox_cost(self, trajectory, goal_position, current_velocity):
        """
        Proximal operator for cost function f(X)
        Minimizes: f(X) + (1/2γ)||X - trajectory||^2
        
        Using gradient descent for smooth cost
        """
        N = len(trajectory)
        x = trajectory.copy()
        
        # Gradient descent steps
        step_size = 0.1
        num_steps = 5
        
        for _ in range(num_steps):
            # Compute gradient of cost function
            grad = self._compute_cost_gradient(x, goal_position)
            
            # Proximal gradient step
            x_new = x - step_size * (grad + (x - trajectory) / self.gamma)
            
            # Line search (simple backtracking)
            cost_new = self.cost_fn.compute_total_cost(x_new, goal_position)
            cost_old = self.cost_fn.compute_total_cost(x, goal_position)
            
            if cost_new < cost_old:
                x = x_new
            else:
                step_size *= 0.5
        
        return x
    
    def _compute_cost_gradient(self, trajectory, goal_position):
        """Compute gradient of cost function"""
        N = len(trajectory)
        gradient = np.zeros_like(trajectory)
        epsilon = 1e-4
        
        # Finite differences (parallelizable in practice)
        for i in range(N):
            for dim in range(2):
                traj_plus = trajectory.copy()
                traj_plus[i, dim] += epsilon
                cost_plus = self.cost_fn.compute_total_cost(traj_plus, goal_position)
                
                traj_minus = trajectory.copy()
                traj_minus[i, dim] -= epsilon
                cost_minus = self.cost_fn.compute_total_cost(traj_minus, goal_position)
                
                gradient[i, dim] = (cost_plus - cost_minus) / (2 * epsilon)
        
        return gradient
    
    def get_statistics(self):
        """Get optimization statistics"""
        return {
            'iterations': self.iteration_count,
            'computation_time': self.computation_time,
            'final_residual': self.convergence_history[-1] if self.convergence_history else 0.0,
            'final_cost': self.cost_history[-1] if self.cost_history else 0.0,
            'convergence_rate': self._compute_convergence_rate()
        }
    
    def _compute_convergence_rate(self):
        """Estimate convergence rate (linear, sublinear, etc.)"""
        if len(self.convergence_history) < 10:
            return 'N/A'
        
        # Fit exponential to residuals
        residuals = np.array(self.convergence_history[-10:])
        if np.all(residuals > 0):
            log_residuals = np.log(residuals)
            # Linear fit
            x = np.arange(len(log_residuals))
            slope = np.polyfit(x, log_residuals, 1)[0]
            
            if slope < -0.1:
                return 'linear'
            elif slope < -0.01:
                return 'sublinear'
            else:
                return 'slow'
        
        return 'unknown'


if __name__ == '__main__':
    # Test Douglas-Rachford optimizer
    from environment import SimulationEnvironment
    from energy_model import EnergyModel
    from cost_function import CostFunction
    import matplotlib.pyplot as plt
    
    # Setup
    env = SimulationEnvironment()
    energy = EnergyModel()
    cost_fn = CostFunction(env, energy)
    optimizer = DouglasRachfordOptimizer(cost_fn, env, energy)
    
    # Initial trajectory (straight line)
    start = np.array([10, 10])
    goal = np.array([85, 85])
    N = 20
    
    initial_traj = np.linspace(start, goal, N)
    
    # Optimize
    print("Running Douglas-Rachford optimization...")
    optimal_traj, info = optimizer.optimize(initial_traj, goal, verbose=True)
    
    print("\nOptimization Results:")
    for key, value in info.items():
        if key not in ['residual_history', 'cost_history']:
            print(f"  {key}: {value}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    
    # 1. Trajectories
    ax = axes[0, 0]
    env.visualize(ax=ax)
    ax.plot(initial_traj[:, 0], initial_traj[:, 1], 'r--', 
           linewidth=2, label='Initial', alpha=0.7)
    ax.plot(optimal_traj[:, 0], optimal_traj[:, 1], 'b-', 
           linewidth=3, label='Optimized')
    ax.plot(start[0], start[1], 'go', markersize=15, label='Start')
    ax.plot(goal[0], goal[1], 'r*', markersize=20, label='Goal')
    ax.legend()
    ax.set_title('Trajectory Optimization Result')
    
    # 2. Convergence
    ax = axes[0, 1]
    ax.semilogy(info['residual_history'], 'b-', linewidth=2)
    ax.axhline(y=optimizer.tolerance, color='r', linestyle='--', 
              label='Tolerance')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Residual')
    ax.set_title('Convergence History')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 3. Cost evolution
    ax = axes[1, 0]
    ax.plot(info['cost_history'], 'g-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    ax.set_title('Cost Function Evolution')
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""
    Optimization Statistics:
    
    Iterations: {info['iterations']}
    Computation Time: {info['computation_time']:.4f} s
    Final Residual: {info['final_residual']:.6f}
    Final Cost: {info['final_cost']:.4f}
    Converged: {info['converged']}
    
    Path Length:
      Initial: {np.sum(np.linalg.norm(np.diff(initial_traj, axis=0), axis=1)):.2f} m
      Optimal: {np.sum(np.linalg.norm(np.diff(optimal_traj, axis=0), axis=1)):.2f} m
    """
    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('/home/claude/dr_optimization_test.png', dpi=150, bbox_inches='tight')
    print("\nDouglas-Rachford optimization test complete!")
