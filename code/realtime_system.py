"""
Main Simulation System
Real-Time Execution Loop with all features integrated
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import time

from config import Config
from environment import SimulationEnvironment
from motion_engine import RobotMotionEngine
from sensor_perception import SensorPerception
from energy_model import EnergyModel
from cost_function import CostFunction
from douglas_rachford import DouglasRachfordOptimizer
from terrain_predictor import TerrainPredictor, generate_terrain_training_data

class RealtimeNavigationSystem:
    """
    Complete real-time autonomous navigation system
    Integrates all modules for continuous operation
    """
    
    def __init__(self, start_position, goal_position, use_terrain_predictor=True):
        self.config = Config()
        
        # Core components
        self.environment = SimulationEnvironment()
        self.robot = RobotMotionEngine(start_position)
        self.sensor = SensorPerception(self.environment)
        self.energy = EnergyModel()
        self.cost_fn = CostFunction(self.environment, self.energy)
        self.optimizer = DouglasRachfordOptimizer(self.cost_fn, self.environment, 
                                                   self.energy)
        
        # Terrain predictor (AI feature)
        self.use_terrain_predictor = use_terrain_predictor
        if use_terrain_predictor:
            self.terrain_predictor = TerrainPredictor()
            self._train_terrain_predictor()
        
        # Mission parameters
        self.goal_position = goal_position
        self.start_position = start_position
        
        # State
        self.current_time = 0.0
        self.mission_complete = False
        self.planned_trajectory = None
        self.replan_counter = 0
        
        # Performance metrics
        self.metrics = {
            'total_replans': 0,
            'optimization_times': [],
            'energy_consumption': [],
            'path_length': 0.0,
            'computation_time_total': 0.0,
            'collisions': 0
        }
        
        # Visualization
        self.trajectory_history = [start_position.copy()]
    
    def _train_terrain_predictor(self):
        """Train the AI terrain predictor"""
        print("Training terrain cost predictor...")
        X_train, y_train = generate_terrain_training_data(
            self.environment, num_samples=1000)
        self.terrain_predictor.train(X_train, y_train, epochs=50, verbose=False)
        print("Terrain predictor ready!")
    
    def plan_trajectory(self, verbose=False):
        """
        Plan/replan trajectory using Douglas-Rachford optimization
        """
        start_time = time.time()
        
        # Create initial trajectory guess
        N = self.config.HORIZON_STEPS
        current_pos = self.robot.position
        
        # Straight line to goal
        initial_traj = np.linspace(current_pos, self.goal_position, N)
        
        # Optimize using Douglas-Rachford
        optimal_traj, info = self.optimizer.optimize(
            initial_traj,
            self.goal_position,
            self.robot.velocity,
            verbose=verbose
        )
        
        self.planned_trajectory = optimal_traj
        
        # Update metrics
        comp_time = time.time() - start_time
        self.metrics['optimization_times'].append(comp_time)
        self.metrics['computation_time_total'] += comp_time
        self.metrics['total_replans'] += 1
        
        if verbose:
            print(f"Replanning #{self.metrics['total_replans']}: "
                  f"Computation time: {comp_time:.4f}s")
        
        return optimal_traj
    
    def should_replan(self):
        """
        Decide if replanning is needed
        Triggers:
          - Dynamic obstacle detected nearby
          - Planned trajectory becomes infeasible
          - Regular interval
        """
        # Replan every N steps
        replan_interval = 10
        if self.replan_counter >= replan_interval:
            self.replan_counter = 0
            return True
        
        # Check for nearby dynamic obstacles
        detected_obstacles = self.sensor.sense_obstacles(self.robot.position)
        for obs in detected_obstacles:
            if obs['type'] == 'dynamic_circle':
                dist = np.linalg.norm(self.robot.position - obs['position'])
                if dist < 10.0:  # Close dynamic obstacle
                    return True
        
        # Check if current plan is collision-free
        if self.planned_trajectory is not None:
            next_pos = self.planned_trajectory[min(1, len(self.planned_trajectory)-1)]
            if self.environment.check_collision(next_pos):
                return True
        
        return False
    
    def execute_step(self, dt):
        """
        Execute one simulation step
        Real-time execution loop iteration
        """
        # Check mission completion
        dist_to_goal = np.linalg.norm(self.robot.position - self.goal_position)
        if dist_to_goal < 1.0:
            self.mission_complete = True
            return
        
        # Sense environment
        detected_obstacles = self.sensor.sense_obstacles(self.robot.position)
        
        # Update dynamic obstacles
        self.environment.update_dynamic_obstacles(dt)
        
        # Decide if replanning needed
        if self.planned_trajectory is None or self.should_replan():
            self.plan_trajectory(verbose=False)
        
        # Get control from planned trajectory
        if len(self.planned_trajectory) > 1:
            target_pos = self.planned_trajectory[1]
            desired_vel = (target_pos - self.robot.position) / dt
        else:
            # Go straight to goal
            desired_vel = (self.goal_position - self.robot.position)
            vel_mag = np.linalg.norm(desired_vel)
            if vel_mag > self.config.MAX_VELOCITY:
                desired_vel = desired_vel / vel_mag * self.config.MAX_VELOCITY
        
        # Get disturbances
        wind_force = self.environment.get_wind_force(self.robot.position)
        random_disturbance = np.random.randn(2) * self.config.EXTERNAL_FORCE_STD
        total_disturbance = wind_force + random_disturbance
        
        # Apply control
        self.robot.apply_control(desired_vel, dt, total_disturbance)
        
        # Check collision
        if self.environment.check_collision(self.robot.position):
            self.metrics['collisions'] += 1
            # Emergency stop and replan
            self.robot.velocity = np.zeros(2)
            self.planned_trajectory = None
        
        # Update energy
        in_recovery, recovery_rate = self.environment.in_recovery_zone(
            self.robot.position)
        
        if self.use_terrain_predictor:
            # Use AI to predict terrain cost
            features = self.terrain_predictor.extract_features(
                self.robot.position, self.environment)
            terrain_cost = self.terrain_predictor.predict(features)
        else:
            terrain_cost = self.environment.get_terrain_cost(self.robot.position)
        
        self.energy.update(
            self.robot.velocity,
            self.robot.acceleration,
            dt,
            in_recovery,
            recovery_rate,
            terrain_cost
        )
        
        # Update state
        self.current_time += dt
        self.replan_counter += 1
        
        # Shift planned trajectory
        if self.planned_trajectory is not None and len(self.planned_trajectory) > 1:
            self.planned_trajectory = self.planned_trajectory[1:]
        
        # Store trajectory
        self.trajectory_history.append(self.robot.position.copy())
        self.metrics['energy_consumption'].append(self.energy.battery_level)
    
    def run_simulation(self, max_time=None, dt=None, verbose=True):
        """
        Run complete simulation
        
        Args:
            max_time: Maximum simulation time (seconds)
            dt: Time step
            verbose: Print progress
        """
        if max_time is None:
            max_time = self.config.MAX_SIMULATION_TIME
        if dt is None:
            dt = self.config.DT
        
        if verbose:
            print("=" * 60)
            print("REAL-TIME TRAJECTORY OPTIMIZATION SIMULATION")
            print("=" * 60)
            print(f"Start: {self.start_position}")
            print(f"Goal:  {self.goal_position}")
            print(f"Max time: {max_time}s, dt: {dt}s")
            print(f"Terrain predictor: {'Enabled' if self.use_terrain_predictor else 'Disabled'}")
            print()
        
        # Initial plan
        self.plan_trajectory(verbose=verbose)
        
        # Simulation loop
        steps = 0
        while self.current_time < max_time and not self.mission_complete:
            self.execute_step(dt)
            steps += 1
            
            # Progress update
            if verbose and steps % 50 == 0:
                print(f"Time: {self.current_time:.1f}s | "
                      f"Battery: {self.energy.battery_level:.1f}% | "
                      f"Dist to goal: {np.linalg.norm(self.robot.position - self.goal_position):.1f}m | "
                      f"Replans: {self.metrics['total_replans']}")
            
            # Check battery
            if self.energy.battery_level < 1.0:
                if verbose:
                    print("\n⚠️  MISSION FAILED: Battery depleted!")
                break
        
        # Final metrics
        self.metrics['path_length'] = self.robot.compute_path_length()
        
        if verbose:
            print("\n" + "=" * 60)
            if self.mission_complete:
                print("✅ MISSION COMPLETE!")
            else:
                print("⏱️  Simulation time limit reached")
            print("=" * 60)
            self._print_statistics()
    
    def _print_statistics(self):
        """Print performance statistics"""
        print("\nPERFORMANCE METRICS:")
        print(f"  Total time:              {self.current_time:.2f} s")
        print(f"  Path length:             {self.metrics['path_length']:.2f} m")
        print(f"  Final battery:           {self.energy.battery_level:.1f} %")
        print(f"  Energy consumed:         {self.energy.total_consumed:.2f} %")
        print(f"  Energy recovered:        {self.energy.total_recovered:.2f} %")
        print(f"  Total replans:           {self.metrics['total_replans']}")
        print(f"  Collisions:              {self.metrics['collisions']}")
        
        if self.metrics['optimization_times']:
            avg_opt_time = np.mean(self.metrics['optimization_times'])
            max_opt_time = np.max(self.metrics['optimization_times'])
            print(f"  Avg optimization time:   {avg_opt_time*1000:.2f} ms")
            print(f"  Max optimization time:   {max_opt_time*1000:.2f} ms")
        
        print(f"  Total computation time:  {self.metrics['computation_time_total']:.3f} s")
        
        if self.mission_complete:
            dist_to_goal = np.linalg.norm(self.robot.position - self.goal_position)
            print(f"  Final distance to goal:  {dist_to_goal:.2f} m")
    
    def visualize_results(self, save_path=None):
        """
        Create comprehensive visualization of results
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Trajectory plot
        ax1 = fig.add_subplot(gs[0:2, 0:2])
        self.environment.visualize(ax=ax1)
        
        trajectory = np.array(self.trajectory_history)
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'b-', 
                linewidth=3, label='Actual Path', alpha=0.8)
        ax1.plot(self.start_position[0], self.start_position[1], 'go', 
                markersize=15, label='Start')
        ax1.plot(self.goal_position[0], self.goal_position[1], 'r*', 
                markersize=20, label='Goal')
        
        if self.mission_complete:
            ax1.plot(trajectory[-1, 0], trajectory[-1, 1], 'gd', 
                    markersize=15, label='Final Position')
        
        ax1.legend(loc='upper right')
        ax1.set_title('Trajectory and Environment', fontsize=14, fontweight='bold')
        
        # 2. Energy over time
        ax2 = fig.add_subplot(gs[0, 2])
        time_array = np.array(self.energy.time_history)
        energy_array = np.array(self.energy.energy_history)
        ax2.plot(time_array, energy_array, 'b-', linewidth=2)
        ax2.axhline(y=self.config.MIN_BATTERY_THRESHOLD, color='r', 
                   linestyle='--', label='Min Threshold')
        ax2.fill_between(time_array, 0, energy_array, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Battery Level (%)')
        ax2.set_title('Energy Profile')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. Velocity profile
        ax3 = fig.add_subplot(gs[1, 2])
        history = self.robot.get_trajectory_history()
        velocities = history['velocities']
        vel_magnitudes = np.linalg.norm(velocities, axis=1)
        ax3.plot(history['times'], vel_magnitudes, 'g-', linewidth=2)
        ax3.axhline(y=self.config.MAX_VELOCITY, color='r', linestyle='--', 
                   label='Max Velocity')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Velocity Profile')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Optimization convergence
        ax4 = fig.add_subplot(gs[2, 0])
        if self.metrics['optimization_times']:
            ax4.bar(range(len(self.metrics['optimization_times'])), 
                   np.array(self.metrics['optimization_times']) * 1000,
                   color='blue', alpha=0.7)
            ax4.axhline(y=100, color='r', linestyle='--', label='100ms')
            ax4.set_xlabel('Replanning Event')
            ax4.set_ylabel('Computation Time (ms)')
            ax4.set_title('Optimization Performance')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        
        # 5. Statistics table
        ax5 = fig.add_subplot(gs[2, 1:])
        ax5.axis('off')
        
        stats_text = f"""
        MISSION STATISTICS
        
        Status:                 {'✅ Complete' if self.mission_complete else '⏱️ Time Limit'}
        
        Distance Metrics:
          Path Length:          {self.metrics['path_length']:.2f} m
          Straight Line:        {np.linalg.norm(self.goal_position - self.start_position):.2f} m
          Path Efficiency:      {(np.linalg.norm(self.goal_position - self.start_position) / max(self.metrics['path_length'], 0.001)) * 100:.1f} %
        
        Energy Metrics:
          Initial Battery:      {self.energy.initial_battery:.1f} %
          Final Battery:        {self.energy.battery_level:.1f} %
          Total Consumed:       {self.energy.total_consumed:.2f} %
          Total Recovered:      {self.energy.total_recovered:.2f} %
          Energy Efficiency:    {self.energy.get_energy_efficiency():.1f} %
        
        Computation Metrics:
          Total Replans:        {self.metrics['total_replans']}
          Avg Replan Time:      {np.mean(self.metrics['optimization_times'])*1000:.2f} ms
          Total Comp. Time:     {self.metrics['computation_time_total']:.3f} s
          Real-time Factor:     {self.current_time / max(self.metrics['computation_time_total'], 0.001):.1f}x
        
        Safety Metrics:
          Collisions:           {self.metrics['collisions']}
          Success Rate:         {((1 - self.metrics['collisions']/max(len(self.trajectory_history), 1)) * 100):.1f} %
        """
        
        ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')
        
        fig.suptitle('Real-Time Trajectory Optimization Results', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nResults saved to: {save_path}")
        
        return fig
    
    def create_animation(self, save_path=None, fps=10):
        """
        Create animated visualization of the mission
        """
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Setup environment
        self.environment.visualize(ax=ax)
        
        # Initialize plot elements
        robot_circle = Circle(self.start_position, self.config.ROBOT_RADIUS, 
                             color='blue', alpha=0.7)
        ax.add_patch(robot_circle)
        
        trajectory_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.6)
        planned_line, = ax.plot([], [], 'r--', linewidth=1, alpha=0.4)
        
        battery_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                             fontsize=12, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Animation update function
        trajectory = np.array(self.trajectory_history)
        
        def update(frame):
            idx = min(frame, len(trajectory) - 1)
            
            # Update robot position
            robot_circle.center = trajectory[idx]
            
            # Update trajectory
            trajectory_line.set_data(trajectory[:idx+1, 0], trajectory[:idx+1, 1])
            
            # Update battery text
            battery_idx = min(idx, len(self.metrics['energy_consumption']) - 1)
            battery = self.metrics['energy_consumption'][battery_idx]
            battery_text.set_text(f'Battery: {battery:.1f}%\nTime: {idx * self.config.DT:.1f}s')
            
            return robot_circle, trajectory_line, battery_text
        
        anim = FuncAnimation(fig, update, frames=len(trajectory),
                           interval=1000/fps, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Animation saved to: {save_path}")
        
        return anim


if __name__ == '__main__':
    # Run a test simulation
    print("Initializing Real-Time Navigation System...")
    
    start = np.array([10, 10])
    goal = np.array([85, 85])
    
    system = RealtimeNavigationSystem(start, goal, use_terrain_predictor=True)
    
    # Run simulation
    system.run_simulation(max_time=50.0, verbose=True)
    
    # Visualize
    system.visualize_results(save_path='/home/claude/simulation_results.png')
    
    print("\nSimulation complete! Check simulation_results.png")
