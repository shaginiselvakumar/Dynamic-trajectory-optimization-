"""
Module 6: Energy Modeling
Tracks battery behavior with motion-based consumption and recovery zones
"""

import numpy as np
from config import Config

class EnergyModel:
    """Battery and energy consumption model"""
    
    def __init__(self, initial_battery=None):
        self.config = Config()
        
        # Battery state
        self.battery_level = (initial_battery if initial_battery is not None 
                             else self.config.INITIAL_BATTERY)
        self.initial_battery = self.battery_level
        
        # Energy tracking
        self.energy_history = [self.battery_level]
        self.consumption_history = []
        self.recovery_history = []
        self.time_history = [0.0]
        
        self.total_consumed = 0.0
        self.total_recovered = 0.0
        self.current_time = 0.0
    
    def update(self, velocity, acceleration, dt, in_recovery_zone=False, 
               recovery_rate=0.0, terrain_cost=1.0):
        """
        Update battery level based on robot activity
        
        Args:
            velocity: Current velocity vector [vx, vy]
            acceleration: Current acceleration vector [ax, ay]
            dt: Time step
            in_recovery_zone: Boolean, whether robot is in recovery zone
            recovery_rate: Energy recovery rate (%/second)
            terrain_cost: Terrain difficulty multiplier
        """
        if self.battery_level <= 0:
            self.battery_level = 0.0
            return
        
        # Base energy consumption (idle)
        base_consumption = self.config.BASE_ENERGY_RATE * dt
        
        # Motion energy (proportional to velocity and terrain)
        velocity_magnitude = np.linalg.norm(velocity)
        motion_consumption = (self.config.MOTION_ENERGY_COEF * 
                             velocity_magnitude * terrain_cost * dt)
        
        # Acceleration energy (proportional to acceleration magnitude)
        acceleration_magnitude = np.linalg.norm(acceleration)
        accel_consumption = (self.config.ACCEL_ENERGY_COEF * 
                            acceleration_magnitude * dt)
        
        # Total consumption
        total_consumption = base_consumption + motion_consumption + accel_consumption
        
        # Energy recovery
        recovery = 0.0
        if in_recovery_zone:
            recovery = recovery_rate * dt
            self.total_recovered += recovery
        
        # Update battery
        net_change = recovery - total_consumption
        self.battery_level = np.clip(self.battery_level + net_change, 0.0, 100.0)
        
        # Track consumption
        self.total_consumed += total_consumption
        self.current_time += dt
        
        # History
        self.energy_history.append(self.battery_level)
        self.consumption_history.append(total_consumption)
        self.recovery_history.append(recovery)
        self.time_history.append(self.current_time)
    
    def predict_energy_consumption(self, velocity_sequence, dt, terrain_costs=None):
        """
        Predict energy consumption for a planned trajectory
        
        Args:
            velocity_sequence: Array of velocities [N x 2]
            dt: Time step
            terrain_costs: Optional terrain cost for each step [N]
            
        Returns:
            predicted_consumption: Total energy predicted to be consumed
            battery_trajectory: Battery levels at each step
        """
        N = len(velocity_sequence)
        
        if terrain_costs is None:
            terrain_costs = np.ones(N)
        
        # Simulate battery
        battery_sim = self.battery_level
        battery_trajectory = [battery_sim]
        total_consumption = 0.0
        
        prev_velocity = np.zeros(2)
        
        for i in range(N):
            # Estimate acceleration
            acceleration = (velocity_sequence[i] - prev_velocity) / dt
            
            # Base consumption
            base = self.config.BASE_ENERGY_RATE * dt
            
            # Motion consumption
            vel_mag = np.linalg.norm(velocity_sequence[i])
            motion = self.config.MOTION_ENERGY_COEF * vel_mag * terrain_costs[i] * dt
            
            # Acceleration consumption
            accel_mag = np.linalg.norm(acceleration)
            accel = self.config.ACCEL_ENERGY_COEF * accel_mag * dt
            
            # Total
            consumption = base + motion + accel
            total_consumption += consumption
            battery_sim = max(battery_sim - consumption, 0.0)
            battery_trajectory.append(battery_sim)
            
            prev_velocity = velocity_sequence[i]
        
        return total_consumption, np.array(battery_trajectory)
    
    def is_feasible(self, min_threshold=None):
        """
        Check if battery level is above minimum threshold
        
        Args:
            min_threshold: Minimum acceptable battery level
            
        Returns:
            Boolean indicating feasibility
        """
        threshold = (min_threshold if min_threshold is not None 
                    else self.config.MIN_BATTERY_THRESHOLD)
        return self.battery_level >= threshold
    
    def get_energy_efficiency(self):
        """
        Compute energy efficiency metric
        
        Returns:
            Efficiency score (higher is better)
        """
        if self.total_consumed == 0:
            return 100.0
        
        # Efficiency = remaining battery / total used
        used = self.initial_battery - self.battery_level
        if used <= 0:
            return 100.0
        
        efficiency = (self.battery_level / self.initial_battery) * 100
        return efficiency
    
    def reset(self, battery_level=None):
        """Reset energy model to initial state"""
        self.battery_level = (battery_level if battery_level is not None 
                             else self.config.INITIAL_BATTERY)
        self.initial_battery = self.battery_level
        
        self.energy_history = [self.battery_level]
        self.consumption_history = []
        self.recovery_history = []
        self.time_history = [0.0]
        
        self.total_consumed = 0.0
        self.total_recovered = 0.0
        self.current_time = 0.0
    
    def get_statistics(self):
        """Get energy usage statistics"""
        return {
            'current_battery': self.battery_level,
            'initial_battery': self.initial_battery,
            'total_consumed': self.total_consumed,
            'total_recovered': self.total_recovered,
            'net_consumption': self.total_consumed - self.total_recovered,
            'efficiency': self.get_energy_efficiency(),
            'time_elapsed': self.current_time,
            'average_consumption_rate': self.total_consumed / max(self.current_time, 0.001)
        }
    
    def compute_energy_cost_to_goal(self, current_position, goal_position, 
                                   environment, avg_velocity=2.0):
        """
        Estimate energy cost to reach goal
        
        Args:
            current_position: Current position
            goal_position: Goal position
            environment: Environment object (for terrain)
            avg_velocity: Assumed average velocity
            
        Returns:
            Estimated energy cost
        """
        # Straight-line distance
        distance = np.linalg.norm(goal_position - current_position)
        
        # Time estimate
        time_estimate = distance / avg_velocity
        
        # Average terrain cost (simplified)
        avg_terrain = 1.5  # Conservative estimate
        
        # Energy estimate
        base_energy = self.config.BASE_ENERGY_RATE * time_estimate
        motion_energy = (self.config.MOTION_ENERGY_COEF * avg_velocity * 
                        avg_terrain * time_estimate)
        
        total_estimate = base_energy + motion_energy
        
        return total_estimate


if __name__ == '__main__':
    # Test energy model
    import matplotlib.pyplot as plt
    
    energy = EnergyModel(initial_battery=100.0)
    
    # Simulate some motion
    dt = 0.1
    times = []
    batteries = []
    
    for i in range(200):
        # Variable velocity
        t = i * dt
        velocity = np.array([2.0 * np.sin(t * 0.5), 2.0 * np.cos(t * 0.5)])
        acceleration = np.array([0.5, 0.3])
        
        # Recovery zone at certain times
        in_recovery = (50 < i < 80)
        recovery_rate = 2.0 if in_recovery else 0.0
        
        # Variable terrain
        terrain = 1.0 + 0.5 * np.sin(t * 0.3)
        
        energy.update(velocity, acceleration, dt, in_recovery, recovery_rate, terrain)
        
        times.append(t)
        batteries.append(energy.battery_level)
    
    # Statistics
    stats = energy.get_statistics()
    print("Energy Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Battery level
    axes[0].plot(times, batteries, 'b-', linewidth=2)
    axes[0].axhline(y=energy.config.MIN_BATTERY_THRESHOLD, color='r', 
                   linestyle='--', label='Min Threshold')
    axes[0].fill_between([5, 8], 0, 100, alpha=0.2, color='green', 
                         label='Recovery Zone')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Battery Level (%)')
    axes[0].set_title('Battery Level Over Time')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Consumption and recovery
    axes[1].plot(energy.time_history[1:], energy.consumption_history, 
                'r-', label='Consumption', linewidth=2)
    axes[1].plot(energy.time_history[1:], energy.recovery_history, 
                'g-', label='Recovery', linewidth=2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Energy Rate (%/s)')
    axes[1].set_title('Energy Consumption and Recovery Rates')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('/home/claude/energy_test.png', dpi=150, bbox_inches='tight')
    print("Energy model test complete!")
