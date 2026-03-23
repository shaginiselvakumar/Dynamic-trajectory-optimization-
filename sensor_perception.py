"""
Module 3: Sensor & Perception Layer
Simulates realistic sensor behavior with noise and limitations
"""

import numpy as np
from config import Config

class SensorPerception:
    """Robot sensors with realistic noise and limitations"""
    
    def __init__(self, environment):
        self.config = Config()
        self.environment = environment
        
        # Sensor characteristics
        self.range = self.config.SENSOR_RANGE
        self.noise_std = self.config.SENSOR_NOISE_STD
        self.detection_prob = self.config.DETECTION_PROBABILITY
    
    def sense_obstacles(self, robot_position):
        """
        Detect obstacles within sensor range with noise
        
        Args:
            robot_position: Current robot position [x, y]
            
        Returns:
            detected_obstacles: List of detected obstacle information
        """
        detected_obstacles = []
        
        # Sense static obstacles
        for obs in self.environment.static_obstacles:
            if obs['type'] == 'circle':
                # Check if within range
                dist_to_center = np.linalg.norm(robot_position - obs['center'])
                
                if dist_to_center - obs['radius'] <= self.range:
                    # Detection probability
                    if np.random.random() < self.detection_prob:
                        # Add measurement noise
                        noisy_center = obs['center'] + np.random.randn(2) * self.noise_std
                        noisy_radius = obs['radius'] + np.random.randn() * self.noise_std * 0.5
                        
                        detected_obstacles.append({
                            'type': 'circle',
                            'center': noisy_center,
                            'radius': max(noisy_radius, 0.5),  # minimum radius
                            'confidence': self._compute_confidence(dist_to_center)
                        })
            
            elif obs['type'] == 'rectangle':
                # Approximate rectangle center
                center = obs['corner'] + np.array([obs['width'], obs['height']]) / 2
                dist_to_center = np.linalg.norm(robot_position - center)
                
                if dist_to_center <= self.range:
                    if np.random.random() < self.detection_prob:
                        # Add noise to corner position
                        noisy_corner = obs['corner'] + np.random.randn(2) * self.noise_std
                        noisy_width = obs['width'] + np.random.randn() * self.noise_std * 0.5
                        noisy_height = obs['height'] + np.random.randn() * self.noise_std * 0.5
                        
                        detected_obstacles.append({
                            'type': 'rectangle',
                            'corner': noisy_corner,
                            'width': max(noisy_width, 1.0),
                            'height': max(noisy_height, 1.0),
                            'confidence': self._compute_confidence(dist_to_center)
                        })
        
        # Sense dynamic obstacles
        for obs in self.environment.dynamic_obstacles:
            dist_to_center = np.linalg.norm(robot_position - obs['position'])
            
            if dist_to_center - obs['radius'] <= self.range:
                if np.random.random() < self.detection_prob:
                    # Add measurement noise
                    noisy_position = obs['position'] + np.random.randn(2) * self.noise_std
                    noisy_radius = obs['radius'] + np.random.randn() * self.noise_std * 0.5
                    
                    # Estimate velocity (noisy)
                    noisy_velocity = obs['velocity'] + np.random.randn(2) * self.noise_std
                    
                    detected_obstacles.append({
                        'type': 'dynamic_circle',
                        'position': noisy_position,
                        'velocity': noisy_velocity,
                        'radius': max(noisy_radius, 0.5),
                        'confidence': self._compute_confidence(dist_to_center)
                    })
        
        return detected_obstacles
    
    def predict_dynamic_obstacles(self, detected_obstacles, time_horizon, dt):
        """
        Predict future positions of dynamic obstacles
        
        Args:
            detected_obstacles: List of detected obstacles
            time_horizon: How far to predict (seconds)
            dt: Time step
            
        Returns:
            predictions: Dict of obstacle predictions
        """
        predictions = {}
        num_steps = int(time_horizon / dt)
        
        for i, obs in enumerate(detected_obstacles):
            if obs['type'] == 'dynamic_circle':
                # Simple constant velocity prediction
                future_positions = np.zeros((num_steps, 2))
                
                for t in range(num_steps):
                    future_positions[t] = obs['position'] + obs['velocity'] * (t + 1) * dt
                
                predictions[f'obstacle_{i}'] = {
                    'positions': future_positions,
                    'radius': obs['radius'],
                    'uncertainty': self.noise_std * np.sqrt(np.arange(1, num_steps + 1))
                }
        
        return predictions
    
    def sense_terrain(self, robot_position):
        """
        Get terrain information at robot position (with noise)
        
        Args:
            robot_position: Current position
            
        Returns:
            terrain_cost: Estimated terrain traversal cost
        """
        # True terrain cost
        true_cost = self.environment.get_terrain_cost(robot_position)
        
        # Add measurement noise (multiplicative)
        noise_factor = 1.0 + np.random.randn() * 0.1
        noisy_cost = true_cost * noise_factor
        
        return max(noisy_cost, 0.1)  # minimum cost
    
    def sense_recovery_zone(self, robot_position):
        """
        Detect if in recovery zone
        
        Args:
            robot_position: Current position
            
        Returns:
            in_zone: Boolean
            recovery_rate: Energy recovery rate (noisy)
        """
        in_zone, true_rate = self.environment.in_recovery_zone(robot_position)
        
        if in_zone:
            # Add noise to recovery rate
            noisy_rate = true_rate * (1.0 + np.random.randn() * 0.05)
            return True, max(noisy_rate, 0.0)
        
        return False, 0.0
    
    def measure_distance_to_goal(self, robot_position, goal_position):
        """
        Measure distance to goal with noise
        
        Args:
            robot_position: Current position
            goal_position: Goal position
            
        Returns:
            noisy_distance: Measured distance
            noisy_bearing: Measured bearing angle
        """
        # True values
        true_vector = goal_position - robot_position
        true_distance = np.linalg.norm(true_vector)
        true_bearing = np.arctan2(true_vector[1], true_vector[0])
        
        # Add noise
        noisy_distance = true_distance + np.random.randn() * self.noise_std
        noisy_bearing = true_bearing + np.random.randn() * 0.05  # 0.05 rad ~ 3 degrees
        
        return max(noisy_distance, 0.0), noisy_bearing
    
    def _compute_confidence(self, distance):
        """
        Compute detection confidence based on distance
        Closer objects have higher confidence
        """
        # Exponential decay with distance
        confidence = np.exp(-distance / (self.range / 2))
        return np.clip(confidence, 0.1, 1.0)
    
    def create_local_occupancy_grid(self, robot_position, grid_size=50, resolution=0.5):
        """
        Create local occupancy grid map
        
        Args:
            robot_position: Center of grid
            grid_size: Grid dimension (grid_size x grid_size)
            resolution: Cell size in meters
            
        Returns:
            occupancy_grid: 2D array (0=free, 1=occupied, 0.5=unknown)
        """
        # Initialize grid (0.5 = unknown)
        grid = np.ones((grid_size, grid_size)) * 0.5
        
        # Grid bounds
        half_size = grid_size * resolution / 2
        x_min = robot_position[0] - half_size
        y_min = robot_position[1] - half_size
        
        # Detect obstacles
        detected_obs = self.sense_obstacles(robot_position)
        
        # Fill in grid
        for i in range(grid_size):
            for j in range(grid_size):
                # Cell center in world coordinates
                cell_x = x_min + (i + 0.5) * resolution
                cell_y = y_min + (j + 0.5) * resolution
                cell_pos = np.array([cell_x, cell_y])
                
                # Check if within sensor range
                if np.linalg.norm(cell_pos - robot_position) > self.range:
                    continue
                
                # Check against detected obstacles
                occupied = False
                for obs in detected_obs:
                    if obs['type'] in ['circle', 'dynamic_circle']:
                        dist = np.linalg.norm(cell_pos - obs.get('center', obs.get('position')))
                        if dist < obs['radius']:
                            occupied = True
                            break
                    elif obs['type'] == 'rectangle':
                        corner = obs['corner']
                        if (corner[0] <= cell_x <= corner[0] + obs['width'] and
                            corner[1] <= cell_y <= corner[1] + obs['height']):
                            occupied = True
                            break
                
                grid[j, i] = 1.0 if occupied else 0.0
        
        return grid
    
    def get_sensor_stats(self):
        """Return sensor specifications"""
        return {
            'range': self.range,
            'noise_std': self.noise_std,
            'detection_probability': self.detection_prob
        }


if __name__ == '__main__':
    # Test sensor system
    from environment import SimulationEnvironment
    import matplotlib.pyplot as plt
    
    env = SimulationEnvironment()
    sensor = SensorPerception(env)
    
    # Test detection
    robot_pos = np.array([30, 30])
    detected = sensor.sense_obstacles(robot_pos)
    
    print(f"Detected {len(detected)} obstacles")
    for i, obs in enumerate(detected):
        print(f"  Obstacle {i}: {obs['type']}, confidence: {obs['confidence']:.2f}")
    
    # Create occupancy grid
    grid = sensor.create_local_occupancy_grid(robot_pos)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Environment
    env.visualize(ax=ax1)
    ax1.plot(robot_pos[0], robot_pos[1], 'b*', markersize=20, label='Robot')
    circle = plt.Circle(robot_pos, sensor.range, color='blue', fill=False, 
                       linestyle='--', linewidth=2, label='Sensor Range')
    ax1.add_patch(circle)
    ax1.legend()
    
    # Occupancy grid
    im = ax2.imshow(grid, origin='lower', cmap='gray', vmin=0, vmax=1)
    ax2.set_title('Local Occupancy Grid')
    ax2.set_xlabel('Grid X')
    ax2.set_ylabel('Grid Y')
    plt.colorbar(im, ax=ax2, label='Occupancy (0=free, 1=occupied)')
    
    plt.tight_layout()
    plt.savefig('/home/claude/sensor_test.png', dpi=150, bbox_inches='tight')
    print("Sensor test complete!")
