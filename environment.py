"""
Module 1: Simulation Environment
Creates the virtual world with obstacles and terrain
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
from config import Config

class SimulationEnvironment:
    """Manages the complete navigation environment"""
    
    def __init__(self):
        self.config = Config()
        self.width, self.height = self.config.WORKSPACE_SIZE
        
        # Initialize components
        self.static_obstacles = []
        self.dynamic_obstacles = []
        self.terrain_zones = []
        self.recovery_zones = []
        self.wind_field = None
        
        self._initialize_environment()
    
    def _initialize_environment(self):
        """Set up all environmental elements"""
        self._create_static_obstacles()
        self._create_dynamic_obstacles()
        self._create_terrain_zones()
        self._create_recovery_zones()
        self._create_wind_field()
    
    def _create_static_obstacles(self):
        """Generate static obstacles (circles and rectangles)"""
        np.random.seed(42)  # for reproducibility
        
        for i in range(self.config.NUM_STATIC_OBSTACLES):
            obs_type = np.random.choice(['circle', 'rectangle'])
            
            if obs_type == 'circle':
                x = np.random.uniform(10, self.width - 10)
                y = np.random.uniform(10, self.height - 10)
                radius = np.random.uniform(2, 5)
                self.static_obstacles.append({
                    'type': 'circle',
                    'center': np.array([x, y]),
                    'radius': radius
                })
            else:
                x = np.random.uniform(10, self.width - 15)
                y = np.random.uniform(10, self.height - 15)
                width = np.random.uniform(3, 8)
                height = np.random.uniform(3, 8)
                self.static_obstacles.append({
                    'type': 'rectangle',
                    'corner': np.array([x, y]),
                    'width': width,
                    'height': height
                })
    
    def _create_dynamic_obstacles(self):
        """Generate moving obstacles"""
        np.random.seed(43)
        
        for i in range(self.config.NUM_DYNAMIC_OBSTACLES):
            x = np.random.uniform(15, self.width - 15)
            y = np.random.uniform(15, self.height - 15)
            
            # Random velocity direction
            angle = np.random.uniform(0, 2 * np.pi)
            vx = self.config.DYNAMIC_OBSTACLE_SPEED * np.cos(angle)
            vy = self.config.DYNAMIC_OBSTACLE_SPEED * np.sin(angle)
            
            self.dynamic_obstacles.append({
                'position': np.array([x, y]),
                'velocity': np.array([vx, vy]),
                'radius': np.random.uniform(1.5, 3.0),
                'id': i
            })
    
    def _create_terrain_zones(self):
        """Create different terrain cost zones"""
        # Rough terrain zone
        self.terrain_zones.append({
            'type': 'rough',
            'polygon': np.array([[20, 20], [40, 20], [40, 40], [20, 40]]),
            'cost_multiplier': self.config.TERRAIN_TYPES['rough']
        })
        
        # Smooth terrain zone
        self.terrain_zones.append({
            'type': 'smooth',
            'polygon': np.array([[60, 60], [80, 60], [80, 80], [60, 80]]),
            'cost_multiplier': self.config.TERRAIN_TYPES['smooth']
        })
    
    def _create_recovery_zones(self):
        """Create energy recovery zones"""
        self.recovery_zones.append({
            'center': np.array([50, 50]),
            'radius': 8.0,
            'recovery_rate': self.config.RECOVERY_RATE
        })
        
        self.recovery_zones.append({
            'center': np.array([80, 20]),
            'radius': 6.0,
            'recovery_rate': self.config.RECOVERY_RATE * 0.7
        })
    
    def _create_wind_field(self):
        """Create a wind field (disturbance)"""
        # Create spatial varying wind field
        grid_size = 20
        x = np.linspace(0, self.width, grid_size)
        y = np.linspace(0, self.height, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Wind pattern (simplified vortex-like)
        cx, cy = self.width / 2, self.height / 2
        dx = X - cx
        dy = Y - cy
        
        self.wind_field = {
            'x_grid': X,
            'y_grid': Y,
            'u': -dy / 10 * self.config.WIND_STRENGTH,  # x-component
            'v': dx / 10 * self.config.WIND_STRENGTH    # y-component
        }
    
    def update_dynamic_obstacles(self, dt):
        """Update positions of moving obstacles"""
        for obs in self.dynamic_obstacles:
            # Update position
            obs['position'] += obs['velocity'] * dt
            
            # Bounce off walls
            if obs['position'][0] <= obs['radius'] or obs['position'][0] >= self.width - obs['radius']:
                obs['velocity'][0] *= -1
                obs['position'][0] = np.clip(obs['position'][0], obs['radius'], 
                                            self.width - obs['radius'])
            
            if obs['position'][1] <= obs['radius'] or obs['position'][1] >= self.height - obs['radius']:
                obs['velocity'][1] *= -1
                obs['position'][1] = np.clip(obs['position'][1], obs['radius'], 
                                            self.height - obs['radius'])
    
    def check_collision(self, position):
        """Check if a position collides with any obstacle"""
        # Check static obstacles
        for obs in self.static_obstacles:
            if obs['type'] == 'circle':
                dist = np.linalg.norm(position - obs['center'])
                if dist < obs['radius'] + self.config.ROBOT_RADIUS:
                    return True
            elif obs['type'] == 'rectangle':
                corner = obs['corner']
                if (corner[0] - self.config.ROBOT_RADIUS <= position[0] <= 
                    corner[0] + obs['width'] + self.config.ROBOT_RADIUS and
                    corner[1] - self.config.ROBOT_RADIUS <= position[1] <= 
                    corner[1] + obs['height'] + self.config.ROBOT_RADIUS):
                    return True
        
        # Check dynamic obstacles
        for obs in self.dynamic_obstacles:
            dist = np.linalg.norm(position - obs['position'])
            if dist < obs['radius'] + self.config.ROBOT_RADIUS:
                return True
        
        return False
    
    def get_terrain_cost(self, position):
        """Get terrain traversal cost at position"""
        # Check if in any terrain zone
        for zone in self.terrain_zones:
            if self._point_in_polygon(position, zone['polygon']):
                return zone['cost_multiplier']
        
        return self.config.TERRAIN_TYPES['normal']
    
    def in_recovery_zone(self, position):
        """Check if position is in recovery zone and return recovery rate"""
        for zone in self.recovery_zones:
            dist = np.linalg.norm(position - zone['center'])
            if dist <= zone['radius']:
                return True, zone['recovery_rate']
        return False, 0.0
    
    def get_wind_force(self, position):
        """Get wind force at a given position"""
        if not self.config.WIND_ENABLED:
            return np.zeros(2)
        
        # Bilinear interpolation
        x_idx = np.clip(int(position[0] / self.width * (self.wind_field['x_grid'].shape[1] - 1)), 
                       0, self.wind_field['x_grid'].shape[1] - 1)
        y_idx = np.clip(int(position[1] / self.height * (self.wind_field['y_grid'].shape[0] - 1)), 
                       0, self.wind_field['y_grid'].shape[0] - 1)
        
        u = self.wind_field['u'][y_idx, x_idx]
        v = self.wind_field['v'][y_idx, x_idx]
        
        # Add random variation
        noise = np.random.randn(2) * self.config.WIND_VARIATION
        
        return np.array([u, v]) + noise
    
    def _point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using ray casting"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def visualize(self, ax=None):
        """Visualize the environment"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Draw static obstacles
        for obs in self.static_obstacles:
            if obs['type'] == 'circle':
                circle = Circle(obs['center'], obs['radius'], 
                              color='gray', alpha=0.7, label='Static Obstacle')
                ax.add_patch(circle)
            elif obs['type'] == 'rectangle':
                rect = Rectangle(obs['corner'], obs['width'], obs['height'],
                               color='gray', alpha=0.7)
                ax.add_patch(rect)
        
        # Draw dynamic obstacles
        for obs in self.dynamic_obstacles:
            circle = Circle(obs['position'], obs['radius'], 
                          color='red', alpha=0.5, label='Dynamic Obstacle')
            ax.add_patch(circle)
        
        # Draw terrain zones
        for zone in self.terrain_zones:
            poly = Polygon(zone['polygon'], alpha=0.3, 
                         color='brown' if zone['type'] == 'rough' else 'green')
            ax.add_patch(poly)
        
        # Draw recovery zones
        for zone in self.recovery_zones:
            circle = Circle(zone['center'], zone['radius'], 
                          color='blue', alpha=0.2, label='Recovery Zone')
            ax.add_patch(circle)
        
        # Draw wind field (quiver plot)
        if self.config.WIND_ENABLED:
            skip = 3
            ax.quiver(self.wind_field['x_grid'][::skip, ::skip], 
                     self.wind_field['y_grid'][::skip, ::skip],
                     self.wind_field['u'][::skip, ::skip], 
                     self.wind_field['v'][::skip, ::skip],
                     alpha=0.3, scale=20)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Simulation Environment')
        
        return ax


if __name__ == '__main__':
    # Test the environment
    env = SimulationEnvironment()
    env.visualize()
    plt.savefig('/home/claude/environment_test.png', dpi=150, bbox_inches='tight')
    print("Environment created and saved!")
