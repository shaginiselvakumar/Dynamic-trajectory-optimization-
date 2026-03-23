"""
Module 2: Robot Motion Engine
Handles physical movement and kinematic constraints
"""

import numpy as np
from config import Config

class RobotMotionEngine:
    """Robot kinematics and dynamics"""
    
    def __init__(self, initial_position, initial_velocity=None):
        self.config = Config()
        
        # State variables
        self.position = np.array(initial_position, dtype=float)
        self.velocity = np.array(initial_velocity if initial_velocity is not None 
                                else [0.0, 0.0], dtype=float)
        self.acceleration = np.zeros(2)
        
        # History for analysis
        self.position_history = [self.position.copy()]
        self.velocity_history = [self.velocity.copy()]
        self.time_history = [0.0]
        
        self.current_time = 0.0
    
    def apply_control(self, desired_velocity, dt, external_force=None):
        """
        Apply control input with acceleration limits
        
        Args:
            desired_velocity: Target velocity [vx, vy]
            dt: Time step
            external_force: External disturbances (wind, etc.)
        """
        # Compute required acceleration
        velocity_error = desired_velocity - self.velocity
        desired_acceleration = velocity_error / dt
        
        # Apply acceleration limits
        accel_magnitude = np.linalg.norm(desired_acceleration)
        if accel_magnitude > self.config.MAX_ACCELERATION:
            desired_acceleration = (desired_acceleration / accel_magnitude * 
                                  self.config.MAX_ACCELERATION)
        
        self.acceleration = desired_acceleration
        
        # Apply external forces (disturbances)
        if external_force is not None:
            # Force affects acceleration (F = ma, assume unit mass)
            self.acceleration += external_force
        
        # Update velocity (with limits)
        self.velocity += self.acceleration * dt
        velocity_magnitude = np.linalg.norm(self.velocity)
        if velocity_magnitude > self.config.MAX_VELOCITY:
            self.velocity = (self.velocity / velocity_magnitude * 
                           self.config.MAX_VELOCITY)
        
        # Update position
        self.position += self.velocity * dt
        
        # Update time
        self.current_time += dt
        
        # Store history
        self.position_history.append(self.position.copy())
        self.velocity_history.append(self.velocity.copy())
        self.time_history.append(self.current_time)
    
    def predict_trajectory(self, control_sequence, dt):
        """
        Predict future trajectory given control sequence
        
        Args:
            control_sequence: Array of velocity commands [N x 2]
            dt: Time step
            
        Returns:
            Predicted positions [N x 2]
        """
        N = len(control_sequence)
        predicted_positions = np.zeros((N, 2))
        
        # Simulate forward from current state
        pos = self.position.copy()
        vel = self.velocity.copy()
        
        for i in range(N):
            # Apply control
            vel_error = control_sequence[i] - vel
            accel = vel_error / dt
            
            # Limit acceleration
            accel_mag = np.linalg.norm(accel)
            if accel_mag > self.config.MAX_ACCELERATION:
                accel = accel / accel_mag * self.config.MAX_ACCELERATION
            
            # Update velocity
            vel += accel * dt
            vel_mag = np.linalg.norm(vel)
            if vel_mag > self.config.MAX_VELOCITY:
                vel = vel / vel_mag * self.config.MAX_VELOCITY
            
            # Update position
            pos += vel * dt
            predicted_positions[i] = pos
        
        return predicted_positions
    
    def get_state(self):
        """Return current state"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'acceleration': self.acceleration.copy(),
            'time': self.current_time
        }
    
    def set_state(self, position, velocity=None):
        """Manually set robot state"""
        self.position = np.array(position, dtype=float)
        if velocity is not None:
            self.velocity = np.array(velocity, dtype=float)
    
    def get_trajectory_history(self):
        """Get complete trajectory history"""
        return {
            'positions': np.array(self.position_history),
            'velocities': np.array(self.velocity_history),
            'times': np.array(self.time_history)
        }
    
    def compute_path_length(self):
        """Compute total path length traveled"""
        positions = np.array(self.position_history)
        if len(positions) < 2:
            return 0.0
        
        diffs = np.diff(positions, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)


class KinematicModel:
    """Different kinematic motion models"""
    
    @staticmethod
    def unicycle_model(state, control, dt):
        """
        Unicycle model: [x, y, theta]
        Control: [v, omega] (linear vel, angular vel)
        """
        x, y, theta = state
        v, omega = control
        
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + omega * dt
        
        return np.array([x_next, y_next, theta_next])
    
    @staticmethod
    def double_integrator(state, control, dt):
        """
        Double integrator: [x, y, vx, vy]
        Control: [ax, ay] (accelerations)
        """
        x, y, vx, vy = state
        ax, ay = control
        
        x_next = x + vx * dt + 0.5 * ax * dt**2
        y_next = y + vy * dt + 0.5 * ay * dt**2
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt
        
        return np.array([x_next, y_next, vx_next, vy_next])
    
    @staticmethod
    def ackermann_model(state, control, dt, wheelbase=2.0):
        """
        Ackermann steering (car-like)
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        """
        x, y, theta, v = state
        a, delta = control
        
        x_next = x + v * np.cos(theta) * dt
        y_next = y + v * np.sin(theta) * dt
        theta_next = theta + (v / wheelbase) * np.tan(delta) * dt
        v_next = v + a * dt
        
        return np.array([x_next, y_next, theta_next, v_next])


if __name__ == '__main__':
    # Test the motion engine
    robot = RobotMotionEngine(initial_position=[10, 10])
    
    # Simulate some motion
    dt = 0.1
    for i in range(50):
        # Simple circular motion
        angle = i * 0.1
        desired_vel = np.array([np.cos(angle), np.sin(angle)]) * 2.0
        robot.apply_control(desired_vel, dt)
    
    history = robot.get_trajectory_history()
    print(f"Path length: {robot.compute_path_length():.2f} m")
    print(f"Final position: {robot.position}")
    
    # Plot trajectory
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.plot(history['positions'][:, 0], history['positions'][:, 1], 'b-', linewidth=2)
    plt.plot(history['positions'][0, 0], history['positions'][0, 1], 'go', 
             markersize=10, label='Start')
    plt.plot(history['positions'][-1, 0], history['positions'][-1, 1], 'ro', 
             markersize=10, label='End')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Robot Motion Test')
    plt.legend()
    plt.savefig('/home/claude/motion_test.png', dpi=150, bbox_inches='tight')
    print("Motion test complete!")
