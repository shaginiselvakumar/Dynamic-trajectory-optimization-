"""
Configuration file for Real-Time Trajectory Optimization
using Douglas-Rachford Splitting
"""

import numpy as np

class Config:
    """System-wide configuration parameters"""
    
    # =========================
    # Environment Settings
    # =========================
    WORKSPACE_SIZE = (100, 100)  # meters
    GRID_RESOLUTION = 0.5  # meters
    DT = 0.1  # time step (seconds)
    
    # =========================
    # Robot Parameters
    # =========================
    MAX_VELOCITY = 5.0  # m/s
    MAX_ACCELERATION = 2.0  # m/s^2
    ROBOT_RADIUS = 0.5  # meters
    
    # =========================
    # Prediction Horizon
    # =========================
    HORIZON_STEPS = 20  # N steps ahead
    HORIZON_TIME = 2.0  # seconds
    
    # =========================
    # Sensor Parameters
    # =========================
    SENSOR_RANGE = 15.0  # meters
    SENSOR_NOISE_STD = 0.3  # meters (Gaussian)
    DETECTION_PROBABILITY = 0.95
    
    # =========================
    # Energy Model
    # =========================
    INITIAL_BATTERY = 100.0  # percent
    BASE_ENERGY_RATE = 0.5  # %/second (idle)
    MOTION_ENERGY_COEF = 0.1  # %/(m/s)
    ACCEL_ENERGY_COEF = 0.05  # %/(m/s^2)
    RECOVERY_RATE = 2.0  # %/second in recovery zone
    MIN_BATTERY_THRESHOLD = 10.0  # percent
    
    # =========================
    # Cost Function Weights
    # =========================
    WEIGHT_LENGTH = 1.0
    WEIGHT_ENERGY = 2.0
    WEIGHT_SMOOTHNESS = 1.5
    WEIGHT_OBSTACLE = 100.0
    WEIGHT_GOAL = 50.0
    
    # =========================
    # Douglas-Rachford Parameters
    # =========================
    DR_MAX_ITERATIONS = 100
    DR_TOLERANCE = 1e-4
    DR_GAMMA = 0.5  # relaxation parameter
    DR_LAMBDA = 1.0  # step size
    
    # =========================
    # Obstacle Parameters
    # =========================
    NUM_STATIC_OBSTACLES = 8
    NUM_DYNAMIC_OBSTACLES = 3
    DYNAMIC_OBSTACLE_SPEED = 1.0  # m/s
    OBSTACLE_SAFETY_MARGIN = 1.0  # meters
    
    # =========================
    # Disturbance Parameters
    # =========================
    WIND_ENABLED = True
    WIND_STRENGTH = 0.5  # m/s
    WIND_VARIATION = 0.2  # randomness
    EXTERNAL_FORCE_STD = 0.3  # random force noise
    
    # =========================
    # Multi-Agent Parameters
    # =========================
    NUM_AGENTS = 3
    INTER_AGENT_DISTANCE = 2.0  # minimum distance between agents
    COMMUNICATION_RANGE = 20.0  # meters
    
    # =========================
    # Terrain Zones
    # =========================
    TERRAIN_TYPES = {
        'normal': 1.0,
        'rough': 2.5,
        'smooth': 0.7,
        'recovery': -1.0  # negative = energy gain
    }
    
    # =========================
    # Visualization
    # =========================
    PLOT_TRAJECTORY = True
    PLOT_ENERGY = True
    ANIMATION_SPEED = 50  # ms per frame
    SAVE_RESULTS = True
    
    # =========================
    # Simulation
    # =========================
    MAX_SIMULATION_TIME = 60.0  # seconds
    REAL_TIME_MODE = False  # set True for actual real-time execution
    
    # =========================
    # Learning Parameters (for terrain prediction)
    # =========================
    TERRAIN_FEATURES = ['gradient', 'roughness', 'elevation']
    LEARNING_RATE = 0.001
    TRAINING_EPOCHS = 50
