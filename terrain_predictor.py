"""
Advanced Feature 1: Learned Terrain Cost Prediction
AI model predicts terrain traversal cost using neural network
"""

import numpy as np
from config import Config

class TerrainPredictor:
    """
    Neural network-based terrain cost predictor
    Learns to predict traversal difficulty from local features
    """
    
    def __init__(self, input_dim=10, hidden_dim=20):
        self.config = Config()
        
        # Simple 2-layer neural network
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = 1  # terrain cost
        
        # Initialize weights (Xavier initialization)
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        
        self.W2 = np.random.randn(hidden_dim, self.output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(self.output_dim)
        
        # Training history
        self.loss_history = []
        self.trained = False
    
    def _relu(self, x):
        """ReLU activation"""
        return np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def _sigmoid(self, x):
        """Sigmoid activation (for output)"""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        """
        Forward pass
        
        Args:
            X: Input features [batch_size x input_dim]
            
        Returns:
            output: Predicted terrain cost [batch_size x 1]
            cache: Intermediate values for backprop
        """
        # Hidden layer
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        
        # Output layer (scaled to [0.5, 3.0] range for terrain costs)
        z2 = np.dot(a1, self.W2) + self.b2
        output = 0.5 + 2.5 * self._sigmoid(z2)
        
        cache = {'X': X, 'z1': z1, 'a1': a1, 'z2': z2}
        return output, cache
    
    def backward(self, output, y_true, cache):
        """
        Backward pass
        
        Args:
            output: Predicted values
            y_true: True labels
            cache: Cached values from forward pass
            
        Returns:
            gradients: Dict of gradients for all parameters
        """
        batch_size = cache['X'].shape[0]
        
        # Output layer gradients
        # For sigmoid output scaled to [0.5, 3.0]
        dz2 = (output - y_true) / batch_size
        dW2 = np.dot(cache['a1'].T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self._relu_derivative(cache['z1'])
        dW1 = np.dot(cache['X'].T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
             epochs=None, learning_rate=None, batch_size=32, verbose=True):
        """
        Train the terrain predictor
        
        Args:
            X_train: Training features [N x input_dim]
            y_train: Training labels [N x 1]
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Mini-batch size
            verbose: Print training progress
        """
        if epochs is None:
            epochs = self.config.TRAINING_EPOCHS
        if learning_rate is None:
            learning_rate = self.config.LEARNING_RATE
        
        n_samples = X_train.shape[0]
        n_batches = max(1, n_samples // batch_size)
        
        if verbose:
            print(f"Training terrain predictor for {epochs} epochs...")
            print(f"Training samples: {n_samples}, Batch size: {batch_size}")
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            
            # Mini-batch gradient descent
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx].reshape(-1, 1)
                
                # Forward pass
                output, cache = self.forward(X_batch)
                
                # Compute loss (MSE)
                loss = np.mean((output - y_batch) ** 2)
                epoch_loss += loss
                
                # Backward pass
                gradients = self.backward(output, y_batch, cache)
                
                # Update weights
                self.W1 -= learning_rate * gradients['dW1']
                self.b1 -= learning_rate * gradients['db1']
                self.W2 -= learning_rate * gradients['dW2']
                self.b2 -= learning_rate * gradients['db2']
            
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)
            
            # Validation
            if X_val is not None and y_val is not None:
                val_output, _ = self.forward(X_val)
                val_loss = np.mean((val_output - y_val.reshape(-1, 1)) ** 2)
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}")
        
        self.trained = True
        if verbose:
            print("Training complete!")
    
    def predict(self, X):
        """
        Predict terrain cost
        
        Args:
            X: Input features [N x input_dim] or [input_dim]
            
        Returns:
            predictions: Terrain cost predictions [N x 1] or scalar
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            single = True
        else:
            single = False
        
        output, _ = self.forward(X)
        
        if single:
            return output[0, 0]
        return output.flatten()
    
    def extract_features(self, position, environment, window_size=5):
        """
        Extract terrain features at a position
        
        Args:
            position: [x, y] position
            environment: Environment object
            window_size: Size of local window
            
        Returns:
            features: Feature vector [input_dim]
        """
        features = []
        
        # 1. Local terrain costs (grid sampling)
        for dx in np.linspace(-window_size, window_size, 3):
            for dy in np.linspace(-window_size, window_size, 3):
                sample_pos = position + np.array([dx, dy])
                terrain_cost = environment.get_terrain_cost(sample_pos)
                features.append(terrain_cost)
        
        # 2. Distance to nearest obstacle
        min_obs_dist = 100.0
        for obs in environment.static_obstacles:
            if obs['type'] == 'circle':
                dist = np.linalg.norm(position - obs['center']) - obs['radius']
                min_obs_dist = min(min_obs_dist, dist)
        features.append(min_obs_dist / 10.0)  # normalize
        
        # Pad or truncate to input_dim
        features = np.array(features)
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]
        
        return features
    
    def save_model(self, filepath):
        """Save model weights"""
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                loss_history=self.loss_history)
    
    def load_model(self, filepath):
        """Load model weights"""
        data = np.load(filepath)
        self.W1 = data['W1']
        self.b1 = data['b1']
        self.W2 = data['W2']
        self.b2 = data['b2']
        self.loss_history = list(data['loss_history'])
        self.trained = True


def generate_terrain_training_data(environment, num_samples=1000):
    """
    Generate training data for terrain predictor
    
    Args:
        environment: Environment object
        num_samples: Number of training samples
        
    Returns:
        X: Features [num_samples x feature_dim]
        y: Labels (terrain costs) [num_samples]
    """
    predictor = TerrainPredictor()
    
    X = []
    y = []
    
    # Sample random positions
    for _ in range(num_samples):
        # Random position in workspace
        pos = np.array([
            np.random.uniform(5, environment.width - 5),
            np.random.uniform(5, environment.height - 5)
        ])
        
        # Extract features
        features = predictor.extract_features(pos, environment)
        
        # Get true terrain cost
        cost = environment.get_terrain_cost(pos)
        
        X.append(features)
        y.append(cost)
    
    return np.array(X), np.array(y)


if __name__ == '__main__':
    # Test terrain predictor
    from environment import SimulationEnvironment
    import matplotlib.pyplot as plt
    
    env = SimulationEnvironment()
    
    # Generate training data
    print("Generating training data...")
    X_train, y_train = generate_terrain_training_data(env, num_samples=2000)
    X_val, y_val = generate_terrain_training_data(env, num_samples=500)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Train predictor
    predictor = TerrainPredictor(input_dim=X_train.shape[1])
    predictor.train(X_train, y_train, X_val, y_val, epochs=100, verbose=True)
    
    # Test predictions
    test_positions = [
        np.array([25, 25]),  # In rough terrain
        np.array([70, 70]),  # In smooth terrain
        np.array([50, 50])   # Normal terrain
    ]
    
    print("\nTest Predictions:")
    for pos in test_positions:
        features = predictor.extract_features(pos, env)
        predicted_cost = predictor.predict(features)
        true_cost = env.get_terrain_cost(pos)
        print(f"  Position {pos}: Predicted={predicted_cost:.3f}, True={true_cost:.3f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Learning curve
    axes[0].plot(predictor.loss_history, 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss History')
    axes[0].grid(True, alpha=0.3)
    
    # Prediction vs True
    y_pred = predictor.predict(X_val)
    axes[1].scatter(y_val, y_pred, alpha=0.5)
    axes[1].plot([0, 3], [0, 3], 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('True Terrain Cost')
    axes[1].set_ylabel('Predicted Terrain Cost')
    axes[1].set_title('Prediction Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/claude/terrain_predictor_test.png', dpi=150, bbox_inches='tight')
    
    # Save model
    predictor.save_model('/home/claude/terrain_model.npz')
    print("\nTerrain predictor test complete!")
