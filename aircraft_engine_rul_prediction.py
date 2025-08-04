"""
Aircraft Engine Remaining Useful Life (RUL) Prediction
Using NASA C-MAPSS Turbofan Engine Degradation Dataset

This script implements a complete pipeline for predicting the remaining useful life
of aircraft engines using LSTM neural networks for time series analysis.

Dataset: NASA C-MAPSS FD001 subset
- train_FD001.txt: Training data
- test_FD001.txt: Test data  
- RUL_FD001.txt: True RUL values for test data

Author: ML Engineer
Date: 2025-08-04
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import keras
from keras import layers
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class AircraftEngineRULPredictor:
    """
    Complete pipeline for aircraft engine RUL prediction using LSTM networks
    """
    
    def __init__(self, sequence_length=50):
        """
        Initialize the RUL predictor
        
        Args:
            sequence_length (int): Length of input sequences for LSTM
        """
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.model = None
        
        # Define column names based on C-MAPSS dataset documentation
        self.column_names = ['unit_number', 'time_in_cycles'] + \
                           [f'setting_{i}' for i in range(1, 4)] + \
                           [f'sensor_{i}' for i in range(1, 22)]
        
        # Columns to drop (constant or low variance)
        self.columns_to_drop = ['setting_1', 'setting_2', 'setting_3', 
                               'sensor_1', 'sensor_5', 'sensor_6', 'sensor_10', 
                               'sensor_16', 'sensor_18', 'sensor_19']
    
    def load_data(self, train_path, test_path, rul_path):
        """
        Load and prepare the dataset files
        
        Args:
            train_path (str): Path to training data file
            test_path (str): Path to test data file
            rul_path (str): Path to true RUL values file
            
        Returns:
            tuple: (train_df, test_df, true_rul)
        """
        print("Loading dataset files...")
        
        # Load training data
        train_df = pd.read_csv(train_path, sep=' ', header=None)
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)  # Remove empty columns
        train_df.columns = self.column_names
        
        # Load test data
        test_df = pd.read_csv(test_path, sep=' ', header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)  # Remove empty columns
        test_df.columns = self.column_names
        
        # Load true RUL values
        true_rul = pd.read_csv(rul_path, sep=' ', header=None)
        true_rul.columns = ['RUL']
        
        # Remove low-variance columns
        train_df = train_df.drop(columns=self.columns_to_drop)
        test_df = test_df.drop(columns=self.columns_to_drop)
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"True RUL shape: {true_rul.shape}")
        
        return train_df, test_df, true_rul
    
    def calculate_rul(self, df):
        """
        Calculate Remaining Useful Life for training data
        
        Args:
            df (DataFrame): Training dataframe
            
        Returns:
            DataFrame: Dataframe with RUL column added
        """
        print("Calculating RUL for training data...")
        
        # Calculate maximum cycle for each engine
        max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycle']
        
        # Merge with original dataframe
        df_with_max = df.merge(max_cycles, on='unit_number', how='left')
        
        # Calculate RUL = max_cycle - current_cycle
        df_with_max['RUL'] = df_with_max['max_cycle'] - df_with_max['time_in_cycles']
        
        # Drop the helper column
        df_with_max = df_with_max.drop(columns=['max_cycle'])
        
        print(f"RUL calculated. Min RUL: {df_with_max['RUL'].min()}, Max RUL: {df_with_max['RUL'].max()}")
        
        return df_with_max
    
    def normalize_data(self, train_df, test_df):
        """
        Normalize sensor data using MinMaxScaler
        
        Args:
            train_df (DataFrame): Training dataframe
            test_df (DataFrame): Test dataframe
            
        Returns:
            tuple: (normalized_train_df, normalized_test_df)
        """
        print("Normalizing sensor data...")
        
        # Get sensor columns (exclude unit_number, time_in_cycles, and RUL if present)
        sensor_cols = [col for col in train_df.columns 
                      if col not in ['unit_number', 'time_in_cycles', 'RUL']]
        
        # Fit scaler on training data only
        self.scaler.fit(train_df[sensor_cols])
        
        # Transform both training and test data
        train_normalized = train_df.copy()
        test_normalized = test_df.copy()
        
        train_normalized[sensor_cols] = self.scaler.transform(train_df[sensor_cols])
        test_normalized[sensor_cols] = self.scaler.transform(test_df[sensor_cols])
        
        print(f"Normalized {len(sensor_cols)} sensor columns")
        
        return train_normalized, test_normalized
    
    def create_sequences(self, df, target_col='RUL'):
        """
        Create sequences for LSTM input
        
        Args:
            df (DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            tuple: (X, y) sequences
        """
        print(f"Creating sequences with length {self.sequence_length}...")
        
        X, y = [], []
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns 
                      if col not in ['unit_number', 'time_in_cycles', target_col]]
        
        # Group by engine unit
        for unit_id in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit_id].sort_values('time_in_cycles')
            
            # Create sequences for this engine
            for i in range(len(unit_data) - self.sequence_length + 1):
                # Input sequence (sensor data)
                sequence = unit_data[sensor_cols].iloc[i:i + self.sequence_length].values
                X.append(sequence)
                
                # Target (RUL at the end of sequence)
                if target_col in df.columns:
                    target = unit_data[target_col].iloc[i + self.sequence_length - 1]
                    y.append(target)
        
        X = np.array(X)
        y = np.array(y) if y else None
        
        print(f"Created {len(X)} sequences")
        print(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def create_test_sequences(self, df):
        """
        Create test sequences - only the last sequence for each engine
        
        Args:
            df (DataFrame): Test dataframe
            
        Returns:
            numpy.array: Test sequences
        """
        print("Creating test sequences (last sequence per engine)...")
        
        X_test = []
        
        # Get sensor columns
        sensor_cols = [col for col in df.columns 
                      if col not in ['unit_number', 'time_in_cycles']]
        
        # For each engine, take the last sequence_length cycles
        for unit_id in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit_id].sort_values('time_in_cycles')
            
            # Take last sequence_length rows (or all if less than sequence_length)
            if len(unit_data) >= self.sequence_length:
                sequence = unit_data[sensor_cols].tail(self.sequence_length).values
            else:
                # Pad with first row if not enough data
                sequence = unit_data[sensor_cols].values
                padding = np.tile(sequence[0], (self.sequence_length - len(sequence), 1))
                sequence = np.vstack([padding, sequence])
            
            X_test.append(sequence)
        
        X_test = np.array(X_test)
        print(f"Created {len(X_test)} test sequences")
        print(f"Test sequence shape: {X_test.shape}")
        
        return X_test
    
    def build_model(self, input_shape):
        """
        Build LSTM model architecture
        
        Args:
            input_shape (tuple): Shape of input sequences
            
        Returns:
            keras.Model: Compiled LSTM model
        """
        print("Building LSTM model...")
        
        model = keras.Sequential([
            # First LSTM layer with return_sequences=True
            layers.LSTM(100, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            
            # Second LSTM layer
            layers.LSTM(50, return_sequences=False),
            layers.Dropout(0.2),
            
            # Dense output layer for regression
            layers.Dense(1, activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print("Model architecture:")
        model.summary()
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, epochs=100, batch_size=32, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            X_train (numpy.array): Training sequences
            y_train (numpy.array): Training targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
            
        Returns:
            keras.callbacks.History: Training history
        """
        print(f"Training model for {epochs} epochs...")
        
        # Define callbacks
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        print("Training completed!")
        return history
    
    def evaluate_model(self, X_test, y_true):
        """
        Evaluate model performance
        
        Args:
            X_test (numpy.array): Test sequences
            y_true (numpy.array): True RUL values
            
        Returns:
            tuple: (predictions, rmse)
        """
        print("Evaluating model...")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predictions = predictions.flatten()
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_true, predictions))
        
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        
        return predictions, rmse
    
    def plot_results(self, y_true, predictions):
        """
        Plot comparison between true and predicted RUL values
        
        Args:
            y_true (numpy.array): True RUL values
            predictions (numpy.array): Predicted RUL values
        """
        print("Creating results visualization...")
        
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Predicted vs True values
        plt.subplot(1, 3, 1)
        plt.scatter(y_true, predictions, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('True RUL')
        plt.ylabel('Predicted RUL')
        plt.title('Predicted vs True RUL')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Time series comparison
        plt.subplot(1, 3, 2)
        plt.plot(y_true, label='True RUL', marker='o', markersize=3)
        plt.plot(predictions, label='Predicted RUL', marker='s', markersize=3)
        plt.xlabel('Engine Index')
        plt.ylabel('RUL')
        plt.title('RUL Comparison by Engine')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Error distribution
        plt.subplot(1, 3, 3)
        errors = predictions - y_true
        plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Prediction Error Distribution')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print additional statistics
        print(f"\nDetailed Performance Metrics:")
        print(f"Mean Absolute Error: {np.mean(np.abs(errors)):.2f}")
        print(f"Mean Error: {np.mean(errors):.2f}")
        print(f"Standard Deviation of Errors: {np.std(errors):.2f}")
        print(f"Max Error: {np.max(np.abs(errors)):.2f}")
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a pre-trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("Aircraft Engine RUL Prediction Pipeline")
    print("=" * 60)
    
    # Initialize predictor
    predictor = AircraftEngineRULPredictor(sequence_length=50)
    
    # Step 1: Load data
    train_df, test_df, true_rul = predictor.load_data(
        'CMaps/train_FD001.txt',
        'CMaps/test_FD001.txt', 
        'CMaps/RUL_FD001.txt'
    )
    
    # Step 2: Calculate RUL for training data
    train_df = predictor.calculate_rul(train_df)
    
    # Step 3: Normalize data
    train_normalized, test_normalized = predictor.normalize_data(train_df, test_df)
    
    # Step 4: Create sequences
    X_train, y_train = predictor.create_sequences(train_normalized)
    X_test = predictor.create_test_sequences(test_normalized)
    
    # Step 5: Build model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = predictor.build_model(input_shape)
    
    # Step 6: Train model
    history = predictor.train_model(X_train, y_train, epochs=100)
    
    # Step 7: Evaluate model
    y_true = true_rul['RUL'].values
    predictions, rmse = predictor.evaluate_model(X_test, y_true)
    
    # Step 8: Visualize results
    predictor.plot_results(y_true, predictions)
    
    # Save model
    predictor.save_model('aircraft_engine_rul_model.h5')
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print(f"Final RMSE: {rmse:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()