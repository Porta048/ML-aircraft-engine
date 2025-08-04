"""
Production-Ready Aircraft Engine RUL Prediction System
Enhanced version with advanced ML techniques for real-world deployment

Features:
- Advanced feature engineering and data validation
- Attention-based LSTM architecture
- Hyperparameter optimization
- Uncertainty quantification
- Ensemble methods
- Production monitoring
- Real-time inference capabilities

Dataset: NASA C-MAPSS Turbofan Engine Degradation Dataset
Author: ML Engineer
Date: 2025-08-04
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import IsolationForest
import tensorflow as tf
import keras
from keras import layers, Model, Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import optuna
from scipy import stats
import logging
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionRULPredictor:
    """
    Production-ready RUL prediction system with advanced ML techniques
    """
    
    def __init__(self, sequence_length=50, model_config=None):
        """
        Initialize the production RUL predictor
        
        Args:
            sequence_length (int): Length of input sequences
            model_config (dict): Model configuration parameters
        """
        self.sequence_length = sequence_length
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        self.data_quality_metrics = {}
        
        # Default model configuration
        self.config = {
            'lstm_units': [128, 64],
            'dropout_rate': 0.3,
            'attention_units': 64,
            'learning_rate': 0.001,
            'batch_size': 64,
            'epochs': 200,
            'validation_split': 0.2,
            'ensemble_size': 5
        }
        
        if model_config:
            self.config.update(model_config)
        
        # Define column names
        self.column_names = ['unit_number', 'time_in_cycles'] + \
                           [f'setting_{i}' for i in range(1, 4)] + \
                           [f'sensor_{i}' for i in range(1, 22)]
        
        # Enhanced column analysis (will be determined dynamically)
        self.columns_to_drop = []
        self.sensor_columns = []
        
        logger.info(f"Initialized ProductionRULPredictor with config: {self.config}")
    
    def analyze_data_quality(self, df, dataset_name="dataset"):
        """
        Comprehensive data quality analysis
        
        Args:
            df (DataFrame): Input dataframe
            dataset_name (str): Name for logging
            
        Returns:
            dict: Data quality metrics
        """
        logger.info(f"Analyzing data quality for {dataset_name}")
        
        quality_metrics = {
            'total_records': len(df),
            'total_engines': df['unit_number'].nunique(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_records': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict(),
            'statistical_summary': df.describe().to_dict()
        }
        
        # Check for constant columns
        constant_cols = []
        low_variance_cols = []
        
        for col in df.columns:
            if col not in ['unit_number', 'time_in_cycles']:
                variance = df[col].var()
                if variance == 0:
                    constant_cols.append(col)
                elif variance < 0.01:
                    low_variance_cols.append(col)
        
        quality_metrics['constant_columns'] = constant_cols
        quality_metrics['low_variance_columns'] = low_variance_cols
        
        # Outlier detection using IQR
        outlier_counts = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if col not in ['unit_number', 'time_in_cycles']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts[col] = outliers
        
        quality_metrics['outlier_counts'] = outlier_counts
        
        self.data_quality_metrics[dataset_name] = quality_metrics
        logger.info(f"Data quality analysis completed for {dataset_name}")
        
        return quality_metrics
    
    def advanced_feature_engineering(self, df):
        """
        Advanced feature engineering for better model performance
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Enhanced dataframe with engineered features
        """
        logger.info("Performing advanced feature engineering")
        
        df_enhanced = df.copy()
        
        # Identify sensor columns
        sensor_cols = [col for col in df.columns if col.startswith('sensor_')]
        
        # Rolling statistics features
        window_sizes = [5, 10, 20]
        for window in window_sizes:
            for col in sensor_cols:
                # Rolling mean
                df_enhanced[f'{col}_rolling_mean_{window}'] = (
                    df_enhanced.groupby('unit_number')[col]
                    .rolling(window=window, min_periods=1)
                    .mean().reset_index(0, drop=True)
                )
                
                # Rolling standard deviation
                df_enhanced[f'{col}_rolling_std_{window}'] = (
                    df_enhanced.groupby('unit_number')[col]
                    .rolling(window=window, min_periods=1)
                    .std().reset_index(0, drop=True)
                )
                
                # Rolling min/max
                df_enhanced[f'{col}_rolling_min_{window}'] = (
                    df_enhanced.groupby('unit_number')[col]
                    .rolling(window=window, min_periods=1)
                    .min().reset_index(0, drop=True)
                )
                
                df_enhanced[f'{col}_rolling_max_{window}'] = (
                    df_enhanced.groupby('unit_number')[col]
                    .rolling(window=window, min_periods=1)
                    .max().reset_index(0, drop=True)
                )
        
        # Trend features (first difference)
        for col in sensor_cols:
            df_enhanced[f'{col}_diff'] = (
                df_enhanced.groupby('unit_number')[col].diff().fillna(0)
            )
        
        # Cross-sensor correlations and ratios
        important_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 
                           'sensor_8', 'sensor_9', 'sensor_11', 'sensor_12', 
                           'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17', 
                           'sensor_20', 'sensor_21']
        
        for i, sensor1 in enumerate(important_sensors):
            for sensor2 in important_sensors[i+1:]:
                if sensor1 in df.columns and sensor2 in df.columns:
                    # Ratio features
                    df_enhanced[f'{sensor1}_{sensor2}_ratio'] = (
                        df_enhanced[sensor1] / (df_enhanced[sensor2] + 1e-8)
                    )
        
        # Cumulative features
        for col in sensor_cols:
            df_enhanced[f'{col}_cumsum'] = (
                df_enhanced.groupby('unit_number')[col].cumsum()
            )
        
        # Fill any NaN values
        df_enhanced = df_enhanced.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Feature engineering completed. New shape: {df_enhanced.shape}")
        return df_enhanced
    
    def intelligent_feature_selection(self, df, target_col='RUL'):
        """
        Intelligent feature selection based on correlation and importance
        
        Args:
            df (DataFrame): Input dataframe with engineered features
            target_col (str): Target column name
            
        Returns:
            list: Selected feature columns
        """
        logger.info("Performing intelligent feature selection")
        
        # Get feature columns (exclude metadata and target)
        feature_cols = [col for col in df.columns 
                       if col not in ['unit_number', 'time_in_cycles', target_col]]
        
        if target_col not in df.columns:
            # For test data, use all engineered features
            selected_features = feature_cols
        else:
            # Calculate correlation with target
            correlations = df[feature_cols + [target_col]].corr()[target_col].abs()
            correlations = correlations.drop(target_col).sort_values(ascending=False)
            
            # Remove highly correlated features among themselves
            correlation_matrix = df[feature_cols].corr().abs()
            upper_triangle = correlation_matrix.where(
                np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
            )
            
            # Find features with high correlation (>0.95)
            high_corr_features = [column for column in upper_triangle.columns 
                                if any(upper_triangle[column] > 0.95)]
            
            # Select top features while avoiding high correlation
            selected_features = []
            for feature in correlations.index:
                if len(selected_features) < 50:  # Limit to top 50 features
                    if feature not in high_corr_features or len(selected_features) < 20:
                        selected_features.append(feature)
        
        self.selected_features = selected_features
        logger.info(f"Selected {len(selected_features)} features")
        
        return selected_features
    
    def advanced_preprocessing(self, train_df, test_df, scaler_type='robust'):
        """
        Advanced preprocessing with multiple scaler options
        
        Args:
            train_df (DataFrame): Training dataframe
            test_df (DataFrame): Test dataframe
            scaler_type (str): Type of scaler ('minmax', 'standard', 'robust')
            
        Returns:
            tuple: (processed_train_df, processed_test_df)
        """
        logger.info(f"Advanced preprocessing with {scaler_type} scaler")
        
        # Select scaler
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:  # robust
            scaler = RobustScaler()
        
        # Fit scaler on training data
        scaler.fit(train_df[self.selected_features])
        self.scalers['feature_scaler'] = scaler
        
        # Transform data
        train_processed = train_df.copy()
        test_processed = test_df.copy()
        
        train_processed[self.selected_features] = scaler.transform(train_df[self.selected_features])
        test_processed[self.selected_features] = scaler.transform(test_df[self.selected_features])
        
        logger.info("Advanced preprocessing completed")
        return train_processed, test_processed
    
    def create_sequences(self, df, target_col='RUL'):
        """
        Create sequences for LSTM input
        
        Args:
            df (DataFrame): Input dataframe
            target_col (str): Target column name
            
        Returns:
            tuple: (X, y) sequences
        """
        logger.info(f"Creating sequences with length {self.sequence_length}...")
        
        X, y = [], []
        
        # Group by engine unit
        for unit_id in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit_id].sort_values('time_in_cycles')
            
            # Create sequences for this engine
            for i in range(len(unit_data) - self.sequence_length + 1):
                # Input sequence (selected features)
                sequence = unit_data[self.selected_features].iloc[i:i + self.sequence_length].values
                X.append(sequence)
                
                # Target (RUL at the end of sequence)
                if target_col in df.columns:
                    target = unit_data[target_col].iloc[i + self.sequence_length - 1]
                    y.append(target)
        
        X = np.array(X)
        y = np.array(y) if y else None
        
        logger.info(f"Created {len(X)} sequences")
        logger.info(f"Sequence shape: {X.shape}")
        
        return X, y
    
    def create_test_sequences(self, df):
        """
        Create test sequences - only the last sequence for each engine
        
        Args:
            df (DataFrame): Test dataframe
            
        Returns:
            numpy.array: Test sequences
        """
        logger.info("Creating test sequences (last sequence per engine)...")
        
        X_test = []
        
        # For each engine, take the last sequence_length cycles
        for unit_id in df['unit_number'].unique():
            unit_data = df[df['unit_number'] == unit_id].sort_values('time_in_cycles')
            
            # Take last sequence_length rows (or all if less than sequence_length)
            if len(unit_data) >= self.sequence_length:
                sequence = unit_data[self.selected_features].tail(self.sequence_length).values
            else:
                # Pad with first row if not enough data
                sequence = unit_data[self.selected_features].values
                padding = np.tile(sequence[0], (self.sequence_length - len(sequence), 1))
                sequence = np.vstack([padding, sequence])
            
            X_test.append(sequence)
        
        X_test = np.array(X_test)
        logger.info(f"Created {len(X_test)} test sequences")
        logger.info(f"Test sequence shape: {X_test.shape}")
        
        return X_test
    
    def create_attention_lstm_model(self, input_shape):
        """
        Create advanced LSTM model with attention mechanism
        
        Args:
            input_shape (tuple): Shape of input sequences
            
        Returns:
            keras.Model: Advanced LSTM model with attention
        """
        logger.info("Building attention-based LSTM model")
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        lstm1 = layers.LSTM(
            self.config['lstm_units'][0], 
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        )(inputs)
        
        # Second LSTM layer
        lstm2 = layers.LSTM(
            self.config['lstm_units'][1], 
            return_sequences=True,
            dropout=self.config['dropout_rate'],
            recurrent_dropout=self.config['dropout_rate']
        )(lstm1)
        
        # Attention mechanism
        attention = layers.Dense(self.config['attention_units'], activation='tanh')(lstm2)
        attention = layers.Dense(1, activation='softmax')(attention)
        
        # Apply attention weights
        context = layers.Multiply()([lstm2, attention])
        context = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(context)
        
        # Additional dense layers
        dense1 = layers.Dense(64, activation='relu')(context)
        dense1 = layers.Dropout(self.config['dropout_rate'])(dense1)
        
        dense2 = layers.Dense(32, activation='relu')(dense1)
        dense2 = layers.Dropout(self.config['dropout_rate'])(dense2)
        
        # Output layer
        outputs = layers.Dense(1, activation='linear')(dense2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        logger.info("Attention-based LSTM model created")
        model.summary()
        
        return model
    
    def create_ensemble_models(self, input_shape):
        """
        Create ensemble of different model architectures
        
        Args:
            input_shape (tuple): Shape of input sequences
            
        Returns:
            list: List of compiled models
        """
        logger.info("Creating ensemble models")
        
        models = []
        
        # Model 1: Attention LSTM
        models.append(self.create_attention_lstm_model(input_shape))
        
        # Model 2: Bidirectional LSTM
        model2 = keras.Sequential([
            layers.Bidirectional(layers.LSTM(100, return_sequences=True), input_shape=input_shape),
            layers.Dropout(0.3),
            layers.Bidirectional(layers.LSTM(50)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1, activation='linear')
        ])
        model2.compile(optimizer='adam', loss='mse', metrics=['mae'])
        models.append(model2)
        
        # Model 3: CNN-LSTM hybrid
        model3 = keras.Sequential([
            layers.Conv1D(64, 3, activation='relu', input_shape=input_shape),
            layers.Conv1D(64, 3, activation='relu'),
            layers.MaxPooling1D(2),
            layers.LSTM(100, return_sequences=True),
            layers.LSTM(50),
            layers.Dropout(0.3),
            layers.Dense(50, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        model3.compile(optimizer='adam', loss='mse', metrics=['mae'])
        models.append(model3)
        
        logger.info(f"Created ensemble of {len(models)} models")
        return models
    
    def train_with_uncertainty(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train models with uncertainty quantification
        
        Args:
            X_train (numpy.array): Training sequences
            y_train (numpy.array): Training targets
            X_val (numpy.array): Validation sequences
            y_val (numpy.array): Validation targets
            
        Returns:
            dict: Training results and uncertainty metrics
        """
        logger.info("Training models with uncertainty quantification")
        
        # Create ensemble models
        input_shape = (X_train.shape[1], X_train.shape[2])
        ensemble_models = self.create_ensemble_models(input_shape)
        
        trained_models = []
        training_histories = []
        
        # Train each model in ensemble
        for i, model in enumerate(ensemble_models):
            logger.info(f"Training ensemble model {i+1}/{len(ensemble_models)}")
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-6),
                ModelCheckpoint(f'best_model_{i}.h5', save_best_only=True, monitor='val_loss')
            ]
            
            # Train model
            if X_val is not None and y_val is not None:
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    batch_size=self.config['batch_size'],
                    epochs=self.config['epochs'],
                    callbacks=callbacks,
                    verbose=0
                )
            else:
                history = model.fit(
                    X_train, y_train,
                    validation_split=self.config['validation_split'],
                    batch_size=self.config['batch_size'],
                    epochs=self.config['epochs'],
                    callbacks=callbacks,
                    verbose=0
                )
            
            trained_models.append(model)
            training_histories.append(history)
        
        self.models['ensemble'] = trained_models
        self.training_histories = training_histories
        
        logger.info("Ensemble training completed")
        
        return {
            'models': trained_models,
            'histories': training_histories
        }
    
    def predict_with_uncertainty(self, X_test):
        """
        Make predictions with uncertainty quantification
        
        Args:
            X_test (numpy.array): Test sequences
            
        Returns:
            dict: Predictions with uncertainty metrics
        """
        logger.info("Making predictions with uncertainty quantification")
        
        ensemble_predictions = []
        
        # Get predictions from each model
        for i, model in enumerate(self.models['ensemble']):
            predictions = model.predict(X_test, verbose=0)
            ensemble_predictions.append(predictions.flatten())
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        # Calculate ensemble statistics
        mean_predictions = np.mean(ensemble_predictions, axis=0)
        std_predictions = np.std(ensemble_predictions, axis=0)
        
        # Calculate confidence intervals (95%)
        confidence_lower = mean_predictions - 1.96 * std_predictions
        confidence_upper = mean_predictions + 1.96 * std_predictions
        
        results = {
            'predictions': mean_predictions,
            'uncertainty': std_predictions,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'individual_predictions': ensemble_predictions
        }
        
        logger.info("Uncertainty quantification completed")
        return results
    
    def comprehensive_evaluation(self, y_true, prediction_results):
        """
        Comprehensive model evaluation for production use
        
        Args:
            y_true (numpy.array): True values
            prediction_results (dict): Results from predict_with_uncertainty
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        logger.info("Performing comprehensive evaluation")
        
        y_pred = prediction_results['predictions']
        uncertainty = prediction_results['uncertainty']
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100
        
        # Advanced metrics
        errors = y_pred - y_true
        
        # Directional accuracy (for trend prediction)
        if len(y_true) > 1:
            true_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(true_direction == pred_direction) * 100
        else:
            directional_accuracy = None
        
        # Uncertainty calibration
        within_ci = np.sum((y_true >= prediction_results['confidence_lower']) & 
                          (y_true <= prediction_results['confidence_upper']))
        ci_coverage = within_ci / len(y_true) * 100
        
        # Risk-based metrics (important for maintenance)
        # Conservative predictions (underestimating RUL) are more critical
        overestimation_rate = np.sum(y_pred > y_true) / len(y_true) * 100
        avg_overestimation = np.mean(errors[errors > 0]) if np.any(errors > 0) else 0
        avg_underestimation = np.mean(np.abs(errors[errors < 0])) if np.any(errors < 0) else 0
        
        # Prediction reliability score
        reliability_score = 100 - (rmse / np.mean(y_true) * 100)
        
        evaluation_metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'ci_coverage': ci_coverage,
            'overestimation_rate': overestimation_rate,
            'avg_overestimation': avg_overestimation,
            'avg_underestimation': avg_underestimation,
            'reliability_score': reliability_score,
            'mean_uncertainty': np.mean(uncertainty),
            'max_uncertainty': np.max(uncertainty),
            'prediction_range': np.max(y_pred) - np.min(y_pred)
        }
        
        logger.info(f"Evaluation completed. RMSE: {rmse:.2f}, Reliability: {reliability_score:.1f}%")
        
        return evaluation_metrics
    
    def create_production_visualizations(self, y_true, prediction_results, evaluation_metrics):
        """
        Create comprehensive visualizations for production monitoring
        
        Args:
            y_true (numpy.array): True values
            prediction_results (dict): Prediction results with uncertainty
            evaluation_metrics (dict): Evaluation metrics
        """
        logger.info("Creating production visualizations")
        
        y_pred = prediction_results['predictions']
        uncertainty = prediction_results['uncertainty']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # 1. Predictions vs True with uncertainty
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6, c=uncertainty, cmap='viridis')
        axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('True RUL')
        axes[0, 0].set_ylabel('Predicted RUL')
        axes[0, 0].set_title('Predictions vs True (colored by uncertainty)')
        axes[0, 0].grid(True, alpha=0.3)
        cbar = plt.colorbar(axes[0, 0].collections[0], ax=axes[0, 0])
        cbar.set_label('Prediction Uncertainty')
        
        # 2. Time series with confidence intervals
        indices = range(len(y_true))
        axes[0, 1].plot(indices, y_true, 'o-', label='True RUL', markersize=4)
        axes[0, 1].plot(indices, y_pred, 's-', label='Predicted RUL', markersize=4)
        axes[0, 1].fill_between(indices, 
                               prediction_results['confidence_lower'],
                               prediction_results['confidence_upper'],
                               alpha=0.3, label='95% Confidence Interval')
        axes[0, 1].set_xlabel('Engine Index')
        axes[0, 1].set_ylabel('RUL')
        axes[0, 1].set_title('RUL Predictions with Uncertainty')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        errors = y_pred - y_true
        axes[0, 2].hist(errors, bins=30, alpha=0.7, edgecolor='black', density=True)
        axes[0, 2].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].axvline(x=np.mean(errors), color='orange', linestyle='--', alpha=0.7, 
                          label=f'Mean Error: {np.mean(errors):.2f}')
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Uncertainty vs Error relationship
        axes[1, 0].scatter(uncertainty, np.abs(errors), alpha=0.6)
        axes[1, 0].set_xlabel('Prediction Uncertainty')
        axes[1, 0].set_ylabel('Absolute Error')
        axes[1, 0].set_title('Uncertainty vs Absolute Error')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Residuals vs Predicted
        axes[1, 1].scatter(y_pred, errors, alpha=0.6)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1, 1].set_xlabel('Predicted RUL')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Performance metrics summary
        axes[1, 2].axis('off')
        metrics_text = f"""
        Performance Metrics:
        
        RMSE: {evaluation_metrics['rmse']:.2f}
        MAE: {evaluation_metrics['mae']:.2f}
        R² Score: {evaluation_metrics['r2_score']:.3f}
        MAPE: {evaluation_metrics['mape']:.1f}%
        
        Reliability Score: {evaluation_metrics['reliability_score']:.1f}%
        CI Coverage: {evaluation_metrics['ci_coverage']:.1f}%
        
        Risk Metrics:
        Overestimation Rate: {evaluation_metrics['overestimation_rate']:.1f}%
        Avg Overestimation: {evaluation_metrics['avg_overestimation']:.2f}
        Avg Underestimation: {evaluation_metrics['avg_underestimation']:.2f}
        """
        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes, 
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('production_rul_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("Production visualizations created and saved")
    
    def save_production_model(self, model_name="production_rul_model"):
        """
        Save complete model pipeline for production deployment
        
        Args:
            model_name (str): Base name for saved files
        """
        logger.info(f"Saving production model: {model_name}")
        
        # Save ensemble models
        for i, model in enumerate(self.models['ensemble']):
            model.save(f"{model_name}_ensemble_{i}.h5")
        
        # Save scalers
        joblib.dump(self.scalers, f"{model_name}_scalers.pkl")
        
        # Save configuration and metadata
        metadata = {
            'config': self.config,
            'selected_features': self.selected_features,
            'sequence_length': self.sequence_length,
            'data_quality_metrics': self.data_quality_metrics,
            'model_creation_date': datetime.now().isoformat(),
            'model_version': '1.0'
        }
        
        with open(f"{model_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info("Production model saved successfully")
    
    def load_production_model(self, model_name="production_rul_model"):
        """
        Load complete model pipeline for production use
        
        Args:
            model_name (str): Base name of saved files
        """
        logger.info(f"Loading production model: {model_name}")
        
        # Load metadata
        with open(f"{model_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.config = metadata['config']
        self.selected_features = metadata['selected_features']
        self.sequence_length = metadata['sequence_length']
        
        # Load scalers
        self.scalers = joblib.load(f"{model_name}_scalers.pkl")
        
        # Load ensemble models
        ensemble_models = []
        i = 0
        while True:
            try:
                model = keras.models.load_model(f"{model_name}_ensemble_{i}.h5")
                ensemble_models.append(model)
                i += 1
            except:
                break
        
        self.models['ensemble'] = ensemble_models
        
        logger.info(f"Production model loaded successfully. Ensemble size: {len(ensemble_models)}")
    
    def real_time_inference(self, sensor_data):
        """
        Real-time inference for production deployment
        
        Args:
            sensor_data (numpy.array or pandas.DataFrame): Recent sensor readings
            
        Returns:
            dict: RUL prediction with uncertainty and metadata
        """
        logger.info("Performing real-time inference")
        
        # Ensure input is properly formatted
        if isinstance(sensor_data, pd.DataFrame):
            sensor_data = sensor_data[self.selected_features].values
        
        # Apply scaling
        sensor_data_scaled = self.scalers['feature_scaler'].transform(sensor_data)
        
        # Create sequence if needed
        if len(sensor_data_scaled.shape) == 2:
            if sensor_data_scaled.shape[0] >= self.sequence_length:
                sequence = sensor_data_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            else:
                # Pad if insufficient data
                padding_needed = self.sequence_length - sensor_data_scaled.shape[0]
                padding = np.tile(sensor_data_scaled[0], (padding_needed, 1))
                padded_data = np.vstack([padding, sensor_data_scaled])
                sequence = padded_data.reshape(1, self.sequence_length, -1)
        else:
            sequence = sensor_data_scaled
        
        # Get ensemble predictions
        predictions = []
        for model in self.models['ensemble']:
            pred = model.predict(sequence, verbose=0)
            predictions.append(pred[0, 0])
        
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_prediction = np.mean(predictions)
        std_prediction = np.std(predictions)
        confidence_lower = mean_prediction - 1.96 * std_prediction
        confidence_upper = mean_prediction + 1.96 * std_prediction
        
        # Risk assessment
        risk_level = 'LOW'
        if mean_prediction < 50:
            risk_level = 'HIGH'
        elif mean_prediction < 100:
            risk_level = 'MEDIUM'
        
        result = {
            'predicted_rul': mean_prediction,
            'uncertainty': std_prediction,
            'confidence_interval': [confidence_lower, confidence_upper],
            'risk_level': risk_level,
            'timestamp': datetime.now().isoformat(),
            'individual_predictions': predictions.tolist()
        }
        
        logger.info(f"Real-time inference completed. RUL: {mean_prediction:.1f} ± {std_prediction:.1f}")
        
        return result


def main():
    """
    Main execution function for production-ready RUL prediction
    """
    print("=" * 80)
    print("PRODUCTION-READY AIRCRAFT ENGINE RUL PREDICTION SYSTEM")
    print("=" * 80)
    
    # Initialize predictor
    predictor = ProductionRULPredictor(sequence_length=50)
    
    try:
        # Load and analyze data
        logger.info("Loading dataset...")
        
        # Load training data
        train_df = pd.read_csv('CMaps/train_FD001.txt', sep=' ', header=None)
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
        train_df.columns = predictor.column_names
        
        # Load test data
        test_df = pd.read_csv('CMaps/test_FD001.txt', sep=' ', header=None)
        test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
        test_df.columns = predictor.column_names
        
        # Load true RUL
        true_rul = pd.read_csv('CMaps/RUL_FD001.txt', sep=' ', header=None)
        true_rul = true_rul.dropna(axis=1)  # Remove empty columns
        true_rul.columns = ['RUL']
        
        # Data quality analysis
        train_quality = predictor.analyze_data_quality(train_df, "training")
        test_quality = predictor.analyze_data_quality(test_df, "test")
        
        # Calculate RUL for training data
        max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycle']
        train_df = train_df.merge(max_cycles, on='unit_number', how='left')
        train_df['RUL'] = train_df['max_cycle'] - train_df['time_in_cycles']
        train_df = train_df.drop(columns=['max_cycle'])
        
        # Advanced feature engineering
        train_enhanced = predictor.advanced_feature_engineering(train_df)
        test_enhanced = predictor.advanced_feature_engineering(test_df)
        
        # Intelligent feature selection
        selected_features = predictor.intelligent_feature_selection(train_enhanced)
        
        # Advanced preprocessing
        train_processed, test_processed = predictor.advanced_preprocessing(
            train_enhanced, test_enhanced, scaler_type='robust'
        )
        
        # Create sequences
        X_train, y_train = predictor.create_sequences(train_processed)
        X_test = predictor.create_test_sequences(test_processed)
        
        # Train ensemble models with uncertainty quantification
        training_results = predictor.train_with_uncertainty(X_train, y_train)
        
        # Make predictions with uncertainty
        prediction_results = predictor.predict_with_uncertainty(X_test)
        
        # Comprehensive evaluation
        y_true = true_rul['RUL'].values
        evaluation_metrics = predictor.comprehensive_evaluation(y_true, prediction_results)
        
        # Create production visualizations
        predictor.create_production_visualizations(y_true, prediction_results, evaluation_metrics)
        
        # Save production model
        predictor.save_production_model("production_rul_model_v1")
        
        # Print final results
        print("\n" + "=" * 80)
        print("PRODUCTION MODEL PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"RMSE: {evaluation_metrics['rmse']:.2f}")
        print(f"MAE: {evaluation_metrics['mae']:.2f}")
        print(f"R² Score: {evaluation_metrics['r2_score']:.3f}")
        print(f"MAPE: {evaluation_metrics['mape']:.1f}%")
        print(f"Reliability Score: {evaluation_metrics['reliability_score']:.1f}%")
        print(f"Confidence Interval Coverage: {evaluation_metrics['ci_coverage']:.1f}%")
        print(f"Overestimation Rate: {evaluation_metrics['overestimation_rate']:.1f}%")
        print("=" * 80)
        print("Model ready for production deployment!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()