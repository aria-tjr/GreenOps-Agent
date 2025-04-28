"""
Workload analysis and spike prediction module for GreenOps Agent.

This module uses machine learning (LSTM) to predict future resource usage
patterns based on historical data, enabling proactive scaling and resource
allocation decisions.
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Use TensorFlow for LSTM implementation if available
TF_AVAILABLE = False
SKLEARN_AVAILABLE = False

# Try to import TensorFlow, but don't output warning at module level
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    # We'll log this only when prediction is actually attempted
    pass

# Scikit-learn for preprocessing if available
try:
    from sklearn.preprocessing import MinMaxScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    # We'll log this only when prediction is actually attempted
    pass

logger = logging.getLogger(__name__)


class WorkloadPredictor:
    """
    Predicts future resource usage based on historical data.
    
    This class implements LSTM-based time series prediction for workload
    forecasting, with fallbacks to simpler methods when dependencies
    are not available or data is insufficient.
    
    Attributes:
        model: LSTM model for prediction
        scaler: Data scaler for normalization
        sequence_length: Number of time steps to use for prediction
        use_lstm: Whether to use LSTM or simple forecasting
    """
    
    def __init__(self, sequence_length: int = 12):
        """
        Initialize the workload predictor.
        
        Args:
            sequence_length: Number of historical points used for prediction
        """
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = None
        self.use_lstm = TF_AVAILABLE and SKLEARN_AVAILABLE
        
        if self.use_lstm:
            logger.info("Initialized WorkloadPredictor with LSTM model")
        else:
            logger.info("Initialized WorkloadPredictor with simple forecasting (LSTM unavailable)")
    
    def _create_model(self, input_shape: Tuple[int, int]) -> Any:
        """
        Create and compile the LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, features)
            
        Returns:
            Compiled Keras Sequential model
        """
        if not TF_AVAILABLE:
            logger.error("Cannot create LSTM model: TensorFlow not available")
            return None
            
        model = Sequential()
        # LSTM layer with 50 units, returning sequences
        model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
        # Second LSTM layer, not returning sequences
        model.add(LSTM(50))
        # Output layer - predicting next value
        model.add(Dense(1))
        
        # Compile model with mean squared error loss and Adam optimizer
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        return model
    
    def _prepare_data(
        self, 
        data: List[Tuple[float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, Optional[MinMaxScaler]]:
        """
        Prepare time series data for LSTM model.
        
        Args:
            data: List of (timestamp, value) tuples
            
        Returns:
            Tuple of (X, y, scaler) with prepared data and scaler
        """
        # Extract values from data tuples
        values = np.array([value for _, value in data]).reshape(-1, 1)
        
        # Scale data
        if SKLEARN_AVAILABLE:
            scaler = MinMaxScaler(feature_range=(0, 1))
            values_scaled = scaler.fit_transform(values)
        else:
            # Simple min-max scaling
            min_val = np.min(values)
            max_val = np.max(values)
            values_scaled = (values - min_val) / (max_val - min_val + 1e-10)
            scaler = None
        
        # Prepare sequences
        X = []
        y = []
        
        for i in range(len(values_scaled) - self.sequence_length):
            X.append(values_scaled[i:i + self.sequence_length])
            y.append(values_scaled[i + self.sequence_length])
        
        return np.array(X), np.array(y), scaler
    
    def train(self, data: List[Tuple[float, float]], epochs: int = 100) -> Dict[str, Any]:
        """
        Train the LSTM model with historical data.
        
        Args:
            data: List of (timestamp, value) tuples
            epochs: Number of training epochs
            
        Returns:
            Dictionary with training results
        """
        if not self.use_lstm or len(data) < self.sequence_length + 10:
            logger.warning(
                "Not enough data for LSTM or LSTM unavailable. "
                f"Need at least {self.sequence_length + 10} data points, got {len(data)}"
            )
            return {"status": "error", "message": "Insufficient data for training"}
        
        try:
            # Prepare data
            X, y, scaler = self._prepare_data(data)
            
            # Create and train model
            self.model = self._create_model((X.shape[1], X.shape[2]))
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=32,
                verbose=0,
                validation_split=0.2
            )
            
            self.scaler = scaler
            
            return {
                "status": "success",
                "message": f"Model trained for {epochs} epochs",
                "loss": history.history['loss'][-1],
                "val_loss": history.history.get('val_loss', [None])[-1]
            }
        
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return {"status": "error", "message": str(e)}
    
    def predict(
        self, 
        data: List[Tuple[float, float]], 
        steps_ahead: int = 6
    ) -> List[Tuple[datetime, float]]:
        """
        Predict future values using the trained model.
        
        Args:
            data: Recent time series data as (timestamp, value) tuples
            steps_ahead: Number of future steps to predict
            
        Returns:
            List of (timestamp, predicted_value) tuples
        """
        if len(data) < self.sequence_length:
            logger.error(f"Insufficient data for prediction. Need {self.sequence_length} points.")
            return []
        
        # Get timestamps and values
        timestamps = [ts for ts, _ in data]
        
        if self.use_lstm and self.model and self.scaler:
            # Use LSTM for prediction
            return self._predict_lstm(data, steps_ahead)
        else:
            # Use simple forecasting method as fallback
            return self._predict_simple(data, steps_ahead)
    
    def _predict_lstm(
        self, 
        data: List[Tuple[float, float]], 
        steps_ahead: int
    ) -> List[Tuple[datetime, float]]:
        """
        Make predictions using the LSTM model.
        
        Args:
            data: Recent time series data as (timestamp, value) tuples
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            List of (timestamp, predicted_value) tuples
        """
        # Extract timestamps and values
        timestamps = [dt for dt, _ in data]
        values = np.array([value for _, value in data]).reshape(-1, 1)
        
        # Scale input data
        values_scaled = self.scaler.transform(values)
        
        # Get the most recent sequence for prediction
        input_sequence = values_scaled[-self.sequence_length:].reshape(1, self.sequence_length, 1)
        
        predictions = []
        last_sequence = input_sequence.copy()
        
        # Calculate interval between timestamps for future projection
        if isinstance(timestamps[0], (int, float)):
            # Timestamps are numeric (unix time)
            avg_interval = sum(timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)) / (len(timestamps)-1)
            next_timestamp = timestamps[-1] + avg_interval
        else:
            # Timestamps are datetime objects
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            next_timestamp = timestamps[-1] + timedelta(seconds=avg_interval)
        
        # Generate predictions for each step ahead
        for _ in range(steps_ahead):
            # Predict next value
            next_pred_scaled = self.model.predict(last_sequence, verbose=0)
            
            # Inverse transform to get actual value
            next_pred = self.scaler.inverse_transform(next_pred_scaled)[0, 0]
            
            # Add to predictions
            predictions.append((next_timestamp, next_pred))
            
            # Update input sequence for next prediction (sliding window)
            last_sequence = np.roll(last_sequence, -1, axis=1)
            last_sequence[0, -1, 0] = next_pred_scaled[0, 0]
            
            # Calculate next timestamp
            if isinstance(next_timestamp, (int, float)):
                next_timestamp = next_timestamp + avg_interval
            else:
                next_timestamp = next_timestamp + timedelta(seconds=avg_interval)
        
        return predictions
    
    def _predict_simple(
        self, 
        data: List[Tuple[float, float]], 
        steps_ahead: int
    ) -> List[Tuple[datetime, float]]:
        """
        Make predictions using simple forecasting methods.
        
        This is used as a fallback when LSTM is not available.
        
        Args:
            data: Recent time series data as (timestamp, value) tuples
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            List of (timestamp, predicted_value) tuples
        """
        # Extract timestamps and values
        timestamps = [dt for dt, _ in data]
        values = [value for _, value in data]
        
        # Calculate interval between timestamps for future projection
        if isinstance(timestamps[0], (int, float)):
            # Timestamps are numeric (unix time)
            avg_interval = sum(timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)) / (len(timestamps)-1)
            next_timestamp = timestamps[-1] + avg_interval
        else:
            # Timestamps are datetime objects
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals)
            next_timestamp = timestamps[-1] + timedelta(seconds=avg_interval)
        
        # Use simple exponential weighted moving average for prediction
        alpha = 0.3  # Smoothing factor
        last_value = values[-1]
        
        # Use trend if there's enough data
        if len(values) > 5:
            # Simple linear trend calculation
            window = min(12, len(values))
            recent_values = values[-window:]
            trend = (recent_values[-1] - recent_values[0]) / (window - 1)
        else:
            trend = 0
        
        predictions = []
        
        # Generate predictions for each step ahead
        for i in range(steps_ahead):
            # Predict with trend
            next_pred = last_value + trend
            
            # Add prediction
            predictions.append((next_timestamp, next_pred))
            
            # Update for next step
            last_value = next_pred
            
            # Update next timestamp
            if isinstance(next_timestamp, (int, float)):
                next_timestamp = next_timestamp + avg_interval
            else:
                next_timestamp = next_timestamp + timedelta(seconds=avg_interval)
        
        return predictions
    
    def detect_anomalies(
        self, 
        actual: List[Tuple[float, float]], 
        predictions: List[Tuple[float, float]], 
        threshold: float = 0.25
    ) -> List[Tuple[datetime, float, float]]:
        """
        Detect anomalies by comparing actual vs predicted values.
        
        Args:
            actual: List of (timestamp, actual_value) tuples
            predictions: List of (timestamp, predicted_value) tuples
            threshold: Relative threshold for anomaly detection (fraction of value)
            
        Returns:
            List of (timestamp, actual, predicted) tuples where anomalies were detected
        """
        anomalies = []
        
        for i, ((act_ts, act_val), (pred_ts, pred_val)) in enumerate(zip(actual, predictions)):
            # Calculate relative error
            if abs(pred_val) > 1e-10:  # Avoid division by zero
                rel_error = abs(act_val - pred_val) / pred_val
            else:
                rel_error = abs(act_val - pred_val)
                
            # Check if error exceeds threshold
            if rel_error > threshold:
                anomalies.append((act_ts, act_val, pred_val))
        
        return anomalies
    
    def analyze_workload(self, data: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Analyze workload patterns and predict future usage.
        
        Args:
            data: Historical time series data as (timestamp, value) tuples
            
        Returns:
            Dictionary with analysis results and predictions
        """
        if len(data) < max(self.sequence_length, 6):
            return {
                "status": "error",
                "message": f"Insufficient data for analysis. Need at least {max(self.sequence_length, 6)} points."
            }
        
        try:
            # Train model if using LSTM
            if self.use_lstm:
                train_result = self.train(data)
                if train_result["status"] != "success":
                    logger.warning(f"LSTM training failed: {train_result['message']}")
            
            # Get recent data for prediction
            recent_data = data[-min(2*self.sequence_length, len(data)):]
            
            # Predict next 6 steps
            predictions = self.predict(recent_data, steps_ahead=6)
            
            # Calculate basic statistics
            values = [v for _, v in data]
            current = values[-1]
            avg = sum(values) / len(values)
            peak = max(values)
            
            # Check for predicted spike
            spike_detected = False
            spike_pct = 0
            
            if predictions:
                max_prediction = max(v for _, v in predictions)
                spike_pct = ((max_prediction / current) - 1) * 100
                spike_detected = spike_pct > 20  # 20% increase is considered a spike
            
            # Return analysis results
            return {
                "status": "success",
                "current_value": current,
                "average_value": avg,
                "peak_value": peak,
                "predictions": predictions,
                "analysis": {
                    "spike_detected": spike_detected,
                    "spike_percentage": spike_pct,
                    "trend": "increasing" if spike_pct > 5 else "decreasing" if spike_pct < -5 else "stable"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in workload analysis: {e}")
            return {
                "status": "error",
                "message": f"Analysis failed: {str(e)}"
            }