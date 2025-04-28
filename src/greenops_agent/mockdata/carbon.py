"""
Mock carbon intensity data module for testing and development.

This module provides mock carbon intensity data that simulates
real-world carbon intensity patterns without requiring API access.
"""

import random
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple


class MockCarbonClient:
    """
    Mock carbon intensity client that provides simulated data.
    
    Attributes:
        region: Region code for carbon intensity data
        base_intensity: Base carbon intensity value for this region
        variation_factor: How much the intensity varies (as a percentage)
    """
    
    def __init__(self, region: str = "DEFAULT"):
        """
        Initialize the mock carbon client.
        
        Args:
            region: Region code to simulate data for
        """
        self.region = region
        
        # Default base intensities for different regions (gCO2eq/kWh)
        region_intensities = {
            "US-CA": 200.0,    # California
            "US-TX": 450.0,    # Texas
            "DE": 350.0,       # Germany
            "FR": 80.0,        # France (nuclear heavy)
            "GB": 250.0,       # Great Britain
            "AU-NSW": 600.0,   # Australia NSW
            "IN": 700.0,       # India
            "CN": 650.0,       # China
            "SE": 30.0,        # Sweden (renewable heavy)
            "DEFAULT": 300.0   # Default fallback value
        }
        
        # Get base intensity for this region
        self.base_intensity = region_intensities.get(region, region_intensities["DEFAULT"])
        
        # Variation factor - how much the intensity varies throughout the day
        self.variation_factor = 0.3  # 30% variation
        
    def get_current_intensity(self) -> Dict[str, Any]:
        """
        Get current simulated carbon intensity.
        
        Returns:
            Dictionary with carbon intensity data
        """
        now = datetime.now()
        hour = now.hour
        
        # Generate a value based on time of day
        # Lower at night, higher during day, with peak around noon
        time_factor = math.sin(math.pi * hour / 12) if hour < 12 else math.sin(math.pi * (24 - hour) / 12)
        
        # Add some randomness (±10%)
        random_factor = 1.0 + (random.random() * 0.2 - 0.1)
        
        # Calculate the intensity value
        intensity = self.base_intensity * (1 + self.variation_factor * time_factor) * random_factor
        
        return {
            "carbon_intensity": intensity,
            "unit": "gCO2eq/kWh",
            "region": self.region,
            "timestamp": now.isoformat()
        }
    
    def get_forecast(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a forecast of carbon intensity for the next hours.
        
        Args:
            hours: Number of hours to forecast
            
        Returns:
            Dictionary with forecast data
        """
        now = datetime.now()
        current_data = self.get_current_intensity()
        current_intensity = current_data["carbon_intensity"]
        
        forecast = []
        for hour_offset in range(hours):
            timestamp = now + timedelta(hours=hour_offset)
            hour = timestamp.hour
            
            # Generate a value based on time of day with some continuity
            # Lower at night, higher during day, with peak around noon
            time_factor = math.sin(math.pi * hour / 12) if hour < 12 else math.sin(math.pi * (24 - hour) / 12)
            
            # Add some randomness (±15%) but maintain trend line
            random_factor = 1.0 + (random.random() * 0.3 - 0.15)
            
            # Calculate the intensity value
            intensity = self.base_intensity * (1 + self.variation_factor * time_factor) * random_factor
            
            # Ensure there's reasonable continuity between hours
            if hour_offset > 0:
                last_intensity = forecast[-1][1]
                # Smooth transitions (max 10% change between hours)
                max_change = last_intensity * 0.1
                if abs(intensity - last_intensity) > max_change:
                    direction = 1 if intensity > last_intensity else -1
                    intensity = last_intensity + direction * max_change
            
            forecast.append((timestamp.isoformat(), intensity))
        
        # Find best time window (lowest intensity)
        forecast_values = [value for _, value in forecast]
        min_idx = forecast_values.index(min(forecast_values))
        best_time = forecast[min_idx][0]
        best_intensity = forecast_values[min_idx]
        
        # Determine if current is a high carbon period (top 25%)
        forecast_values.sort()
        threshold_idx = int(len(forecast_values) * 0.75)
        threshold_value = forecast_values[min(threshold_idx, len(forecast_values)-1)]
        is_high_carbon = current_intensity > threshold_value
        
        return {
            "current_intensity": current_intensity,
            "unit": "gCO2eq/kWh",
            "region": self.region,
            "timestamp": now.isoformat(),
            "forecast": forecast,
            "analysis": {
                "is_high_carbon_period": is_high_carbon,
                "best_time_window": {
                    "start_time": best_time,
                    "intensity": best_intensity
                }
            }
        }


def create_mock_carbon_client(region: str = "DEFAULT") -> MockCarbonClient:
    """
    Create a mock carbon client for a specific region.
    
    Args:
        region: Region code to simulate data for
        
    Returns:
        A configured mock carbon client
    """
    return MockCarbonClient(region)