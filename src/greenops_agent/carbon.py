"""
Carbon intensity module for GreenOps Agent.

This module is responsible for fetching carbon intensity data from external APIs
such as Electricity Maps. Carbon intensity is used to calculate the environmental
impact of running workloads.
"""

import logging
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class CarbonIntensityFetcher:
    """
    Fetches carbon intensity data from various APIs.
    
    Attributes:
        api_key: API key for the carbon intensity service
        region: Region code for which to fetch carbon intensity
        base_url: Base URL for the API
        cache_duration: How long to cache results in seconds
    """
    
    def __init__(
        self,
        api_key: str,
        region: str,
        base_url: str = "https://api.electricitymap.org/v3",
        cache_duration: int = 300  # 5 minutes
    ):
        """
        Initialize the carbon intensity fetcher.
        
        Args:
            api_key: API key for the carbon intensity service
            region: Region code for which to fetch carbon intensity
            base_url: Base URL for the API
            cache_duration: How long to cache results in seconds
        """
        self.api_key = api_key
        self.region = region
        self.base_url = base_url
        self.cache_duration = cache_duration
        self._last_fetch_time = 0
        self._cached_intensity = None
        self._cached_forecast = None
        
        logger.info(f"Initialized CarbonIntensityFetcher for region: {region}")
    
    def get_current_intensity(self, force_refresh: bool = False) -> Optional[float]:
        """
        Get the current carbon intensity for the configured region.
        
        Args:
            force_refresh: If True, ignore cache and fetch fresh data
            
        Returns:
            Carbon intensity in gCO2eq/kWh, or None on failure
        """
        current_time = time.time()
        
        # Return cached value if valid and not forcing refresh
        if not force_refresh and self._cached_intensity and \
           (current_time - self._last_fetch_time) < self.cache_duration:
            return self._cached_intensity
        
        try:
            headers = {
                "auth-token": self.api_key
            }
            
            response = requests.get(
                f"{self.base_url}/carbon-intensity/latest",
                params={
                    "zone": self.region
                },
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract the carbon intensity value
            intensity = data.get("carbonIntensity", None)
            
            # Update cache
            if intensity is not None:
                self._cached_intensity = float(intensity)
                self._last_fetch_time = current_time
            
            return self._cached_intensity
            
        except requests.RequestException as e:
            logger.error(f"Error fetching carbon intensity: {e}")
            return self._cached_intensity  # Return old cached value on error
    
    def get_intensity_forecast(self) -> List[Tuple[datetime, float]]:
        """
        Get carbon intensity forecast for the next hours.
        
        Returns:
            List of (datetime, intensity) tuples, or empty list on failure
        """
        try:
            headers = {
                "auth-token": self.api_key
            }
            
            response = requests.get(
                f"{self.base_url}/carbon-intensity/forecast",
                params={
                    "zone": self.region
                },
                headers=headers,
                timeout=10
            )
            
            response.raise_for_status()
            data = response.json()
            
            forecast = []
            for point in data.get("forecast", []):
                timestamp = datetime.fromisoformat(point.get("datetime", "").replace("Z", "+00:00"))
                value = float(point.get("carbonIntensity", 0))
                forecast.append((timestamp, value))
            
            # Cache forecast data
            self._cached_forecast = forecast
            return forecast
            
        except (requests.RequestException, ValueError) as e:
            logger.error(f"Error fetching carbon intensity forecast: {e}")
            return self._cached_forecast or []
    
    def get_best_time_window(self, hours_ahead: int = 24, window_size: int = 1) -> Tuple[datetime, float]:
        """
        Find the time window with the lowest average carbon intensity.
        
        Args:
            hours_ahead: How many hours to look ahead in the forecast
            window_size: Size of the window in hours
            
        Returns:
            Tuple of (start_time, average_intensity) for the best window
        """
        forecast = self.get_intensity_forecast()
        
        if not forecast:
            return (datetime.now(), 0)
        
        # Filter forecast to requested hours ahead
        now = datetime.now()
        forecast = [(dt, val) for dt, val in forecast if dt <= now + timedelta(hours=hours_ahead)]
        
        if len(forecast) < window_size:
            # Not enough data for a full window
            return (now, self._cached_intensity or 0)
        
        # Find window with lowest average
        best_start_idx = 0
        lowest_avg = float('inf')
        
        for i in range(len(forecast) - window_size):
            window = forecast[i:i+window_size]
            avg = sum(val for _, val in window) / window_size
            
            if avg < lowest_avg:
                lowest_avg = avg
                best_start_idx = i
        
        return (forecast[best_start_idx][0], lowest_avg)
    
    def is_current_intensity_high(self, threshold_percentile: float = 75) -> bool:
        """
        Determine if current carbon intensity is considered high.
        
        Args:
            threshold_percentile: Percentile threshold for "high" (0-100)
            
        Returns:
            True if current intensity is above the threshold compared to forecast
        """
        current = self.get_current_intensity()
        forecast = self.get_intensity_forecast()
        
        if current is None or not forecast:
            return False
        
        # Extract all intensity values from forecast
        forecast_values = [val for _, val in forecast]
        
        if not forecast_values:
            return False
            
        # Calculate the threshold value
        forecast_values.sort()
        threshold_idx = int(len(forecast_values) * threshold_percentile / 100)
        threshold_value = forecast_values[min(threshold_idx, len(forecast_values)-1)]
        
        return current > threshold_value
    
    def get_carbon_data(self) -> Dict[str, Any]:
        """
        Get a comprehensive carbon intensity report.
        
        Returns:
            Dictionary with current intensity, forecast, and analysis
        """
        try:
            current = self.get_current_intensity()
            forecast = self.get_intensity_forecast()
            
            # Find best time window in the next 24 hours
            best_time, best_intensity = self.get_best_time_window(hours_ahead=24, window_size=1)
            
            # Determine if current intensity is high
            is_high = self.is_current_intensity_high()
            
            return {
                'current_intensity': current,
                'unit': 'gCO2eq/kWh',
                'region': self.region,
                'timestamp': datetime.now().isoformat(),
                'forecast': [(dt.isoformat(), value) for dt, value in forecast],
                'analysis': {
                    'is_high_carbon_period': is_high,
                    'best_time_window': {
                        'start_time': best_time.isoformat(),
                        'intensity': best_intensity
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error generating carbon data report: {e}")
            return {
                'error': str(e),
                'current_intensity': self._cached_intensity,
                'region': self.region,
                'timestamp': datetime.now().isoformat()
            }


class FallbackCarbonIntensityFetcher(CarbonIntensityFetcher):
    """
    A fallback carbon intensity fetcher that uses static data.
    
    This is useful when no API key is available or the service is down.
    """
    
    def __init__(self, region: str = "DEFAULT"):
        """
        Initialize the fallback carbon intensity fetcher.
        
        Args:
            region: Region code (mostly for logging)
        """
        # No call to super().__init__() as we don't need API key
        self.region = region
        self._static_intensity = {
            "US-CA": 200.0,  # California
            "US-TX": 450.0,  # Texas
            "DE": 350.0,     # Germany
            "FR": 80.0,      # France (nuclear heavy)
            "GB": 250.0,     # Great Britain
            "DEFAULT": 300.0  # Default fallback value
        }
        logger.info(f"Initialized FallbackCarbonIntensityFetcher for region: {region}")
    
    def get_current_intensity(self, force_refresh: bool = False) -> float:
        """
        Get a static carbon intensity value for the region.
        
        Args:
            force_refresh: Ignored in fallback implementation
            
        Returns:
            Carbon intensity in gCO2eq/kWh
        """
        return self._static_intensity.get(self.region, self._static_intensity["DEFAULT"])
    
    def get_intensity_forecast(self) -> List[Tuple[datetime, float]]:
        """
        Generate a fake carbon intensity forecast.
        
        Returns:
            List of (datetime, intensity) tuples
        """
        now = datetime.now()
        base_value = self.get_current_intensity()
        
        # Generate 24 hourly points with some variation
        forecast = []
        for hour in range(24):
            # Add some sinusoidal variation (±20%) to simulate daily pattern
            variation = 0.2 * base_value * (0.5 + 0.5 * ((hour % 12) / 12))
            if hour >= 12:  # Lower at night, higher during day
                value = base_value - variation
            else:
                value = base_value + variation
            
            timestamp = now + timedelta(hours=hour)
            forecast.append((timestamp, value))
        
        return forecast


class MockCarbonIntensityFetcher(CarbonIntensityFetcher):
    """
    A mock carbon intensity fetcher using the mockdata module.
    
    This is useful for testing and development without external API calls.
    """
    
    def __init__(self, region: str = "DEFAULT"):
        """
        Initialize the mock carbon intensity fetcher.
        
        Args:
            region: Region code for carbon intensity data
        """
        # No call to super().__init__() as we don't need API key
        self.region = region
        
        # Import here to avoid circular imports
        from .mockdata.carbon import create_mock_carbon_client
        self._mock_client = create_mock_carbon_client(region=region)
        
        logger.info(f"Initialized MockCarbonIntensityFetcher for region: {region}")
    
    def get_current_intensity(self, force_refresh: bool = False) -> float:
        """
        Get mock carbon intensity value for the region.
        
        Args:
            force_refresh: Ignored in mock implementation
            
        Returns:
            Carbon intensity in gCO2eq/kWh
        """
        data = self._mock_client.get_current_intensity()
        return data["carbon_intensity"]
    
    def get_intensity_forecast(self) -> List[Tuple[datetime, float]]:
        """
        Generate a mock carbon intensity forecast.
        
        Returns:
            List of (datetime, intensity) tuples
        """
        data = self._mock_client.get_forecast(hours=24)
        forecast = []
        
        for ts_str, value in data["forecast"]:
            ts = datetime.fromisoformat(ts_str)
            forecast.append((ts, value))
        
        return forecast
    
    def get_carbon_data(self) -> Dict[str, Any]:
        """
        Get a comprehensive mock carbon intensity report.
        
        Returns:
            Dictionary with current intensity, forecast, and analysis
        """
        data = self._mock_client.get_forecast(hours=24)
        
        # Transform the forecast data to match the format expected by clients
        forecast = [(ts_str, value) for ts_str, value in data["forecast"]]
        
        return {
            'current_intensity': data["current_intensity"],
            'unit': data["unit"],
            'region': self.region,
            'timestamp': datetime.now().isoformat(),
            'forecast': forecast,
            'analysis': data["analysis"]
        }


def get_carbon_fetcher(api_key: str, region: str, use_mock: bool = False) -> CarbonIntensityFetcher:
    """
    Factory function to get appropriate carbon intensity fetcher.
    
    Args:
        api_key: API key for carbon intensity service
        region: Region code
        use_mock: Whether to use mock data instead of real API
        
    Returns:
        A carbon intensity fetcher instance
    """
    if use_mock:
        logger.info("Using mock carbon intensity data")
        return MockCarbonIntensityFetcher(region)
    elif not api_key or api_key.lower() in ('none', 'fallback', ''):
        logger.warning("No API key provided, using fallback carbon intensity data")
        return FallbackCarbonIntensityFetcher(region)
    else:
        return CarbonIntensityFetcher(api_key, region)