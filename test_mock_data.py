#!/usr/bin/env python

import sys
import json
from datetime import datetime

# Add the src directory to the path to ensure we can import the modules
sys.path.insert(0, ".")

try:
    from src.greenops_agent.mockdata.prometheus import create_mock_prometheus_client
    from src.greenops_agent.mockdata.carbon import create_mock_carbon_client
    
    print("Successfully imported mock data modules!")
    
    # Test Prometheus mock client
    prometheus_client = create_mock_prometheus_client("test-cluster")
    metrics_data = prometheus_client.query_metrics(
        metric_names=["cpu", "memory", "power"],
        step_seconds=300  # 5-minute intervals
    )
    
    print("\n=== Prometheus Mock Data Sample ===")
    print(f"Cluster: test-cluster")
    print(f"Metrics collected: {', '.join(metrics_data['metrics'].keys())}")
    print(f"Number of data points: {len(metrics_data['timestamps'])}")
    print(f"Nodes: {', '.join(metrics_data['nodes'].keys())}")
    
    # Sample CPU data
    if 'cpu' in metrics_data['metrics']:
        cpu_samples = metrics_data['metrics']['cpu']
        print(f"\nCluster CPU utilization (first 3 samples):")
        for i in range(min(3, len(cpu_samples))):
            print(f"  {i+1}: {cpu_samples[i]*100:.1f}%")
    
    # Test Carbon mock client
    carbon_client = create_mock_carbon_client("US-CA")
    carbon_data = carbon_client.get_current_intensity()
    forecast = carbon_client.get_forecast(hours=12)
    
    print("\n=== Carbon Mock Data Sample ===")
    print(f"Region: {carbon_data['region']}")
    print(f"Current intensity: {carbon_data['carbon_intensity']:.1f} {carbon_data['unit']}")
    print(f"Forecast available for {len(forecast['forecast'])} hours")
    print(f"Best time window starts at: {forecast['analysis']['best_time_window']['start_time']}")
    print(f"Best time intensity: {forecast['analysis']['best_time_window']['intensity']:.1f} {forecast['unit']}")
    
    print("\nMock data test successful!")
except ImportError as e:
    print(f"Failed to import mock data modules: {e}")
    print(f"Python path: {sys.path}")

def test_prometheus_mock_client():
    from src.greenops_agent.mockdata.prometheus import create_mock_prometheus_client
    client = create_mock_prometheus_client("test-cluster")
    metrics_data = client.query_metrics(metric_names=["cpu", "memory", "power"], step_seconds=300)
    assert "cpu" in metrics_data["metrics"]
    assert "memory" in metrics_data["metrics"]
    assert "power" in metrics_data["metrics"]
    assert len(metrics_data["timestamps"]) > 0
    assert len(metrics_data["nodes"]) > 0

def test_carbon_mock_client():
    from src.greenops_agent.mockdata.carbon import create_mock_carbon_client
    client = create_mock_carbon_client("US-CA")
    carbon_data = client.get_current_intensity()
    forecast = client.get_forecast(hours=12)
    assert "region" in carbon_data
    assert "carbon_intensity" in carbon_data
    assert "unit" in carbon_data
    assert "forecast" in forecast
    assert "analysis" in forecast
    assert len(forecast["forecast"]) == 12
