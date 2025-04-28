#!/usr/bin/env python
"""
Test script to verify and debug mock data module availability.

This script tests the mock data availability in the GreenOps agent
and reports detailed information about what modules can be imported.
"""

import sys
import os
import importlib
import logging
from pathlib import Path

# Add the src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(script_dir, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    print(f"Added {src_path} to Python path")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock-test")

def test_direct_import():
    """Test direct import from mockdata modules."""
    print("\n=== Testing Direct Imports ===")
    try:
        from greenops_agent.mockdata import prometheus as mock_prometheus
        print("✅ Successfully imported greenops_agent.mockdata.prometheus")
    except ImportError as e:
        print(f"❌ Failed to import greenops_agent.mockdata.prometheus: {e}")
    
    try:
        from greenops_agent.mockdata import carbon as mock_carbon
        print("✅ Successfully imported greenops_agent.mockdata.carbon")
    except ImportError as e:
        print(f"❌ Failed to import greenops_agent.mockdata.carbon: {e}")
    
    try:
        from greenops_agent.mockdata import workload as mock_workload
        print("✅ Successfully imported greenops_agent.mockdata.workload")
    except ImportError as e:
        print(f"❌ Failed to import greenops_agent.mockdata.workload: {e}")

def test_importlib_import():
    """Test import via importlib (more detailed error reporting)."""
    print("\n=== Testing Imports via importlib ===")
    modules_to_test = [
        "greenops_agent",
        "greenops_agent.mockdata",
        "greenops_agent.mockdata.prometheus",
        "greenops_agent.mockdata.carbon",
        "greenops_agent.mockdata.workload",
        "src.greenops_agent.mockdata",
        "src.greenops_agent.mockdata.prometheus",
        "src.greenops_agent.mockdata.carbon",
        "src.greenops_agent.mockdata.workload"
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ Successfully imported {module_name}")
            if module_name.endswith("prometheus"):
                if hasattr(module, "create_mock_prometheus_client"):
                    print(f"   - Module has create_mock_prometheus_client function")
                else:
                    print(f"   - Module does NOT have create_mock_prometheus_client function")
            elif module_name.endswith("carbon"):
                if hasattr(module, "create_mock_carbon_client"):
                    print(f"   - Module has create_mock_carbon_client function")
                else:
                    print(f"   - Module does NOT have create_mock_carbon_client function")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")

def test_sys_path():
    """Print system path for debugging."""
    print("\n=== Python sys.path ===")
    for i, path in enumerate(sys.path):
        print(f"{i}: {path}")

def test_package_structure():
    """Check the package structure in the file system."""
    print("\n=== Package Structure ===")
    
    # Check current directory
    cwd = os.path.abspath(os.curdir)
    print(f"Current directory: {cwd}")
    
    # Check for src directory
    src_path = os.path.join(cwd, "src")
    if os.path.exists(src_path):
        print(f"✅ src directory exists: {src_path}")
    else:
        print(f"❌ src directory NOT found")
    
    # Check for module directories
    paths_to_check = [
        ["src", "greenops_agent"],
        ["src", "greenops_agent", "mockdata"],
        ["greenops_agent"],
        ["greenops_agent", "mockdata"]
    ]
    
    for path_parts in paths_to_check:
        path = os.path.join(cwd, *path_parts)
        if os.path.exists(path):
            print(f"✅ Directory exists: {path}")
            
            # Check for __init__.py
            init_file = os.path.join(path, "__init__.py")
            if os.path.exists(init_file):
                print(f"   - __init__.py exists: {init_file}")
            else:
                print(f"   - __init__.py NOT found in {path}")
                
            # For mockdata directory, check for mock modules
            if path_parts[-1] == "mockdata":
                for module in ["prometheus.py", "carbon.py", "workload.py"]:
                    module_file = os.path.join(path, module)
                    if os.path.exists(module_file):
                        print(f"   - {module} exists")
                    else:
                        print(f"   - {module} NOT found")
        else:
            print(f"❌ Directory NOT found: {path}")

def test_direct_usage():
    """Test direct usage of mock modules."""
    print("\n=== Testing Mock Module Usage ===")
    
    # Test prometheus mock
    try:
        # Try direct import first
        try:
            from greenops_agent.mockdata.prometheus import create_mock_prometheus_client
        except ImportError:
            from src.greenops_agent.mockdata.prometheus import create_mock_prometheus_client
        
        prometheus_client = create_mock_prometheus_client("test-cluster")
        metrics_data = prometheus_client.query_metrics(
            metric_names=["cpu", "memory", "power"],
            step_seconds=300  # 5-minute intervals
        )
        
        print("✅ Successfully used mock Prometheus client")
        print(f"   - Metrics collected: {', '.join(metrics_data['metrics'].keys())}")
        print(f"   - Number of data points: {len(metrics_data['timestamps'])}")
        print(f"   - Nodes: {', '.join(metrics_data['nodes'].keys())}")
    except Exception as e:
        print(f"❌ Failed to use mock Prometheus client: {e}")
    
    # Test carbon mock
    try:
        # Try direct import first
        try:
            from greenops_agent.mockdata.carbon import create_mock_carbon_client
        except ImportError:
            from src.greenops_agent.mockdata.carbon import create_mock_carbon_client
        
        carbon_client = create_mock_carbon_client("US-CA")
        carbon_data = carbon_client.get_current_intensity()
        forecast = carbon_client.get_forecast(hours=12)
        
        print("✅ Successfully used mock Carbon client")
        print(f"   - Region: {carbon_data['region']}")
        print(f"   - Current intensity: {carbon_data['carbon_intensity']:.1f} {carbon_data['unit']}")
        print(f"   - Forecast available for {len(forecast['forecast'])} hours")
    except Exception as e:
        print(f"❌ Failed to use mock Carbon client: {e}")

def test_agent_with_mocks():
    """Test the agent with mock data."""
    print("\n=== Testing GreenOps Agent with Mock Data ===")
    try:
        from greenops_agent.metrics import MetricsCollector
        from greenops_agent.carbon import get_carbon_fetcher
        
        # Create instances with mock data
        metrics_collector = MetricsCollector(prometheus_url=None, use_mock_data=True)
        carbon_fetcher = get_carbon_fetcher(api_key=None, region="DEFAULT", use_mock=True)
        
        # Try to get metrics
        metrics = metrics_collector.collect_all_metrics()
        if metrics:
            print("✅ Successfully collected mock metrics")
            if "current" in metrics:
                print(f"   - Current metrics available: {', '.join(metrics['current'].keys())}")
            if "historical" in metrics:
                print(f"   - Historical metrics available")
        else:
            print("❌ Failed to collect mock metrics (empty result)")
        
        # Try to get carbon data
        carbon_data = carbon_fetcher.get_carbon_data()
        if carbon_data:
            print("✅ Successfully collected mock carbon data")
            print(f"   - Region: {carbon_data.get('region', 'unknown')}")
            print(f"   - Carbon intensity: {carbon_data.get('current_intensity', 0):.1f}")
            if carbon_data.get('forecast'):
                print(f"   - Forecast points: {len(carbon_data['forecast'])}")
        else:
            print("❌ Failed to collect mock carbon data (empty result)")
            
    except Exception as e:
        print(f"❌ Agent test failed: {e}")

if __name__ == "__main__":
    print("\n🔍 GreenOps Agent Mock Data Availability Test")
    print("============================================")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Run all tests
    test_sys_path()
    test_package_structure()
    test_direct_import()
    test_importlib_import()
    test_direct_usage()
    test_agent_with_mocks()
    
    print("\n✅ Testing complete!")