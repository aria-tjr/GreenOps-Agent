"""
Mock workload data generator for GreenOps Agent testing.

This module generates realistic synthetic workload patterns for testing
the GreenOps Agent without requiring real Kubernetes metrics.
"""

import random
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any


def generate_sine_pattern(
    points: int, 
    min_val: float = 10.0, 
    max_val: float = 90.0,
    period_hours: float = 24.0,
    noise_factor: float = 0.1
) -> List[Tuple[datetime, float]]:
    """
    Generate a sine wave pattern with noise to simulate daily workloads.
    
    Args:
        points: Number of data points to generate
        min_val: Minimum value in the pattern
        max_val: Maximum value in the pattern
        period_hours: Period of the sine wave in hours
        noise_factor: Amount of random noise to add (0-1)
        
    Returns:
        List of (timestamp, value) tuples
    """
    now = datetime.now()
    data = []
    amplitude = (max_val - min_val) / 2
    offset = min_val + amplitude
    
    for i in range(points):
        # Generate timestamp going back in time
        ts = now - timedelta(hours=(points - i - 1) / (points / period_hours))
        
        # Generate sine wave value with period_hours
        hour_fraction = (points - i - 1) / (points / period_hours)
        sine_val = math.sin(hour_fraction * 2 * math.pi / period_hours)
        
        # Scale to min-max range and add noise
        value = offset + amplitude * sine_val
        noise = random.uniform(-noise_factor * amplitude, noise_factor * amplitude)
        final_val = max(min_val, min(max_val, value + noise))
        
        data.append((ts, final_val))
    
    return data


def generate_spike_pattern(
    points: int,
    base_val: float = 30.0,
    spike_val: float = 90.0,
    spike_width: int = 5,
    noise_factor: float = 0.1
) -> List[Tuple[datetime, float]]:
    """
    Generate a pattern with spikes to simulate traffic spikes.
    
    Args:
        points: Number of data points to generate
        base_val: Baseline value
        spike_val: Value during spikes
        spike_width: Width of spikes in data points
        noise_factor: Amount of random noise to add (0-1)
        
    Returns:
        List of (timestamp, value) tuples
    """
    now = datetime.now()
    data = []
    
    # Decide where to put spikes
    spike_positions = [
        random.randint(spike_width, points//3 - spike_width),
        random.randint(points//3 + spike_width, 2*points//3 - spike_width),
        random.randint(2*points//3 + spike_width, points - spike_width)
    ]
    
    for i in range(points):
        # Generate timestamp going back in time
        ts = now - timedelta(hours=points - i - 1)
        
        # Determine if this point is part of a spike
        in_spike = any(abs(i - pos) < spike_width/2 for pos in spike_positions)
        
        # Generate value based on whether it's in a spike
        if in_spike:
            # Calculate distance to nearest spike center
            min_dist = min(abs(i - pos) for pos in spike_positions)
            # Smooth spike shape using cosine
            spike_factor = math.cos(min_dist / (spike_width/2) * math.pi/2)
            value = base_val + (spike_val - base_val) * spike_factor
        else:
            value = base_val
            
        # Add noise
        noise = random.uniform(-noise_factor * base_val, noise_factor * base_val)
        final_val = value + noise
        
        data.append((ts, final_val))
    
    return data


def generate_growth_pattern(
    points: int,
    start_val: float = 10.0,
    end_val: float = 80.0,
    noise_factor: float = 0.1
) -> List[Tuple[datetime, float]]:
    """
    Generate a pattern with steady growth to simulate increasing workloads.
    
    Args:
        points: Number of data points to generate
        start_val: Starting value
        end_val: Ending value
        noise_factor: Amount of random noise to add (0-1)
        
    Returns:
        List of (timestamp, value) tuples
    """
    now = datetime.now()
    data = []
    
    for i in range(points):
        # Generate timestamp going back in time
        ts = now - timedelta(hours=points - i - 1)
        
        # Linear interpolation between start and end values
        value = start_val + (end_val - start_val) * (i / (points - 1))
        
        # Add some random noise
        noise = random.uniform(-noise_factor * value, noise_factor * value)
        final_val = value + noise
        
        data.append((ts, final_val))
    
    return data


def generate_cpu_memory_patterns(
    pattern_type: str = "daily",
    points: int = 144,  # 6 days of hourly data
    namespace: str = "default",
    deployment: str = "example-app"
) -> Dict[str, Dict[str, List[Tuple[datetime, float]]]]:
    """
    Generate related CPU and memory patterns for a deployment.
    
    Args:
        pattern_type: Type of pattern ("daily", "spike", or "growth")
        points: Number of data points to generate
        namespace: Kubernetes namespace
        deployment: Name of the deployment
        
    Returns:
        Dictionary of resource metrics with (timestamp, value) tuples
    """
    metrics = {
        "cpu": {},
        "memory": {}
    }
    
    # Generate CPU usage pattern
    if pattern_type == "daily":
        metrics["cpu"]["usage"] = generate_sine_pattern(
            points, 
            min_val=10.0, 
            max_val=70.0, 
            period_hours=24.0
        )
        # Memory follows CPU but with less variation and lag
        cpu_values = [v for _, v in metrics["cpu"]["usage"]]
        timestamps = [ts for ts, _ in metrics["cpu"]["usage"]]
        mem_values = []
        for i in range(points):
            # Memory lags behind CPU but is more stable
            lag_idx = max(0, i - 3)
            # Base memory on average of current CPU and lagged CPU
            base = (cpu_values[i] + cpu_values[lag_idx]) / 2
            # Less variation in memory
            mem_val = base * 1.5 + 20 + random.uniform(-5, 5)
            mem_values.append(mem_val)
        
        metrics["memory"]["usage"] = list(zip(timestamps, mem_values))
        
    elif pattern_type == "spike":
        metrics["cpu"]["usage"] = generate_spike_pattern(
            points, 
            base_val=30.0, 
            spike_val=90.0
        )
        # Memory follows CPU spikes but with dampening
        cpu_values = [v for _, v in metrics["cpu"]["usage"]]
        timestamps = [ts for ts, _ in metrics["cpu"]["usage"]]
        mem_values = []
        for i in range(points):
            # Memory increases more slowly than CPU during spikes
            if i > 0:
                prev_mem = mem_values[i-1]
                # If CPU is going up, memory follows more slowly
                if i < len(cpu_values) - 1:
                    cpu_change = cpu_values[i+1] - cpu_values[i]
                    mem_change = cpu_change * 0.7  # Memory changes slower
                else:
                    mem_change = 0
                mem_val = prev_mem + mem_change + random.uniform(-3, 3)
            else:
                mem_val = cpu_values[0] * 1.2 + random.uniform(-5, 5)
                
            mem_values.append(mem_val)
            
        metrics["memory"]["usage"] = list(zip(timestamps, mem_values))
        
    else:  # growth pattern
        metrics["cpu"]["usage"] = generate_growth_pattern(
            points, 
            start_val=20.0, 
            end_val=85.0
        )
        # Memory grows along with CPU
        cpu_values = [v for _, v in metrics["cpu"]["usage"]]
        timestamps = [ts for ts, _ in metrics["cpu"]["usage"]]
        mem_values = []
        
        # Memory starts higher but grows slightly slower
        mem_start = cpu_values[0] * 1.5 + 10
        mem_end = cpu_values[-1] * 1.3 + 15
        
        for i in range(points):
            progress = i / (points - 1)
            mem_val = mem_start + (mem_end - mem_start) * progress
            mem_val += random.uniform(-5, 5)
            mem_values.append(mem_val)
            
        metrics["memory"]["usage"] = list(zip(timestamps, mem_values))
    
    # Generate CPU and memory limits
    cpu_limit = max([v for _, v in metrics["cpu"]["usage"]]) * 1.5
    memory_limit = max([v for _, v in metrics["memory"]["usage"]]) * 1.3
    
    # Generate requests as percentage of limits
    cpu_request = cpu_limit * 0.5
    memory_request = memory_limit * 0.7
    
    # Create static limit and request values
    now = datetime.now()
    timestamps = [now - timedelta(hours=i) for i in range(points)]
    metrics["cpu"]["limit"] = [(ts, cpu_limit) for ts in timestamps]
    metrics["cpu"]["request"] = [(ts, cpu_request) for ts in timestamps]
    metrics["memory"]["limit"] = [(ts, memory_limit) for ts in timestamps]
    metrics["memory"]["request"] = [(ts, memory_request) for ts in timestamps]
    
    # Add metadata
    result = {
        "metadata": {
            "namespace": namespace,
            "deployment": deployment,
            "pattern": pattern_type,
            "points": points,
            "timestamp": datetime.now().isoformat()
        },
        "metrics": metrics
    }
    
    return result


def generate_cluster_workload(
    num_deployments: int = 5,
    points_per_deployment: int = 72,  # 3 days of hourly data
    include_inefficient: bool = True
) -> Dict[str, Any]:
    """
    Generate a complete cluster workload with multiple deployments.
    
    Args:
        num_deployments: Number of deployments to simulate
        points_per_deployment: Number of data points per deployment
        include_inefficient: Whether to include inefficient deployments
        
    Returns:
        Dictionary with complete cluster workload data
    """
    deployments = []
    namespaces = ["default", "kube-system", "monitoring", "application", "backend"]
    pattern_types = ["daily", "spike", "growth"]
    
    for i in range(num_deployments):
        namespace = random.choice(namespaces)
        deployment = f"deployment-{i+1}"
        pattern = random.choice(pattern_types)
        
        # Generate workload data for this deployment
        deployment_data = generate_cpu_memory_patterns(
            pattern_type=pattern,
            points=points_per_deployment,
            namespace=namespace,
            deployment=deployment
        )
        
        # If including inefficient deployments, make some of them inefficient
        if include_inefficient and random.random() < 0.4:
            metrics = deployment_data["metrics"]
            
            # Make CPU overprovisioned (high request/limit but low usage)
            if random.random() < 0.5:
                cpu_usage_max = max(v for _, v in metrics["cpu"]["usage"])
                # Set request much higher than needed
                new_cpu_request = cpu_usage_max * random.uniform(2.0, 4.0)
                # Set limit even higher
                new_cpu_limit = new_cpu_request * random.uniform(1.2, 1.5)
                
                # Replace the values
                metrics["cpu"]["request"] = [(ts, new_cpu_request) for ts, _ in metrics["cpu"]["request"]]
                metrics["cpu"]["limit"] = [(ts, new_cpu_limit) for ts, _ in metrics["cpu"]["limit"]]
                
                # Add inefficiency metadata
                deployment_data["metadata"]["inefficient"] = "cpu-overprovisioned"
            else:
                # Make memory underprovisioned (low request/limit but high usage)
                mem_usage_max = max(v for _, v in metrics["memory"]["usage"])
                # Set request lower than actual usage
                new_mem_request = mem_usage_max * random.uniform(0.5, 0.8)
                # Set limit only slightly above request
                new_mem_limit = new_mem_request * random.uniform(1.05, 1.2)
                
                # Replace the values
                metrics["memory"]["request"] = [(ts, new_mem_request) for ts, _ in metrics["memory"]["request"]]
                metrics["memory"]["limit"] = [(ts, new_mem_limit) for ts, _ in metrics["memory"]["limit"]]
                
                # Add inefficiency metadata
                deployment_data["metadata"]["inefficient"] = "memory-underprovisioned"
        
        deployments.append(deployment_data)
    
    # Return complete cluster data
    return {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_deployments": num_deployments,
            "data_points": points_per_deployment,
            "namespaces": list(set(d["metadata"]["namespace"] for d in deployments))
        },
        "deployments": deployments
    }


if __name__ == "__main__":
    # Example usage
    data = generate_cluster_workload(num_deployments=3, points_per_deployment=48)
    print(f"Generated mock data for {len(data['deployments'])} deployments")
    
    # Example of accessing the data
    for i, deployment in enumerate(data["deployments"]):
        metadata = deployment["metadata"]
        cpu_usage = deployment["metrics"]["cpu"]["usage"]
        print(f"Deployment {i+1}: {metadata['namespace']}/{metadata['deployment']}")
        print(f"  Pattern: {metadata.get('pattern', 'unknown')}")
        print(f"  Inefficient: {metadata.get('inefficient', 'no')}")
        print(f"  CPU usage (last value): {cpu_usage[-1][1]:.2f}")
        print(f"  Number of data points: {len(cpu_usage)}")
        print("---")