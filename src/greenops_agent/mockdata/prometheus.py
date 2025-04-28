"""
Mock Prometheus data module for testing and development.

This module provides simulated metrics data that follows realistic patterns
without requiring an actual Prometheus server connection.
"""

import random
import time
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional


class MockPrometheusClient:
    """
    Mock Prometheus client that provides simulated metrics data.
    
    This class simulates common metrics that would typically be 
    collected from a Kubernetes cluster including CPU, memory,
    network traffic, and power consumption estimates.
    """
    
    def __init__(self, cluster_name: str = "mock-cluster"):
        """
        Initialize the mock Prometheus client.
        
        Args:
            cluster_name: Name of the simulated cluster
        """
        self.cluster_name = cluster_name
        self.start_time = time.time()
        
        # Configure simulated cluster
        self.nodes = [
            {"name": "node-1", "cpus": 32, "memory_gb": 128, "power_watts_idle": 100, "power_watts_max": 450},
            {"name": "node-2", "cpus": 32, "memory_gb": 128, "power_watts_idle": 100, "power_watts_max": 450},
            {"name": "node-3", "cpus": 16, "memory_gb": 64, "power_watts_idle": 60, "power_watts_max": 280},
            {"name": "node-4", "cpus": 16, "memory_gb": 64, "power_watts_idle": 60, "power_watts_max": 280},
        ]
        
        self.namespaces = ["default", "kube-system", "monitoring", "app-production", "app-staging"]
        
        # Simulated workloads with patterns
        self.workloads = {
            "web-frontend": {"cpu_pattern": "variable", "memory_pattern": "stable"},
            "api-service": {"cpu_pattern": "spiky", "memory_pattern": "growing"},
            "database": {"cpu_pattern": "cyclic", "memory_pattern": "stable"},
            "batch-job": {"cpu_pattern": "periodic", "memory_pattern": "periodic"},
            "etl-processor": {"cpu_pattern": "periodic", "memory_pattern": "sawtooth"},
        }
        
        # Time patterns - different for each workload
        self.time_patterns = {}
        for workload in self.workloads:
            # Random offset to stagger workload patterns
            self.time_patterns[workload] = random.random() * 10000
    
    def _generate_cpu_value(self, workload: str, node: Dict[str, Any], timestamp: Optional[float] = None) -> float:
        """
        Generate a simulated CPU usage value based on the workload pattern.
        
        Args:
            workload: Name of the workload
            node: Node metadata dictionary
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            CPU usage as a float between 0 and 1
        """
        if timestamp is None:
            timestamp = time.time()
        
        pattern = self.workloads[workload]["cpu_pattern"]
        time_offset = self.time_patterns[workload]
        
        base_value = 0.0
        if pattern == "stable":
            base_value = 0.2 + random.random() * 0.1  # 20-30%
        elif pattern == "variable":
            # Sinusoidal with 15-minute period
            cycle = math.sin((timestamp + time_offset) / 900 * 2 * math.pi)
            base_value = 0.3 + 0.2 * cycle  # 10-50%
        elif pattern == "spiky":
            # Occasional spikes
            if random.random() < 0.05:  # 5% chance of spike
                base_value = 0.6 + random.random() * 0.3  # 60-90% spike
            else:
                base_value = 0.1 + random.random() * 0.2  # 10-30% baseline
        elif pattern == "cyclic":
            # 1-hour cycle
            cycle = 0.5 + 0.5 * math.sin((timestamp + time_offset) / 3600 * 2 * math.pi)
            base_value = 0.2 + 0.4 * cycle  # 20-60%
        elif pattern == "periodic":
            # 4-hour cycle with sharp peaks
            periodic = (timestamp + time_offset) % 14400 / 14400  # 0-1 over 4 hours
            if periodic < 0.1:  # Peak for 10% of cycle
                base_value = 0.7 + random.random() * 0.2  # 70-90%
            else:
                base_value = 0.05 + random.random() * 0.1  # 5-15%
        
        # Add small noise
        noise = random.random() * 0.05 - 0.025  # ±2.5% noise
        
        # Ensure value is between 0 and 1
        return max(0.01, min(0.99, base_value + noise))
    
    def _generate_memory_value(self, workload: str, node: Dict[str, Any], timestamp: Optional[float] = None) -> float:
        """
        Generate a simulated memory usage value based on the workload pattern.
        
        Args:
            workload: Name of the workload
            node: Node metadata dictionary
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Memory usage as a float between 0 and 1
        """
        if timestamp is None:
            timestamp = time.time()
        
        pattern = self.workloads[workload]["memory_pattern"]
        time_offset = self.time_patterns[workload]
        elapsed_hours = (timestamp - self.start_time) / 3600
        
        base_value = 0.0
        if pattern == "stable":
            base_value = 0.4 + random.random() * 0.1  # 40-50%
        elif pattern == "growing":
            # Gradually increasing over time with cap
            growth_factor = min(0.3, elapsed_hours / 24 * 0.1)  # Grow 10% per day, max 30% growth
            base_value = 0.3 + growth_factor + random.random() * 0.05  # 30-65% (capped)
        elif pattern == "sawtooth":
            # Increases over 6 hours then drops
            saw_cycle = ((timestamp + time_offset) % 21600) / 21600  # 0-1 over 6 hours
            if saw_cycle < 0.9:  # 90% of cycle is growth
                base_value = 0.3 + saw_cycle * 0.4  # 30-70% growing
            else:
                base_value = 0.3  # Reset to 30%
        elif pattern == "periodic":
            # Changes with CPU pattern for batch jobs
            cpu_pattern = self.workloads[workload]["cpu_pattern"]
            if cpu_pattern == "periodic":
                periodic = (timestamp + time_offset) % 14400 / 14400  # 0-1 over 4 hours
                if periodic < 0.1:  # Peak with CPU
                    base_value = 0.6 + random.random() * 0.1  # 60-70%
                else:
                    base_value = 0.2 + random.random() * 0.1  # 20-30%
            else:
                base_value = 0.4  # Default 40%
        
        # Add small noise
        noise = random.random() * 0.03 - 0.015  # ±1.5% noise
        
        # Ensure value is between 0 and 1
        return max(0.05, min(0.95, base_value + noise))
    
    def _generate_power_value(self, node: Dict[str, Any], cpu_usage: float) -> float:
        """
        Generate a simulated power consumption value based on CPU usage.
        
        Args:
            node: Node metadata dictionary
            cpu_usage: Current CPU usage (0-1)
            
        Returns:
            Power consumption in watts
        """
        idle_power = node["power_watts_idle"]
        max_power = node["power_watts_max"]
        
        # Power follows CPU usage but with non-linear curve
        # Even at 0% CPU, servers consume significant power (idle_power)
        power_curve = idle_power + (max_power - idle_power) * (cpu_usage ** 1.4)
        
        # Add some noise (±5%)
        noise_factor = 1.0 + (random.random() * 0.1 - 0.05)
        
        return power_curve * noise_factor
    
    def _generate_node_metrics(self, node: Dict[str, Any], timestamp: Optional[float] = None) -> Dict[str, float]:
        """
        Generate metrics for a specific node.
        
        Args:
            node: Node metadata dictionary
            timestamp: Optional timestamp, defaults to current time
            
        Returns:
            Dictionary with generated metrics
        """
        if timestamp is None:
            timestamp = time.time()
            
        # Calculate total cluster utilization effects - nodes influence each other
        total_cpu = 0
        for workload in self.workloads:
            # Distribute workloads across nodes differently
            workload_factor = hash(workload + node["name"]) % 100 / 100.0
            cpu_contribution = self._generate_cpu_value(workload, node, timestamp) * workload_factor
            total_cpu += cpu_contribution
            
        # Scale to reasonable range based on node size
        node_scale_factor = node["cpus"] / 32  # Scale based on CPU count
        total_cpu = min(0.95, total_cpu * node_scale_factor)
        
        # Memory follows its own pattern but is correlated with CPU
        total_memory = 0
        for workload in self.workloads:
            # Distribute workloads across nodes differently
            workload_factor = hash(workload + node["name"] + "mem") % 100 / 100.0
            memory_contribution = self._generate_memory_value(workload, node, timestamp) * workload_factor
            total_memory += memory_contribution
            
        # Scale memory and ensure reasonable range
        total_memory = min(0.9, total_memory * node_scale_factor)
        
        # Calculate power based on CPU usage and node characteristics
        power_usage = self._generate_power_value(node, total_cpu)
        
        # Network traffic - correlates somewhat with CPU but has its own pattern
        network_in = 100 * 1024 * 1024 * (0.2 + 0.8 * total_cpu) * (0.5 + random.random())  # 20-100 MB/s base
        network_out = network_in * (0.7 + 0.6 * random.random())  # Output typically less than input
        
        return {
            "cpu_usage": total_cpu,
            "memory_usage": total_memory,
            "power_watts": power_usage,
            "network_in_bytes": network_in,
            "network_out_bytes": network_out,
        }
    
    def query_metrics(self, 
                     metric_names: List[str] = None, 
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None,
                     step_seconds: int = 60) -> Dict[str, Any]:
        """
        Query mock metrics data.
        
        Args:
            metric_names: List of metrics to query, defaults to all
            start_time: Start timestamp, defaults to 1 hour ago
            end_time: End timestamp, defaults to now
            step_seconds: Step size in seconds
            
        Returns:
            Dictionary with queried metrics
        """
        if metric_names is None:
            metric_names = ["cpu", "memory", "power", "network"]
            
        current_time = time.time()
        if start_time is None:
            start_time = current_time - 3600  # 1 hour ago
        if end_time is None:
            end_time = current_time
        
        # Generate time series
        timestamps = []
        time_point = start_time
        while time_point <= end_time:
            timestamps.append(time_point)
            time_point += step_seconds
        
        results = {
            "timestamps": timestamps,
            "nodes": {},
            "metrics": {}
        }
        
        for metric in metric_names:
            results["metrics"][metric] = {}
        
        # Generate metrics for each node at each timestamp
        for node in self.nodes:
            node_name = node["name"]
            results["nodes"][node_name] = {
                "cpu_capacity": node["cpus"],
                "memory_capacity_gb": node["memory_gb"],
                "metrics": {}
            }
            
            for metric in metric_names:
                results["nodes"][node_name]["metrics"][metric] = []
            
            for ts in timestamps:
                node_metrics = self._generate_node_metrics(node, ts)
                
                if "cpu" in metric_names:
                    results["nodes"][node_name]["metrics"]["cpu"].append(node_metrics["cpu_usage"])
                    
                if "memory" in metric_names:
                    results["nodes"][node_name]["metrics"]["memory"].append(node_metrics["memory_usage"])
                    
                if "power" in metric_names:
                    results["nodes"][node_name]["metrics"]["power"].append(node_metrics["power_watts"])
                    
                if "network" in metric_names:
                    # Combine in/out for simplicity
                    results["nodes"][node_name]["metrics"]["network"].append({
                        "in_bytes": node_metrics["network_in_bytes"],
                        "out_bytes": node_metrics["network_out_bytes"]
                    })
        
        # Calculate cluster-wide aggregates
        for metric in metric_names:
            if metric == "cpu":
                results["metrics"]["cpu"] = [
                    sum(results["nodes"][node["name"]]["metrics"]["cpu"][i] * node["cpus"] 
                        for node in self.nodes) / sum(node["cpus"] for node in self.nodes)
                    for i in range(len(timestamps))
                ]
            elif metric == "memory":
                results["metrics"]["memory"] = [
                    sum(results["nodes"][node["name"]]["metrics"]["memory"][i] * node["memory_gb"] 
                        for node in self.nodes) / sum(node["memory_gb"] for node in self.nodes)
                    for i in range(len(timestamps))
                ]
            elif metric == "power":
                results["metrics"]["power"] = [
                    sum(results["nodes"][node["name"]]["metrics"]["power"][i] for node in self.nodes)
                    for i in range(len(timestamps))
                ]
            elif metric == "network":
                results["metrics"]["network"] = [
                    {
                        "in_bytes": sum(results["nodes"][node["name"]]["metrics"]["network"][i]["in_bytes"] 
                                       for node in self.nodes),
                        "out_bytes": sum(results["nodes"][node["name"]]["metrics"]["network"][i]["out_bytes"] 
                                        for node in self.nodes)
                    }
                    for i in range(len(timestamps))
                ]
        
        # Generate pod-level mock data for compatibility with MetricsCollector
        pods = {}
        pod_names = [
            "web-frontend-abc123",
            "api-service-def456",
            "database-xyz789",
            "batch-job-ghi012",
            "etl-processor-jkl345"
        ]
        for pod_name in pod_names:
            # Simulate resource usage and requests/limits
            cpu_usage = [(ts, random.uniform(0.05, 0.5)) for ts in timestamps]
            mem_usage = [(ts, random.uniform(50, 500)) for ts in timestamps]  # MB
            pods[pod_name] = {
                "resources": {
                    "cpu": {
                        "usage": cpu_usage,
                        "request": random.uniform(0.1, 0.3),
                        "limit": random.uniform(0.4, 1.0)
                    },
                    "memory": {
                        "usage_mb": mem_usage,
                        "request_mb": random.uniform(100, 200),
                        "limit_mb": random.uniform(300, 600)
                    }
                }
            }
        results["pods"] = pods
        
        return results


def create_mock_prometheus_client(cluster_name: str = "mock-cluster") -> MockPrometheusClient:
    """
    Create a mock Prometheus client.
    
    Args:
        cluster_name: Name for the mock cluster
        
    Returns:
        A configured mock Prometheus client
    """
    return MockPrometheusClient(cluster_name)