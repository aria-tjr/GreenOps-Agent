"""
Mock data generators for GreenOps Agent testing.

This package contains mock data generators that can be used to test
the GreenOps Agent components without requiring real external services
like Prometheus or carbon intensity APIs.
"""

from .carbon import create_mock_carbon_client
from .prometheus import create_mock_client, MockPrometheusClient
from .workload import (
    generate_sine_pattern,
    generate_spike_pattern,
    generate_growth_pattern,
    generate_cpu_memory_patterns,
    generate_cluster_workload
)

__all__ = [
    'create_mock_carbon_client',
    'create_mock_client',
    'MockPrometheusClient',
    'generate_sine_pattern',
    'generate_spike_pattern',
    'generate_growth_pattern',
    'generate_cpu_memory_patterns',
    'generate_cluster_workload'
]