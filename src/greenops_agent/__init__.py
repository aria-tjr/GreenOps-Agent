"""
GreenOps Agent - Optimize Kubernetes workloads for energy efficiency and carbon footprint.

This package provides tools to analyze Kubernetes resource usage,
correlate with carbon intensity data, and provide recommendations for
optimizing workloads to reduce energy consumption and carbon emissions.
"""

__version__ = "0.1.0"

from greenops_agent.metrics import MetricsCollector
from greenops_agent.carbon import get_carbon_fetcher, CarbonIntensityFetcher, FallbackCarbonIntensityFetcher
from greenops_agent.predictor import WorkloadPredictor
from greenops_agent.recommender import RecommendationEngine
from greenops_agent.main import run_cli, run_api