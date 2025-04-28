"""
GreenOps Agent main module.

This module serves as the entry point for the GreenOps Agent, implementing
both a command-line interface (CLI) and a FastAPI-based web service for
optimizing Kubernetes workloads for energy efficiency.
"""

import os
import sys
import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich import print as rprint
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks

from greenops_agent.metrics import MetricsCollector
from greenops_agent.carbon import get_carbon_fetcher
from greenops_agent.predictor import WorkloadPredictor
from greenops_agent.recommender import RecommendationEngine

# Load environment variables from .env file if present
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger("greenops_agent")

# Create Typer app for CLI
cli = typer.Typer(help="GreenOps Agent - Optimize Kubernetes workloads for energy efficiency")

# Create FastAPI app for web service
api = FastAPI(
    title="GreenOps Agent API",
    description="API for optimizing Kubernetes workloads for energy efficiency and carbon footprint reduction",
    version="0.1.0"
)

# Rich console for nice CLI output
console = Console()

# Global cache for metrics and recommendations
cache = {
    "last_metrics": None,
    "last_carbon_data": None,
    "last_workload_prediction": None,
    "last_recommendations": None,
    "last_update_time": None
}


def get_config_from_env() -> Dict[str, Any]:
    """
    Get configuration from environment variables.
    
    Returns:
        Dictionary with configuration values
    """
    return {
        "prometheus_url": os.getenv("PROMETHEUS_URL", "http://prometheus-server:9090"),
        "carbon_api_key": os.getenv("CARBON_API_KEY", ""),
        "region": os.getenv("REGION", "DEFAULT"),
        "update_interval": int(os.getenv("UPDATE_INTERVAL", "300")),  # 5 minutes default
        "debug": os.getenv("DEBUG", "false").lower() == "true"
    }


async def collect_all_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect all data needed for analysis.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with collected metrics, carbon data, and predictions
    """
    try:
        # Check if we should use mock data
        use_mock = config.get("use_mock", False)
        
        # Collect metrics with mock data if specified
        metrics_collector = MetricsCollector(
            prometheus_url=config["prometheus_url"],
            use_mock_data=use_mock
        )
        metrics = metrics_collector.collect_all_metrics()
        
        # Get carbon intensity data, with mock if specified
        carbon_fetcher = get_carbon_fetcher(
            api_key=config["carbon_api_key"],
            region=config["region"],
            use_mock=use_mock
        )
        carbon_data = carbon_fetcher.get_carbon_data()
        
        # Get workload prediction
        predictor = WorkloadPredictor()
        workload_prediction = None
        
        # Only do prediction if we have historical data
        if metrics and "historical" in metrics and "cluster_cpu" in metrics["historical"]:
            cluster_cpu_history = metrics["historical"]["cluster_cpu"]
            if len(cluster_cpu_history) > 12:  # Need enough data points
                workload_prediction = predictor.analyze_workload(cluster_cpu_history)
        
        # Generate recommendations
        recommender = RecommendationEngine()
        recommendations = recommender.generate_recommendations(
            metrics=metrics,
            carbon_data=carbon_data,
            workload_prediction=workload_prediction
        )
        
        # Update cache
        cache["last_metrics"] = metrics
        cache["last_carbon_data"] = carbon_data
        cache["last_workload_prediction"] = workload_prediction
        cache["last_recommendations"] = recommendations
        cache["last_update_time"] = datetime.now().isoformat()
        
        return {
            "metrics": metrics,
            "carbon_data": carbon_data,
            "workload_prediction": workload_prediction,
            "recommendations": recommendations
        }
    
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return {"error": str(e)}


def format_bytes(size_bytes: float) -> str:
    """
    Format bytes as human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.23 MB")
    """
    if size_bytes < 1024:
        return f"{size_bytes:.2f} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 ** 3:
        return f"{size_bytes / (1024 ** 2):.2f} MB"
    else:
        return f"{size_bytes / (1024 ** 3):.2f} GB"


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a summary of collected metrics.
    
    Args:
        metrics: Dictionary of metrics from MetricsCollector
    """
    if not metrics:
        console.print("[bold red]No metrics data available[/bold red]")
        return
    
    console.print(Panel("[bold]Cluster Metrics Summary[/bold]", style="green"))
    
    # Current usage stats
    current_cpu = metrics.get("current", {}).get("cpu_usage", {})
    current_memory = metrics.get("current", {}).get("memory_usage", {})
    
    if current_cpu and current_memory:
        total_cpu_usage = sum(current_cpu.values())
        total_memory_usage = sum(current_memory.values())
        pod_count = len(set(current_cpu.keys()) | set(current_memory.keys()))
        
        console.print(f"[bold]Current Usage:[/bold]")
        console.print(f"Total CPU: {total_cpu_usage:.2f} cores")
        console.print(f"Total Memory: {format_bytes(total_memory_usage)}")
        console.print(f"Pod Count: {pod_count}")
        
        # Node capacity
        node_capacity = metrics.get("node_capacity", {})
        if node_capacity:
            total_cpu_capacity = sum(node["cpu"] for node in node_capacity.values())
            total_memory_capacity = sum(node["memory"] for node in node_capacity.values())
            node_count = len(node_capacity)
            
            cpu_util_pct = (total_cpu_usage / total_cpu_capacity) * 100 if total_cpu_capacity > 0 else 0
            memory_util_pct = (total_memory_usage / total_memory_capacity) * 100 if total_memory_capacity > 0 else 0
            
            console.print(f"\n[bold]Cluster Capacity:[/bold]")
            console.print(f"Total CPU: {total_cpu_capacity:.2f} cores")
            console.print(f"Total Memory: {format_bytes(total_memory_capacity)}")
            console.print(f"Node Count: {node_count}")
            
            # Utilization percentages
            console.print(f"\n[bold]Utilization:[/bold]")
            
            # Color-code utilization
            cpu_color = "green" if cpu_util_pct < 50 else "yellow" if cpu_util_pct < 80 else "red"
            memory_color = "green" if memory_util_pct < 50 else "yellow" if memory_util_pct < 80 else "red"
            
            console.print(f"CPU Utilization: [{cpu_color}]{cpu_util_pct:.1f}%[/{cpu_color}]")
            console.print(f"Memory Utilization: [{memory_color}]{memory_util_pct:.1f}%[/{memory_color}]")


def print_carbon_intensity(carbon_data: Dict[str, Any]) -> None:
    """
    Print carbon intensity information.
    
    Args:
        carbon_data: Dictionary of carbon intensity data
    """
    if not carbon_data:
        console.print("[bold red]No carbon intensity data available[/bold red]")
        return
    
    console.print(Panel("[bold]Carbon Intensity Data[/bold]", style="green"))
    
    current_intensity = carbon_data.get("current_intensity")
    region = carbon_data.get("region", "Unknown")
    
    if current_intensity is not None:
        # Color code based on intensity (rough thresholds)
        intensity_color = "green" if current_intensity < 200 else "yellow" if current_intensity < 400 else "red"
        
        console.print(f"Region: {region}")
        console.print(f"Current Carbon Intensity: [{intensity_color}]{current_intensity:.1f} gCO2eq/kWh[/{intensity_color}]")
        
        # Show best time if available
        best_time = carbon_data.get("analysis", {}).get("best_time_window", {})
        if best_time:
            best_time_start = best_time.get("start_time")
            best_intensity = best_time.get("intensity")
            
            if best_time_start and best_intensity:
                console.print(f"\n[bold]Best Time for Intensive Jobs:[/bold]")
                console.print(f"Time: {best_time_start}")
                console.print(f"Expected Intensity: {best_intensity:.1f} gCO2eq/kWh")
                
                if best_intensity < current_intensity:
                    reduction_pct = ((current_intensity - best_intensity) / current_intensity) * 100
                    console.print(f"Potential Carbon Reduction: {reduction_pct:.1f}%")


def print_workload_prediction(prediction: Dict[str, Any]) -> None:
    """
    Print workload prediction information.
    
    Args:
        prediction: Dictionary of workload prediction data
    """
    if not prediction or prediction.get("status") != "success":
        console.print("[bold red]No workload prediction available[/bold red]")
        return
    
    console.print(Panel("[bold]Workload Prediction[/bold]", style="green"))
    
    current = prediction.get("current_value", 0)
    avg = prediction.get("average_value", 0)
    peak = prediction.get("peak_value", 0)
    
    console.print(f"Current CPU Usage: {current:.2f} cores")
    console.print(f"Average Usage: {avg:.2f} cores")
    console.print(f"Peak Usage: {peak:.2f} cores")
    
    # Show prediction
    analysis = prediction.get("analysis", {})
    trend = analysis.get("trend", "stable")
    spike_detected = analysis.get("spike_detected", False)
    spike_pct = analysis.get("spike_percentage", 0)
    
    trend_color = "yellow" if trend == "increasing" else "green" if trend == "decreasing" else "blue"
    console.print(f"\n[bold]Predicted Trend: [{trend_color}]{trend.capitalize()}[/{trend_color}][/bold]")
    
    if spike_detected:
        console.print(f"[bold yellow]Spike Detected! Expected increase of {spike_pct:.1f}%[/bold yellow]")


def background_data_collection(app_state: Dict[str, Any]) -> None:
    """
    Background task for periodic data collection.
    
    Args:
        app_state: Application state dictionary
    """
    config = get_config_from_env()
    
    async def collection_task():
        while True:
            logger.info("Running scheduled data collection...")
            result = await collect_all_data(config)
            app_state["last_collection"] = datetime.now()
            app_state["last_result"] = result
            
            # Sleep until next collection
            await asyncio.sleep(config["update_interval"])
    
    # Create event loop if not exists
    if not asyncio.get_event_loop().is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(collection_task())
    else:
        asyncio.create_task(collection_task())


#
# CLI Commands
#

@cli.command("analyze")
def analyze_command(
    prometheus_url: str = typer.Option(
        None, "--prometheus-url", "-p", help="Prometheus API URL"
    ),
    carbon_api_key: str = typer.Option(
        None, "--carbon-api-key", "-k", help="Carbon intensity API key"
    ),
    region: str = typer.Option(
        None, "--region", "-r", help="Region code for carbon intensity"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Save recommendations to file"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show detailed output"
    ),
    use_mock: bool = typer.Option(
        False, "--use-mock", "-m", help="Use mock data instead of real metrics"
    ),
):
    """
    Analyze Kubernetes resources and generate optimization recommendations.
    """
    with console.status("[bold green]Collecting metrics and generating recommendations..."):
        # Get configuration
        config = get_config_from_env()
        
        # Override with command line args if provided
        if prometheus_url:
            config["prometheus_url"] = prometheus_url
        if carbon_api_key:
            config["carbon_api_key"] = carbon_api_key
        if region:
            config["region"] = region
        
        # Set use_mock in config
        config["use_mock"] = use_mock
        if use_mock:
            console.print("[bold blue]Using mock data for analysis[/bold blue]")
        
        # Run analysis
        result = asyncio.run(collect_all_data(config))
    
    if "error" in result:
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        sys.exit(1)
    
    # Get components from result
    metrics = result["metrics"]
    carbon_data = result["carbon_data"]
    prediction = result["workload_prediction"]
    recommendations = result["recommendations"]
    
    # Print results
    if verbose:
        print_metrics_summary(metrics)
        console.print()
        print_carbon_intensity(carbon_data)
        console.print()
        print_workload_prediction(prediction)
        console.print()
    
    # Always print recommendations
    recommender = RecommendationEngine()
    recs_text = recommender.format_recommendations_text(recommendations)
    rprint(recs_text)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            # Save the recommendations as formatted text
            f.write(recs_text)
        console.print(f"\n[bold green]Recommendations saved to {output_file}[/bold green]")


@cli.command("metrics")
def metrics_command(
    prometheus_url: str = typer.Option(
        None, "--prometheus-url", "-p", help="Prometheus API URL"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Save metrics to JSON file"
    ),
):
    """
    Collect and display cluster metrics.
    """
    with console.status("[bold green]Collecting metrics..."):
        # Get configuration
        config = get_config_from_env()
        
        # Override with command line args if provided
        if prometheus_url:
            config["prometheus_url"] = prometheus_url
        
        # Collect metrics
        metrics_collector = MetricsCollector(config["prometheus_url"])
        metrics = metrics_collector.collect_all_metrics()
    
    if not metrics:
        console.print("[bold red]Error: Failed to collect metrics[/bold red]")
        sys.exit(1)
    
    # Print metrics summary
    print_metrics_summary(metrics)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        console.print(f"\n[bold green]Metrics saved to {output_file}[/bold green]")


@cli.command("carbon")
def carbon_command(
    api_key: str = typer.Option(
        None, "--api-key", "-k", help="Carbon intensity API key"
    ),
    region: str = typer.Option(
        None, "--region", "-r", help="Region code for carbon intensity"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Save carbon data to JSON file"
    ),
):
    """
    Fetch and display current carbon intensity data.
    """
    with console.status("[bold green]Fetching carbon intensity data..."):
        # Get configuration
        config = get_config_from_env()
        
        # Override with command line args if provided
        if api_key:
            config["carbon_api_key"] = api_key
        if region:
            config["region"] = region
        
        # Get carbon intensity data
        carbon_fetcher = get_carbon_fetcher(config["carbon_api_key"], config["region"])
        carbon_data = carbon_fetcher.get_carbon_data()
    
    if not carbon_data:
        console.print("[bold red]Error: Failed to fetch carbon intensity data[/bold red]")
        sys.exit(1)
    
    # Print carbon intensity
    print_carbon_intensity(carbon_data)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(carbon_data, f, indent=2)
        console.print(f"\n[bold green]Carbon data saved to {output_file}[/bold green]")


@cli.command("predict")
def predict_command(
    prometheus_url: str = typer.Option(
        None, "--prometheus-url", "-p", help="Prometheus API URL"
    ),
    hours_back: int = typer.Option(
        6, "--hours-back", "-h", help="Hours of historical data to use"
    ),
    output_file: Path = typer.Option(
        None, "--output", "-o", help="Save prediction to JSON file"
    ),
):
    """
    Predict future workload patterns based on historical data.
    """
    with console.status("[bold green]Collecting data and generating prediction..."):
        # Get configuration
        config = get_config_from_env()
        
        # Override with command line args if provided
        if prometheus_url:
            config["prometheus_url"] = prometheus_url
        
        # Collect metrics
        metrics_collector = MetricsCollector(config["prometheus_url"])
        cluster_cpu = metrics_collector.get_cluster_cpu_usage(hours=hours_back)
        
        if not cluster_cpu:
            console.print("[bold red]Error: Failed to collect historical CPU data[/bold red]")
            sys.exit(1)
        
        # Make prediction
        predictor = WorkloadPredictor()
        result = predictor.analyze_workload(cluster_cpu)
    
    if not result or result.get("status") != "success":
        error_msg = result.get("message", "Unknown error") if result else "Failed to generate prediction"
        console.print(f"[bold red]Error: {error_msg}[/bold red]")
        sys.exit(1)
    
    # Print prediction
    print_workload_prediction(result)
    
    # Save to file if requested
    if output_file:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=2)
        console.print(f"\n[bold green]Prediction saved to {output_file}[/bold green]")


@cli.command("serve")
def serve_command(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind the server to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind the server to"),
    prometheus_url: str = typer.Option(
        None, "--prometheus-url", help="Prometheus API URL"
    ),
    carbon_api_key: str = typer.Option(
        None, "--carbon-api-key", help="Carbon intensity API key"
    ),
    region: str = typer.Option(
        None, "--region", help="Region code for carbon intensity"
    ),
    update_interval: int = typer.Option(
        300, "--update-interval", "-i", help="Data update interval in seconds"
    ),
):
    """
    Start the GreenOps Agent API server.
    """
    # Get configuration
    config = get_config_from_env()
    
    # Override with command line args if provided
    if prometheus_url:
        config["prometheus_url"] = prometheus_url
    if carbon_api_key:
        config["carbon_api_key"] = carbon_api_key
    if region:
        config["region"] = region
    if update_interval:
        config["update_interval"] = update_interval
    
    # Update app state
    app_state = {
        "config": config,
        "last_collection": None,
        "last_result": None
    }
    
    # Set app state
    api.state.app_state = app_state
    
    # Start server
    console.print(f"[bold green]Starting GreenOps Agent API server on {host}:{port}[/bold green]")
    console.print(f"Prometheus URL: {config['prometheus_url']}")
    console.print(f"Region: {config['region']}")
    console.print(f"Update interval: {config['update_interval']} seconds")
    
    uvicorn.run(api, host=host, port=port)


#
# FastAPI routes
#

@api.on_event("startup")
async def startup_event():
    """Initialize the API server on startup."""
    logger.info("Starting GreenOps Agent API server")
    
    # Start background task for data collection
    background_tasks = BackgroundTasks()
    background_tasks.add_task(background_data_collection, api.state.app_state)


@api.get("/")
async def root():
    """Root endpoint returning basic information."""
    return {
        "name": "GreenOps Agent API",
        "version": "0.1.0",
        "description": "API for optimizing Kubernetes workloads for energy efficiency"
    }


@api.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@api.get("/metrics")
async def get_metrics(refresh: bool = False):
    """
    Get current cluster metrics.
    
    Args:
        refresh: If true, fetch new data instead of using cache
    """
    app_state = api.state.app_state
    
    if refresh or app_state["last_result"] is None:
        # Force refresh
        result = await collect_all_data(app_state["config"])
        app_state["last_result"] = result
        app_state["last_collection"] = datetime.now()
    else:
        result = app_state["last_result"]
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result["metrics"]


@api.get("/carbon")
async def get_carbon(refresh: bool = False):
    """
    Get current carbon intensity data.
    
    Args:
        refresh: If true, fetch new data instead of using cache
    """
    app_state = api.state.app_state
    
    if refresh or app_state["last_result"] is None:
        # Force refresh
        result = await collect_all_data(app_state["config"])
        app_state["last_result"] = result
        app_state["last_collection"] = datetime.now()
    else:
        result = app_state["last_result"]
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result["carbon_data"]


@api.get("/predict")
async def get_prediction(refresh: bool = False):
    """
    Get workload prediction.
    
    Args:
        refresh: If true, fetch new data instead of using cache
    """
    app_state = api.state.app_state
    
    if refresh or app_state["last_result"] is None:
        # Force refresh
        result = await collect_all_data(app_state["config"])
        app_state["last_result"] = result
        app_state["last_collection"] = datetime.now()
    else:
        result = app_state["last_result"]
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result["workload_prediction"]


@api.get("/recommendations")
async def get_recommendations(refresh: bool = False):
    """
    Get optimization recommendations.
    
    Args:
        refresh: If true, fetch new data and generate new recommendations
    """
    app_state = api.state.app_state
    
    if refresh or app_state["last_result"] is None:
        # Force refresh
        result = await collect_all_data(app_state["config"])
        app_state["last_result"] = result
        app_state["last_collection"] = datetime.now()
    else:
        result = app_state["last_result"]
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result["recommendations"]


@api.get("/report")
async def get_report(refresh: bool = False):
    """
    Get a formatted text report of recommendations.
    
    Args:
        refresh: If true, fetch new data and generate new recommendations
    """
    app_state = api.state.app_state
    
    if refresh or app_state["last_result"] is None:
        # Force refresh
        result = await collect_all_data(app_state["config"])
        app_state["last_result"] = result
        app_state["last_collection"] = datetime.now()
    else:
        result = app_state["last_result"]
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    # Format recommendations as text
    recommender = RecommendationEngine()
    text = recommender.format_recommendations_text(result["recommendations"])
    
    return {"report": text}


# Set up module entry point for CLI and API
def run_cli():
    """Entry point for CLI."""
    cli()


def run_api(host="0.0.0.0", port=8000):
    """Entry point for API server."""
    uvicorn.run(api, host=host, port=port)


if __name__ == "__main__":
    # Run CLI by default when script is executed directly
    cli()