# GreenOps Agent

![Build](https://img.shields.io/github/actions/workflow/status/ariatajeri/greenops-agent/python-app.yml?branch=main)
![PyPI](https://img.shields.io/pypi/v/greenops-agent)
![License](https://img.shields.io/github/license/ariatajeri/greenops-agent)
![Python](https://img.shields.io/badge/python-3.10+-blue)

## Why GreenOps Agent?

- 🌎 Combat climate change by optimizing energy usage in cloud environments.
- 🚀 Boost Kubernetes cluster efficiency without sacrificing performance.
- 🛠️ Easy integration with existing monitoring stacks like Prometheus.
- 🧠 Empower DevOps teams with ML-driven predictive insights.
- ⚡ Instant impact: deploy in minutes, save energy immediately.

A Kubernetes agent for optimizing workloads for energy efficiency and lower carbon footprint.

![GreenOps Logo](docs/images/greenops-logo.png)

## Overview

GreenOps Agent is a software solution designed to analyze Kubernetes workloads for energy efficiency and help reduce carbon footprint. It collects metrics from Prometheus, correlates them with carbon intensity data, and generates resource optimization recommendations using machine learning predictions.

### Key Features

- **Resource Optimization**: Analyzes CPU and memory usage to identify over-provisioned and under-provisioned workloads
- **Carbon Awareness**: Integrates with carbon intensity APIs to recommend scheduling during low-carbon periods
- **Workload Prediction**: Uses machine learning (LSTM) to predict future resource needs and optimize proactively
- **Multiple Interfaces**: Provides both CLI and REST API access to recommendations
- **Kubernetes Integration**: Deploys easily into Kubernetes clusters with Helm
- **Mock Data Mode**: Test and develop without requiring actual Prometheus or carbon intensity data sources
- **Sample Output Provided**: Quickly see how the system works even without setup

## Architecture

The GreenOps Agent consists of several components:

1. **Metrics Collector**: Fetches resource utilization data from Prometheus
2. **Carbon Intensity Fetcher**: Gets current and predicted carbon intensity from electricity grid APIs
3. **Workload Predictor**: Uses ML to predict future workload patterns
4. **Recommendation Engine**: Generates optimization suggestions based on all data
5. **Service Layer**: Provides CLI and API interfaces to the system

![Architecture Diagram](docs/images/architecture.png)

## Example Output

```bash
$ greenops-agent analyze --use-mock

[INFO] Cluster Carbon Intensity: 502 gCO₂/kWh
[INFO] Pod recommendation:
 - Service: payment-api
   Current CPU request: 1000m
   Observed Peak Usage: 250m
   Suggested CPU request: 300m
 - Service: analytics-worker
   Current Memory limit: 2Gi
   Observed Peak: 600Mi
   Suggested limit: 800Mi

[ALERT] Predicted CPU spike in next 30 minutes. Consider scaling critical services.

[NOTE] High carbon intensity detected! Delay non-critical workloads if possible.
```

## Installation

### Prerequisites

- Kubernetes cluster with Prometheus installed
- Python 3.10+ (for local development)
- Docker (for containerized deployment)

### Using Helm

```bash
# Add the GreenOps Helm repository (when available)
# helm repo add greenops https://charts.greenops.io

# Install the chart with default values
helm install greenops-agent ./deployment/helm/greenops-agent

# Install with custom values
helm install greenops-agent ./deployment/helm/greenops-agent \
  --set config.prometheusUrl=http://prometheus.monitoring:9090 \
  --set config.region=US-CA \
  --set config.carbonApiKey=your-api-key-here
```

### Using Kubernetes Manifests

```bash
# Edit configuration in deployment/kubernetes/config.yaml
# Set your prometheus URL and carbon intensity API key

# Apply resources
kubectl apply -f deployment/kubernetes/config.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### Using Docker

```bash
# Build the Docker image
docker build -t greenops-agent:latest .

# Run the container in API server mode
docker run -p 8000:8000 \
  -e PROMETHEUS_URL=http://prometheus:9090 \
  -e REGION=US-CA \
  -e CARBON_API_KEY=your-api-key-here \
  greenops-agent:latest
```

## Usage

### CLI Mode

```bash
# Analyze resources and generate recommendations
greenops-agent analyze --prometheus-url http://prometheus:9090

# Use mock data for testing without a real Prometheus server
greenops-agent analyze --use-mock --verbose

# Show cluster metrics
greenops-agent metrics

# Show carbon intensity data 
greenops-agent carbon --region US-CA --api-key your-api-key

# Predict future workload
greenops-agent predict --hours-back 24
```

### Mock Data Mode

The GreenOps Agent supports a mock data mode for testing and development without requiring an actual Prometheus server or carbon intensity data source. This is particularly useful for:

- Evaluating the tool without setting up Kubernetes and Prometheus
- Development and testing of new features
- Demonstrations and presentations

To use mock data mode:

```bash
# Run the analyze command with mock data
greenops-agent analyze --use-mock --verbose

# Run the API server with mock data
GREENOPS_USE_MOCK=true greenops-agent serve
```

The mock data generator simulates:

- Cluster metrics (CPU, memory, and power consumption)
- Node capacity and utilization patterns
- Pod resource usage with realistic patterns
- Carbon intensity data with daily variations
- Workload patterns for different application types

### API Mode

```bash
# Start the API server
greenops-agent serve
```

#### API Endpoints

- `GET /metrics` - Get current cluster metrics
- `GET /carbon` - Get carbon intensity data
- `GET /predict` - Get workload predictions
- `GET /recommendations` - Get resource optimization recommendations
- `GET /report` - Get a formatted text report of recommendations

## Configuration

### Environment Variables

- `PROMETHEUS_URL` - URL for Prometheus API
- `CARBON_API_KEY` - API key for carbon intensity data
- `REGION` - Region code for carbon intensity data
- `UPDATE_INTERVAL` - Data update interval in seconds (default: 300)
- `DEBUG` - Enable debug logging (default: false)
- `GREENOPS_USE_MOCK` - Set to "true" to use mock data instead of real metrics (for testing)

### Supported Regions

The carbon intensity module supports the following regions:

- `US-CA` - California
- `US-TX` - Texas
- `DE` - Germany
- `FR` - France
- `GB` - Great Britain
- `DEFAULT` - Fallback with static data

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/greenops-agent.git
cd greenops-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_metrics.py
```

### Testing with Mock Data

The mock data functionality allows you to test the application without external dependencies:

```bash
# Run with mock data
greenops-agent analyze --use-mock

# Save mock data recommendations to file
greenops-agent analyze --use-mock --output recommendations.txt

# Show verbose output with mock data
greenops-agent analyze --use-mock --verbose
```

This is useful for:
- CI/CD pipelines where real infrastructure isn't available
- Quick debugging without cluster access
- Demo environments and presentations

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Electricity Maps API for carbon intensity data
- Prometheus for metrics collection
- TensorFlow for ML prediction capabilities