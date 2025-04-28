"""
Metrics collection module for GreenOps Agent.

This module is responsible for collecting metrics from Prometheus regarding
Kubernetes resource utilization and workload patterns.
"""

import logging
import requests
import time
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

logger = logging.getLogger(__name__)

# Import mock data module if available
MOCK_DATA_AVAILABLE = False
try:
    from greenops_agent.mockdata import prometheus as mock_prometheus
    MOCK_DATA_AVAILABLE = True
except ImportError:
    try:
        # Try with src prefix for development environment
        from src.greenops_agent.mockdata import prometheus as mock_prometheus
        MOCK_DATA_AVAILABLE = True
    except ImportError:
        logger.warning("Mock data module not available")
        MOCK_DATA_AVAILABLE = False

# Make create_mock_prometheus_client available
if MOCK_DATA_AVAILABLE:
    try:
        create_mock_prometheus_client = mock_prometheus.create_mock_prometheus_client
    except (ImportError, AttributeError):
        logger.warning("create_mock_prometheus_client function not available")
        MOCK_DATA_AVAILABLE = False


class MetricsCollector:
    """
    Collects metrics from Prometheus regarding Kubernetes resource utilization.
    
    Attributes:
        prometheus_url: Base URL for the Prometheus API
        timeout: Request timeout in seconds
        use_mock_data: Whether to use mock data instead of real Prometheus
    """
    
    def __init__(self, prometheus_url: Optional[str] = None, timeout: int = 10, use_mock_data: bool = False):
        """
        Initialize the metrics collector with Prometheus connection details.
        
        Args:
            prometheus_url: Base URL for the Prometheus API (e.g., http://prometheus:9090)
            timeout: Request timeout in seconds
            use_mock_data: Whether to use mock data instead of real Prometheus
        """
        self.use_mock_data = use_mock_data or not prometheus_url
        
        if self.use_mock_data:
            if not MOCK_DATA_AVAILABLE:
                logger.warning("Mock data requested but module not available")
            else:
                logger.info("Using mock Prometheus data for metrics")
            self.prometheus_url = None
        else:
            if not prometheus_url:
                raise ValueError("Prometheus URL must be provided when not using mock data")
            self.prometheus_url = prometheus_url.rstrip('/')
            logger.info(f"Initialized MetricsCollector with Prometheus URL: {prometheus_url}")
            
        self.timeout = timeout
        self._mock_data_cache = None
    
    def _get_mock_data(self) -> Dict[str, Any]:
        """Get or create mock metrics data."""
        if not MOCK_DATA_AVAILABLE:
            return {}
            
        if self._mock_data_cache is None:
            # Create a mock client instead of calling get_mock_prometheus_data
            mock_client = mock_prometheus.create_mock_prometheus_client("mock-cluster")
            # Query metrics data for the last 24 hours with 1 hour steps
            self._mock_data_cache = mock_client.query_metrics(
                metric_names=["cpu", "memory", "power", "network"],
                start_time=time.time() - 86400,  # 24 hours ago
                end_time=time.time(),
                step_seconds=3600  # 1 hour intervals
            )
            
        return self._mock_data_cache

    def query(self, query: str, time: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Execute a PromQL query at a specific instant in time.
        
        Args:
            query: PromQL query string
            time: Optional time for the query (defaults to now)
            
        Returns:
            Query result as a dictionary
            
        Raises:
            requests.RequestException: If the request to Prometheus fails
        """
        if self.use_mock_data:
            # For mock data, we just return a successful empty result
            # Specific mock data is handled in the specialized methods
            return {
                'status': 'success',
                'data': {
                    'resultType': 'vector',
                    'result': []
                }
            }
            
        params = {'query': query}
        if time:
            params['time'] = time.timestamp()
        
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error querying Prometheus: {e}")
            raise
    
    def query_range(
        self,
        query: str,
        start: datetime,
        end: datetime,
        step: str = "5m"
    ) -> Dict[str, Any]:
        """
        Execute a PromQL query over a range of time.
        
        Args:
            query: PromQL query string
            start: Start time for the range query
            end: End time for the range query
            step: Resolution step (e.g., "5m" for 5 minutes)
            
        Returns:
            Query result as a dictionary
            
        Raises:
            requests.RequestException: If the request to Prometheus fails
        """
        if self.use_mock_data:
            # For mock data, we just return a successful empty result
            # Specific mock data is handled in the specialized methods
            return {
                'status': 'success',
                'data': {
                    'resultType': 'matrix',
                    'result': []
                }
            }
            
        try:
            response = requests.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params={
                    'query': query,
                    'start': start.timestamp(),
                    'end': end.timestamp(),
                    'step': step
                },
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error querying Prometheus range: {e}")
            raise
    
    def get_current_cpu_usage(self) -> Dict[str, float]:
        """
        Get current CPU usage for all pods.
        
        Returns:
            Dictionary mapping pod names to CPU usage (in cores)
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            pod_cpu = {}
            
            for pod_id, pod_info in mock_data.get('pods', {}).items():
                # Get the last CPU usage value
                cpu_values = pod_info.get('resources', {}).get('cpu', {}).get('usage', [])
                if cpu_values:
                    # Get the most recent value
                    pod_cpu[pod_id] = cpu_values[-1][1]
                else:
                    pod_cpu[pod_id] = 0.0
                    
            return pod_cpu
            
        # Query: Rate of CPU usage over 5min, aggregated by pod
        query = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (pod)'
        result = self.query(query)
        
        pod_cpu = {}
        if result['status'] == 'success':
            for item in result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                cpu_value = float(item.get('value', [0, '0'])[1])
                pod_cpu[pod_name] = cpu_value
                
        return pod_cpu
    
    def get_current_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage for all pods.
        
        Returns:
            Dictionary mapping pod names to memory usage (in bytes)
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            pod_memory = {}
            
            for pod_id, pod_info in mock_data.get('pods', {}).items():
                # Get the last memory usage value
                memory_values = pod_info.get('resources', {}).get('memory', {}).get('usage_mb', [])
                if memory_values:
                    # Get the most recent value and convert from MB to bytes
                    pod_memory[pod_id] = memory_values[-1][1] * 1024 * 1024
                else:
                    pod_memory[pod_id] = 0.0
                    
            return pod_memory
            
        # Query: Current memory usage by pod
        query = 'sum(container_memory_usage_bytes{container!="POD",container!=""}) by (pod)'
        result = self.query(query)
        
        pod_memory = {}
        if result['status'] == 'success':
            for item in result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                # Memory value in bytes
                memory_value = float(item.get('value', [0, '0'])[1])
                pod_memory[pod_name] = memory_value
                
        return pod_memory
    
    def get_resource_requests(self) -> Dict[str, Dict[str, float]]:
        """
        Get CPU and memory requests for all pods.
        
        Returns:
            Dictionary mapping pod names to their resource requests
            {pod_name: {'cpu': cpu_request, 'memory': memory_request}}
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            resources = {}
            
            for pod_id, pod_info in mock_data.get('pods', {}).items():
                cpu_request = pod_info.get('resources', {}).get('cpu', {}).get('request', 0.0)
                # Memory in the mock data is in MB, convert to bytes for consistency with Prometheus
                memory_mb = pod_info.get('resources', {}).get('memory', {}).get('request_mb', 0.0)
                memory_bytes = memory_mb * 1024 * 1024
                
                resources[pod_id] = {'cpu': cpu_request, 'memory': memory_bytes}
                    
            return resources
            
        # Query for CPU requests
        cpu_query = 'sum(kube_pod_container_resource_requests{resource="cpu"}) by (pod)'
        cpu_result = self.query(cpu_query)
        
        # Query for Memory requests
        mem_query = 'sum(kube_pod_container_resource_requests{resource="memory"}) by (pod)'
        mem_result = self.query(mem_query)
        
        resources = {}
        
        # Process CPU requests
        if cpu_result['status'] == 'success':
            for item in cpu_result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                if pod_name not in resources:
                    resources[pod_name] = {'cpu': 0, 'memory': 0}
                resources[pod_name]['cpu'] = float(item.get('value', [0, '0'])[1])
        
        # Process Memory requests
        if mem_result['status'] == 'success':
            for item in mem_result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                if pod_name not in resources:
                    resources[pod_name] = {'cpu': 0, 'memory': 0}
                resources[pod_name]['memory'] = float(item.get('value', [0, '0'])[1])
        
        return resources
    
    def get_resource_limits(self) -> Dict[str, Dict[str, float]]:
        """
        Get CPU and memory limits for all pods.
        
        Returns:
            Dictionary mapping pod names to their resource limits
            {pod_name: {'cpu': cpu_limit, 'memory': memory_limit}}
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            resources = {}
            
            for pod_id, pod_info in mock_data.get('pods', {}).items():
                cpu_limit = pod_info.get('resources', {}).get('cpu', {}).get('limit', 0.0)
                # Memory in the mock data is in MB, convert to bytes for consistency with Prometheus
                memory_mb = pod_info.get('resources', {}).get('memory', {}).get('limit_mb', 0.0)
                memory_bytes = memory_mb * 1024 * 1024
                
                resources[pod_id] = {'cpu': cpu_limit, 'memory': memory_bytes}
                    
            return resources
            
        # Query for CPU limits
        cpu_query = 'sum(kube_pod_container_resource_limits{resource="cpu"}) by (pod)'
        cpu_result = self.query(cpu_query)
        
        # Query for Memory limits
        mem_query = 'sum(kube_pod_container_resource_limits{resource="memory"}) by (pod)'
        mem_result = self.query(mem_query)
        
        resources = {}
        
        # Process CPU limits
        if cpu_result['status'] == 'success':
            for item in cpu_result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                if pod_name not in resources:
                    resources[pod_name] = {'cpu': 0, 'memory': 0}
                resources[pod_name]['cpu'] = float(item.get('value', [0, '0'])[1])
        
        # Process Memory limits
        if mem_result['status'] == 'success':
            for item in mem_result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                if pod_name not in resources:
                    resources[pod_name] = {'cpu': 0, 'memory': 0}
                resources[pod_name]['memory'] = float(item.get('value', [0, '0'])[1])
        
        return resources
    
    def get_historical_cpu_usage(
        self,
        hours: int = 6,
        step: str = "5m"
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Get historical CPU usage for the past specified hours.
        
        Args:
            hours: Number of hours to look back
            step: Resolution step (e.g., "5m" for 5 minutes)
            
        Returns:
            Dictionary mapping pod names to list of (timestamp, value) tuples
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            pod_cpu_history = {}
            
            for pod_id, pod_info in mock_data.get('pods', {}).items():
                # Get CPU usage time series
                cpu_values = pod_info.get('resources', {}).get('cpu', {}).get('usage', [])
                if cpu_values:
                    pod_cpu_history[pod_id] = cpu_values
                
            return pod_cpu_history
            
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        query = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m])) by (pod)'
        result = self.query_range(query, start_time, end_time, step)
        
        pod_cpu_history = {}
        if result['status'] == 'success':
            for item in result.get('data', {}).get('result', []):
                pod_name = item['metric'].get('pod', 'unknown')
                
                # Each value is [timestamp, value]
                values = [(float(v[0]), float(v[1])) for v in item.get('values', [])]
                pod_cpu_history[pod_name] = values
                
        return pod_cpu_history
    
    def get_cluster_cpu_usage(self, hours: int = 6, step: str = "5m") -> List[Tuple[float, float]]:
        """
        Get historical total cluster CPU usage.
        
        Args:
            hours: Number of hours to look back
            step: Resolution step (e.g., "5m" for 5 minutes)
            
        Returns:
            List of (timestamp, cpu_cores) tuples
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            
            # Aggregate CPU usage across all nodes
            cluster_cpu = {}
            
            # First collect all timestamps
            for node_id, node_info in mock_data.get('nodes', {}).items():
                cpu_values = node_info.get('usage', {}).get('cpu', [])
                for timestamp, value in cpu_values:
                    if timestamp not in cluster_cpu:
                        cluster_cpu[timestamp] = 0
                    cluster_cpu[timestamp] += value
            
            # Convert to sorted list of tuples
            result = sorted(cluster_cpu.items())
            return result
            
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)
        
        # Query total CPU usage across all pods
        query = 'sum(rate(container_cpu_usage_seconds_total{container!="POD",container!=""}[5m]))'
        result = self.query_range(query, start_time, end_time, step)
        
        cpu_usage = []
        if result['status'] == 'success' and result.get('data', {}).get('result'):
            # There should be just one result for the sum query
            values = result['data']['result'][0].get('values', [])
            cpu_usage = [(float(v[0]), float(v[1])) for v in values]
                
        return cpu_usage
    
    def get_node_capacity(self) -> Dict[str, Dict[str, float]]:
        """
        Get capacity information for all nodes.
        
        Returns:
            Dictionary mapping node names to their capacity
            {node_name: {'cpu': cpu_capacity, 'memory': memory_capacity}}
        """
        if self.use_mock_data and MOCK_DATA_AVAILABLE:
            mock_data = self._get_mock_data()
            capacity = {}
            
            for node_id, node_info in mock_data.get('nodes', {}).items():
                cpu_capacity = node_info.get('capacity', {}).get('cpu', 0.0)
                # Memory in the mock data is in MB, convert to bytes for consistency with Prometheus
                memory_mb = node_info.get('capacity', {}).get('memory_mb', 0.0)
                memory_bytes = memory_mb * 1024 * 1024
                
                capacity[node_id] = {'cpu': cpu_capacity, 'memory': memory_bytes}
                    
            return capacity
            
        # Query for CPU capacity
        cpu_query = 'kube_node_status_capacity{resource="cpu"}'
        cpu_result = self.query(cpu_query)
        
        # Query for Memory capacity
        mem_query = 'kube_node_status_capacity{resource="memory"}'
        mem_result = self.query(mem_query)
        
        capacity = {}
        
        # Process CPU capacity
        if cpu_result['status'] == 'success':
            for item in cpu_result.get('data', {}).get('result', []):
                node_name = item['metric'].get('node', 'unknown')
                if node_name not in capacity:
                    capacity[node_name] = {'cpu': 0, 'memory': 0}
                capacity[node_name]['cpu'] = float(item.get('value', [0, '0'])[1])
        
        # Process Memory capacity
        if mem_result['status'] == 'success':
            for item in mem_result.get('data', {}).get('result', []):
                node_name = item['metric'].get('node', 'unknown')
                if node_name not in capacity:
                    capacity[node_name] = {'cpu': 0, 'memory': 0}
                # Memory value is typically in bytes
                capacity[node_name]['memory'] = float(item.get('value', [0, '0'])[1])
        
        return capacity
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """
        Collect all relevant metrics in one call.
        
        Returns:
            Dictionary containing various metrics and resource stats
        """
        try:
            metrics = {
                'current': {
                    'cpu_usage': self.get_current_cpu_usage(),
                    'memory_usage': self.get_current_memory_usage()
                },
                'requests': self.get_resource_requests(),
                'limits': self.get_resource_limits(),
                'node_capacity': self.get_node_capacity()
            }
            
            # Also get historical data for prediction
            metrics['historical'] = {
                'cluster_cpu': self.get_cluster_cpu_usage(hours=6)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return {}