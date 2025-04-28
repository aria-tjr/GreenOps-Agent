"""
Recommendation engine module for GreenOps Agent.

This module generates resource optimization recommendations based on
collected metrics, carbon intensity data, and workload predictions.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class RecommendationEngine:
    """
    Generates resource optimization recommendations for Kubernetes workloads.
    
    The recommendation engine analyzes resource usage, requests, limits,
    and carbon intensity to provide actionable recommendations that reduce
    energy consumption and carbon footprint.
    """
    
    def __init__(
        self,
        cpu_threshold_pct: float = 0.6,
        memory_threshold_pct: float = 0.7,
        min_cpu_request: float = 0.1,  # minimum CPU request in cores
        min_memory_request: float = 128 * 1024 * 1024,  # 128 MB in bytes
        buffer_factor: float = 1.3  # safety buffer for recommendations
    ):
        """
        Initialize the recommendation engine with threshold parameters.
        
        Args:
            cpu_threshold_pct: CPU usage threshold (as a fraction of request)
            memory_threshold_pct: Memory usage threshold (as a fraction of request)
            min_cpu_request: Minimum CPU request recommendation (cores)
            min_memory_request: Minimum memory request recommendation (bytes)
            buffer_factor: Safety buffer for recommendations (multiplier)
        """
        self.cpu_threshold_pct = cpu_threshold_pct
        self.memory_threshold_pct = memory_threshold_pct
        self.min_cpu_request = min_cpu_request
        self.min_memory_request = min_memory_request
        self.buffer_factor = buffer_factor
        
        logger.info("Initialized RecommendationEngine")
    
    def generate_pod_recommendations(
        self,
        pod_name: str,
        current_cpu: float,
        current_memory: float,
        requested_cpu: float,
        requested_memory: float,
        limit_cpu: Optional[float] = None,
        limit_memory: Optional[float] = None,
        cpu_history: Optional[List[Tuple[float, float]]] = None,
        predicted_cpu: Optional[List[Tuple[datetime, float]]] = None
    ) -> Dict[str, Any]:
        """
        Generate recommendations for a single pod.
        
        Args:
            pod_name: Name of the pod
            current_cpu: Current CPU usage (cores)
            current_memory: Current memory usage (bytes)
            requested_cpu: Requested CPU resources (cores)
            requested_memory: Requested memory resources (bytes)
            limit_cpu: CPU limit, if set (cores)
            limit_memory: Memory limit, if set (bytes)
            cpu_history: Historical CPU usage if available
            predicted_cpu: Predicted future CPU usage if available
            
        Returns:
            Dictionary with recommendations
        """
        recommendations = {
            "pod_name": pod_name,
            "cpu": {},
            "memory": {},
            "reason": "",
            "priority": "low"
        }
        
        # Calculate 95th percentile of CPU usage if history available
        p95_cpu = current_cpu
        if cpu_history and len(cpu_history) > 10:
            cpu_values = [v for _, v in cpu_history]
            cpu_values.sort()
            p95_idx = int(len(cpu_values) * 0.95)
            p95_cpu = cpu_values[min(p95_idx, len(cpu_values)-1)]
        
        # Determine max predicted CPU if available
        max_predicted_cpu = None
        if predicted_cpu and predicted_cpu:
            max_predicted_cpu = max(v for _, v in predicted_cpu)
        
        # Calculate recommended CPU based on actual usage plus buffer
        safe_cpu_usage = max(p95_cpu, current_cpu, max_predicted_cpu or 0)
        recommended_cpu = max(safe_cpu_usage * self.buffer_factor, self.min_cpu_request)
        
        if requested_cpu > 0:  # Avoid division by zero
            cpu_utilization = current_cpu / requested_cpu
            
            # Check if CPU is over-provisioned
            if cpu_utilization < self.cpu_threshold_pct and recommended_cpu < requested_cpu * 0.8:
                recommendations["cpu"]["action"] = "decrease"
                recommendations["cpu"]["current_request"] = requested_cpu
                recommendations["cpu"]["recommended_request"] = recommended_cpu
                recommendations["cpu"]["utilization"] = cpu_utilization * 100  # as percentage
                recommendations["reason"] += f"CPU utilization is {cpu_utilization*100:.1f}%, which is below target threshold. "
                recommendations["priority"] = "medium"
                
            # Check if CPU is under-provisioned
            elif cpu_utilization > 0.9 or (max_predicted_cpu and max_predicted_cpu > requested_cpu):
                recommendations["cpu"]["action"] = "increase"
                recommendations["cpu"]["current_request"] = requested_cpu
                recommendations["cpu"]["recommended_request"] = recommended_cpu
                recommendations["cpu"]["utilization"] = cpu_utilization * 100
                recommendations["reason"] += f"CPU utilization is high at {cpu_utilization*100:.1f}%. "
                if max_predicted_cpu and max_predicted_cpu > requested_cpu:
                    recommendations["reason"] += f"Predicted usage ({max_predicted_cpu:.2f} cores) exceeds current request. "
                recommendations["priority"] = "high"
        
        # Calculate recommended memory based on actual usage plus buffer
        recommended_memory = max(current_memory * self.buffer_factor, self.min_memory_request)
        
        if requested_memory > 0:  # Avoid division by zero
            memory_utilization = current_memory / requested_memory
            
            # Check if memory is over-provisioned
            if memory_utilization < self.memory_threshold_pct and recommended_memory < requested_memory * 0.8:
                recommendations["memory"]["action"] = "decrease"
                recommendations["memory"]["current_request"] = requested_memory
                recommendations["memory"]["recommended_request"] = recommended_memory
                recommendations["memory"]["utilization"] = memory_utilization * 100
                recommendations["reason"] += f"Memory utilization is {memory_utilization*100:.1f}%, which is below target threshold. "
                if "priority" != "high":  # Don't downgrade from high
                    recommendations["priority"] = "medium"
                    
            # Check if memory is under-provisioned
            elif memory_utilization > 0.9:
                recommendations["memory"]["action"] = "increase"
                recommendations["memory"]["current_request"] = requested_memory
                recommendations["memory"]["recommended_request"] = recommended_memory
                recommendations["memory"]["utilization"] = memory_utilization * 100
                recommendations["reason"] += f"Memory utilization is high at {memory_utilization*100:.1f}%. "
                recommendations["priority"] = "high"
        
        # If both CPU and memory have recommendations, improve the explanation
        if "action" in recommendations["cpu"] and "action" in recommendations["memory"]:
            if recommendations["cpu"]["action"] == recommendations["memory"]["action"]:
                action = recommendations["cpu"]["action"]
                recommendations["reason"] = f"Recommend {action} both CPU and memory: {recommendations['reason']}"
        elif "action" in recommendations["cpu"]:
            action = recommendations["cpu"]["action"]
            recommendations["reason"] = f"Recommend {action} CPU: {recommendations['reason']}"
        elif "action" in recommendations["memory"]:
            action = recommendations["memory"]["action"]
            recommendations["reason"] = f"Recommend {action} memory: {recommendations['reason']}"
        
        # No recommendations needed
        if "action" not in recommendations["cpu"] and "action" not in recommendations["memory"]:
            recommendations["reason"] = "Resource utilization is within optimal range."
            recommendations["priority"] = "none"
        
        return recommendations
    
    def generate_cluster_recommendations(
        self,
        node_capacity: Dict[str, Dict[str, float]],
        current_usage: Dict[str, Dict[str, float]],
        carbon_data: Optional[Dict[str, Any]] = None,
        workload_prediction: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate cluster-level recommendations.
        
        Args:
            node_capacity: Dictionary of node capacity
            current_usage: Dictionary of aggregate resource usage
            carbon_data: Carbon intensity data if available
            workload_prediction: Workload prediction data if available
            
        Returns:
            List of cluster-level recommendations
        """
        recommendations = []
        
        # Calculate total capacity and usage
        total_cpu_capacity = sum(node["cpu"] for node in node_capacity.values())
        total_memory_capacity = sum(node["memory"] for node in node_capacity.values())
        
        total_cpu_usage = sum(pod_data.get("cpu", 0) for pod_data in current_usage.values())
        total_memory_usage = sum(pod_data.get("memory", 0) for pod_data in current_usage.values())
        
        # Calculate utilization percentages
        cpu_utilization = total_cpu_usage / total_cpu_capacity if total_cpu_capacity > 0 else 0
        memory_utilization = total_memory_usage / total_memory_capacity if total_memory_capacity > 0 else 0
        
        # Check if cluster is underutilized
        if cpu_utilization < 0.4 and memory_utilization < 0.4:
            # Count nodes
            node_count = len(node_capacity)
            if node_count > 1:
                # Estimate how many nodes could be removed
                potential_reduction = max(1, int(node_count * (1 - max(cpu_utilization, memory_utilization) / 0.7)))
                
                recommendations.append({
                    "type": "cluster_scaling",
                    "action": "scale_down",
                    "details": {
                        "current_nodes": node_count,
                        "potential_reduction": potential_reduction,
                        "cpu_utilization": cpu_utilization * 100,
                        "memory_utilization": memory_utilization * 100
                    },
                    "reason": (
                        f"Cluster is underutilized (CPU: {cpu_utilization*100:.1f}%, Memory: {memory_utilization*100:.1f}%). "
                        f"Consider scaling down by {potential_reduction} node(s) after optimizing workload resources."
                    ),
                    "priority": "high" if potential_reduction > 1 else "medium"
                })
        
        # Carbon-aware scheduling recommendations
        if carbon_data:
            current_intensity = carbon_data.get("current_intensity")
            is_high_carbon = carbon_data.get("analysis", {}).get("is_high_carbon_period", False)
            best_time = carbon_data.get("analysis", {}).get("best_time_window", {})
            
            if is_high_carbon and best_time:
                best_time_start = best_time.get("start_time")
                best_intensity = best_time.get("intensity")
                
                if best_intensity and current_intensity and best_intensity < current_intensity * 0.7:
                    recommendations.append({
                        "type": "carbon_aware_scheduling",
                        "action": "defer_batch_jobs",
                        "details": {
                            "current_intensity": current_intensity,
                            "best_time_intensity": best_intensity,
                            "best_time_start": best_time_start
                        },
                        "reason": (
                            f"Current carbon intensity is high ({current_intensity:.1f} gCO2eq/kWh). "
                            f"Consider deferring non-urgent batch jobs until {best_time_start} "
                            f"when intensity drops to {best_intensity:.1f} gCO2eq/kWh."
                        ),
                        "priority": "medium"
                    })
        
        # Workload prediction-based recommendations
        if workload_prediction and workload_prediction.get("status") == "success":
            analysis = workload_prediction.get("analysis", {})
            spike_detected = analysis.get("spike_detected", False)
            trend = analysis.get("trend", "stable")
            
            if spike_detected:
                spike_pct = analysis.get("spike_percentage", 0)
                recommendations.append({
                    "type": "workload_prediction",
                    "action": "prepare_for_spike",
                    "details": {
                        "spike_percentage": spike_pct,
                        "current_usage": workload_prediction.get("current_value", 0)
                    },
                    "reason": (
                        f"Predicted workload spike of {spike_pct:.1f}% in the near future. "
                        f"Consider proactive scaling to handle the increased load."
                    ),
                    "priority": "high" if spike_pct > 50 else "medium"
                })
            elif trend == "decreasing" and cpu_utilization < 0.5:
                recommendations.append({
                    "type": "workload_prediction",
                    "action": "prepare_for_decrease",
                    "details": {
                        "trend": trend,
                        "current_usage": workload_prediction.get("current_value", 0)
                    },
                    "reason": (
                        "Workload is predicted to decrease. "
                        "Consider scaling down resources or scheduling maintenance during this period."
                    ),
                    "priority": "low"
                })
        
        return recommendations
    
    def generate_recommendations(
        self,
        metrics: Dict[str, Any],
        carbon_data: Optional[Dict[str, Any]] = None,
        workload_prediction: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive recommendations based on all available data.
        
        Args:
            metrics: Dictionary of collected metrics
            carbon_data: Carbon intensity data if available
            workload_prediction: Workload prediction data if available
            
        Returns:
            Dictionary with all recommendations
        """
        if not metrics:
            return {"error": "No metrics data available"}
        
        try:
            # Extract required data from metrics
            current_cpu = metrics.get("current", {}).get("cpu_usage", {})
            current_memory = metrics.get("current", {}).get("memory_usage", {})
            requests = metrics.get("requests", {})
            limits = metrics.get("limits", {})
            node_capacity = metrics.get("node_capacity", {})
            
            # Recommendations for individual pods
            pod_recommendations = []
            
            for pod_name in set(current_cpu.keys()) | set(current_memory.keys()):
                # Skip if pod doesn't have both CPU and memory usage
                if pod_name not in current_cpu or pod_name not in current_memory:
                    continue
                
                # Get current usage
                pod_cpu = current_cpu.get(pod_name, 0)
                pod_memory = current_memory.get(pod_name, 0)
                
                # Get requests
                pod_cpu_request = requests.get(pod_name, {}).get("cpu", 0)
                pod_memory_request = requests.get(pod_name, {}).get("memory", 0)
                
                # Get limits
                pod_cpu_limit = limits.get(pod_name, {}).get("cpu")
                pod_memory_limit = limits.get(pod_name, {}).get("memory")
                
                # Generate recommendations for this pod
                pod_recs = self.generate_pod_recommendations(
                    pod_name=pod_name,
                    current_cpu=pod_cpu,
                    current_memory=pod_memory,
                    requested_cpu=pod_cpu_request,
                    requested_memory=pod_memory_request,
                    limit_cpu=pod_cpu_limit,
                    limit_memory=pod_memory_limit
                )
                
                # Only add if there's an actual recommendation
                if pod_recs["priority"] != "none":
                    pod_recommendations.append(pod_recs)
            
            # Generate cluster-level recommendations
            cluster_recommendations = self.generate_cluster_recommendations(
                node_capacity=node_capacity,
                current_usage={"overall": {
                    "cpu": sum(current_cpu.values()),
                    "memory": sum(current_memory.values())
                }},
                carbon_data=carbon_data,
                workload_prediction=workload_prediction
            )
            
            # Create energy savings summary
            potential_cpu_savings = sum(
                rec["cpu"].get("current_request", 0) - rec["cpu"].get("recommended_request", 0)
                for rec in pod_recommendations
                if "cpu" in rec and rec["cpu"].get("action") == "decrease"
            )
            
            potential_memory_savings = sum(
                rec["memory"].get("current_request", 0) - rec["memory"].get("recommended_request", 0)
                for rec in pod_recommendations
                if "memory" in rec and rec["memory"].get("action") == "decrease"
            )
            
            # Sort recommendations by priority
            priority_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
            pod_recommendations.sort(key=lambda x: priority_order.get(x["priority"], 4))
            
            # Format memory values for better readability in the summary
            memory_mb = potential_memory_savings / (1024 * 1024)
            
            # Compute aggregate metrics for the summary
            over_provisioned_pods = len([
                rec for rec in pod_recommendations 
                if ("cpu" in rec and rec["cpu"].get("action") == "decrease") or 
                   ("memory" in rec and rec["memory"].get("action") == "decrease")
            ])
            under_provisioned_pods = len([
                rec for rec in pod_recommendations 
                if ("cpu" in rec and rec["cpu"].get("action") == "increase") or 
                   ("memory" in rec and rec["memory"].get("action") == "increase")
            ])
            
            # Generate carbon impact info if data available
            carbon_impact = None
            if carbon_data and carbon_data.get("current_intensity") and potential_cpu_savings > 0:
                # Rough estimate: 1 CPU core ≈ 10 watts, run for 1 hour = 0.01 kWh
                # So savings in kWh ≈ potential_cpu_savings * 0.01 per hour
                kwh_savings_per_hour = potential_cpu_savings * 0.01
                carbon_saved_per_hour = kwh_savings_per_hour * carbon_data.get("current_intensity")
                
                carbon_impact = {
                    "estimated_energy_savings_kwh_per_hour": kwh_savings_per_hour,
                    "estimated_carbon_reduction_g_per_hour": carbon_saved_per_hour,
                    "carbon_intensity_g_per_kwh": carbon_data.get("current_intensity")
                }
            
            return {
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_pods_analyzed": len(set(current_cpu.keys()) | set(current_memory.keys())),
                    "over_provisioned_pods": over_provisioned_pods,
                    "under_provisioned_pods": under_provisioned_pods,
                    "potential_cpu_savings_cores": potential_cpu_savings,
                    "potential_memory_savings_mb": memory_mb,
                    "carbon_impact": carbon_impact
                },
                "pod_recommendations": pod_recommendations,
                "cluster_recommendations": cluster_recommendations
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {"error": f"Failed to generate recommendations: {str(e)}"}
    
    def format_recommendations_text(self, recommendations: Dict[str, Any]) -> str:
        """
        Format recommendations as human-readable text.
        
        Args:
            recommendations: Recommendation dictionary from generate_recommendations
            
        Returns:
            Formatted text output
        """
        if "error" in recommendations:
            return f"Error: {recommendations['error']}"
        
        summary = recommendations.get("summary", {})
        pod_recs = recommendations.get("pod_recommendations", [])
        cluster_recs = recommendations.get("cluster_recommendations", [])
        
        lines = []
        
        # Title
        lines.append("=== GreenOps Agent Recommendations ===")
        lines.append(f"Generated at: {recommendations.get('timestamp', datetime.now().isoformat())}")
        lines.append("")
        
        # Summary
        lines.append("=== Summary ===")
        lines.append(f"Total pods analyzed: {summary.get('total_pods_analyzed', 0)}")
        lines.append(f"Over-provisioned pods: {summary.get('over_provisioned_pods', 0)}")
        lines.append(f"Under-provisioned pods: {summary.get('under_provisioned_pods', 0)}")
        
        cpu_savings = summary.get("potential_cpu_savings_cores", 0)
        memory_savings = summary.get("potential_memory_savings_mb", 0)
        
        if cpu_savings > 0 or memory_savings > 0:
            lines.append("\nPotential resource savings:")
            if cpu_savings > 0:
                lines.append(f"- CPU: {cpu_savings:.2f} cores")
            if memory_savings > 0:
                lines.append(f"- Memory: {memory_savings:.1f} MB")
                
            # Carbon impact
            carbon = summary.get("carbon_impact")
            if carbon:
                kwh = carbon.get("estimated_energy_savings_kwh_per_hour", 0)
                co2 = carbon.get("estimated_carbon_reduction_g_per_hour", 0)
                
                if kwh > 0:
                    lines.append(f"\nEstimated energy savings: {kwh:.3f} kWh per hour")
                    lines.append(f"Estimated carbon reduction: {co2:.1f} gCO2eq per hour")
        
        # Cluster recommendations
        if cluster_recs:
            lines.append("\n=== Cluster Recommendations ===")
            for i, rec in enumerate(cluster_recs, 1):
                lines.append(f"{i}. [{rec.get('priority', 'medium').upper()}] {rec.get('reason', 'No details')}")
        
        # Pod recommendations
        if pod_recs:
            lines.append("\n=== Pod Recommendations ===")
            
            # Sort by priority
            priority_order = {"high": 0, "medium": 1, "low": 2, "none": 3}
            sorted_pod_recs = sorted(
                pod_recs,
                key=lambda x: (priority_order.get(x.get("priority", "low"), 4), x.get("pod_name", ""))
            )
            
            for i, rec in enumerate(sorted_pod_recs, 1):
                pod_name = rec.get("pod_name", "unknown")
                reason = rec.get("reason", "No details")
                priority = rec.get("priority", "low").upper()
                
                lines.append(f"{i}. [{priority}] Pod: {pod_name}")
                lines.append(f"   {reason}")
                
                # Add specific CPU recommendation details
                if "cpu" in rec and "action" in rec["cpu"]:
                    cpu_action = rec["cpu"]["action"]
                    current = rec["cpu"].get("current_request", 0)
                    recommended = rec["cpu"].get("recommended_request", 0)
                    util = rec["cpu"].get("utilization", 0)
                    
                    lines.append(f"   CPU: {cpu_action} from {current:.2f} to {recommended:.2f} cores (utilization: {util:.1f}%)")
                
                # Add specific memory recommendation details
                if "memory" in rec and "action" in rec["memory"]:
                    mem_action = rec["memory"]["action"]
                    current = rec["memory"].get("current_request", 0) / (1024 * 1024)  # Convert to MB
                    recommended = rec["memory"].get("recommended_request", 0) / (1024 * 1024)
                    util = rec["memory"].get("utilization", 0)
                    
                    lines.append(f"   Memory: {mem_action} from {current:.1f} MB to {recommended:.1f} MB (utilization: {util:.1f}%)")
                
                lines.append("")  # Add space between pod recommendations
        
        # Final tips and advice
        lines.append("\n=== Additional Tips ===")
        lines.append("1. After applying recommended resource adjustments, monitor workloads to ensure performance.")
        lines.append("2. Consider using Kubernetes Vertical Pod Autoscaler in recommendation mode.")
        lines.append("3. For critical services, maintain some headroom above the recommended values.")
        
        if summary.get("carbon_impact"):
            lines.append("4. Check carbon intensity before scheduling batch jobs to reduce emissions.")
        
        return "\n".join(lines)