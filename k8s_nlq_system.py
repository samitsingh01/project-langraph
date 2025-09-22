# Kubernetes Natural Language Query System with AWS Bedrock - FIXED VERSION
# Requirements: pip install langgraph langchain kubernetes streamlit boto3 langchain-aws

import streamlit as st
import os
import boto3
from kubernetes import client, config
from kubernetes.client.rest import ApiException
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_aws import ChatBedrock
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
import json
import re
import warnings
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
class QueryType(Enum):
    LIST_PODS = "list_pods"
    LIST_DEPLOYMENTS = "list_deployments" 
    LIST_SERVICES = "list_services"
    LIST_CONFIGMAPS = "list_configmaps"
    LIST_NAMESPACES = "list_namespaces"
    LIST_NODES = "list_nodes"
    GET_POD_DETAILS = "get_pod_details"
    GET_DEPLOYMENT_STATUS = "get_deployment_status"
    LIST_EVENTS = "list_events"
    GET_RESOURCE_USAGE = "get_resource_usage"
    CHECK_HEALTH = "check_health"
    FIND_ISSUES = "find_issues"
    UNSUPPORTED = "unsupported"

@dataclass
class QueryResult:
    success: bool
    data: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

# State definition for LangGraph
class QueryState(TypedDict):
    user_input: str
    parsed_query: Dict[str, Any]
    k8s_command: str
    result: QueryResult
    warnings: List[str]

class KubernetesNLProcessor:
    def __init__(self):
        # Initialize AWS Bedrock client
        self.bedrock_client = None
        self.llm = None
        self.bedrock_connected = False
        self.bedrock_error = None
        
        # Initialize Kubernetes client
        self.v1 = None
        self.apps_v1 = None
        self.k8s_connected = False
        self.k8s_error = None
        
        # Initialize connections
        self._init_bedrock()
        self._init_kubernetes()
        
        # Sensitive data patterns
        self.sensitive_patterns = [
            r'password', r'secret', r'token', r'key', r'credential',
            r'cert', r'private', r'auth', r'api[_-]?key'
        ]
    
    def _init_bedrock(self):
        """Initialize AWS Bedrock connection with better error handling."""
        try:
            # Check for AWS credentials
            aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
            aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
            aws_region = os.getenv('AWS_REGION', 'us-east-1')
            
            if not aws_access_key or not aws_secret_key:
                self.bedrock_connected = False
                return
            
            # Initialize Bedrock client
            self.bedrock_client = boto3.client(
                'bedrock-runtime',
                region_name=aws_region,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
            
            # Initialize ChatBedrock with corrected import
            self.llm = ChatBedrock(
                client=self.bedrock_client,
                model_id=os.getenv('BEDROCK_MODEL_ID', 'anthropic.claude-3-haiku-20240307-v1:0'),
                model_kwargs={
                    "max_tokens": 1000,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            )
            
            # Test with a simple invoke to verify connection
            test_response = self.llm.invoke([HumanMessage(content="test")])
            
            self.bedrock_connected = True
            
        except Exception as e:
            self.bedrock_connected = False
            # Store error for debugging but don't show to user immediately
            self.bedrock_error = str(e)

    def _init_kubernetes(self):
        """Initialize Kubernetes connection with multiple fallback options."""
        try:
            # Method 1: Try in-cluster config (when running inside K8s)
            try:
                config.load_incluster_config()
            except config.ConfigException:
                # Method 2: Try loading from default kubeconfig location
                try:
                    config.load_kube_config()
                except config.ConfigException:
                    # Method 3: Try loading from custom path
                    kubeconfig_paths = [
                        os.path.expanduser("~/.kube/config"),
                        "/home/appuser/.kube/config",  # Docker container path
                        os.getenv('KUBECONFIG', '')
                    ]
                    
                    loaded = False
                    for path in kubeconfig_paths:
                        if path and os.path.exists(path):
                            try:
                                config.load_kube_config(config_file=path)
                                loaded = True
                                break
                            except Exception as e:
                                continue
                    
                    if not loaded:
                        raise config.ConfigException("No valid kubeconfig found")
            
            # Initialize API clients
            self.v1 = client.CoreV1Api()
            self.apps_v1 = client.AppsV1Api()
            
            # Test connection by listing namespaces
            try:
                namespaces = self.v1.list_namespace(limit=1)
                self.k8s_connected = True
            except ApiException as e:
                raise Exception(f"Kubernetes API test failed: {e.reason}")
                
        except Exception as e:
            self.k8s_connected = False
            self.k8s_error = str(e)
        
    def parse_with_bedrock(self, user_input: str) -> Dict[str, Any]:
        """Use Bedrock LLM to parse natural language query."""
        if not self.bedrock_connected or not self.llm:
            return self.parse_with_rules(user_input)
        
        try:
            prompt = f"""
            You are a Kubernetes query parser. Parse the following natural language query into a structured format.
            
            User Query: "{user_input}"
            
            Return ONLY a JSON object with these fields:
            - "type": one of ["list_pods", "list_deployments", "list_services", "list_configmaps", "list_namespaces", "list_nodes", "get_pod_details", "get_deployment_status", "list_events", "check_health", "find_issues", "unsupported"]
            - "namespace": the namespace name or "default" or "all"
            - "resource_name": specific resource name if mentioned (optional)
            
            Examples:
            - "list all pods" -> {{"type": "list_pods", "namespace": "all"}}
            - "show deployments in kube-system" -> {{"type": "list_deployments", "namespace": "kube-system"}}
            - "what pods are running" -> {{"type": "list_pods", "namespace": "all"}}
            - "check health of my cluster" -> {{"type": "check_health", "namespace": "all"}}
            - "find problems in default namespace" -> {{"type": "find_issues", "namespace": "default"}}
            - "show me recent events" -> {{"type": "list_events", "namespace": "all"}}
            - "what's wrong with my cluster" -> {{"type": "find_issues", "namespace": "all"}}
            - "are my services working" -> {{"type": "list_services", "namespace": "all"}}
            
            JSON Response:
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            # Clean up the response and parse JSON
            response_text = response.content.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            parsed = json.loads(response_text)
            
            # Validate the response
            valid_types = ["list_pods", "list_deployments", "list_services", 
                          "list_configmaps", "list_namespaces", "list_nodes", 
                          "get_pod_details", "get_deployment_status", "list_events", 
                          "check_health", "find_issues", "unsupported"]
            
            if parsed.get("type") not in valid_types:
                parsed["type"] = "unsupported"
            
            # Convert type string to enum
            type_mapping = {
                "list_pods": QueryType.LIST_PODS,
                "list_deployments": QueryType.LIST_DEPLOYMENTS,
                "list_services": QueryType.LIST_SERVICES,
                "list_configmaps": QueryType.LIST_CONFIGMAPS,
                "list_namespaces": QueryType.LIST_NAMESPACES,
                "list_nodes": QueryType.LIST_NODES,
                "get_pod_details": QueryType.GET_POD_DETAILS,
                "get_deployment_status": QueryType.GET_DEPLOYMENT_STATUS,
                "list_events": QueryType.LIST_EVENTS,
                "check_health": QueryType.CHECK_HEALTH,
                "find_issues": QueryType.FIND_ISSUES,
                "unsupported": QueryType.UNSUPPORTED
            }
            
            parsed["type"] = type_mapping.get(parsed["type"], QueryType.UNSUPPORTED)
            
            return parsed
            
        except Exception as e:
            return self.parse_with_rules(user_input)
    
    def parse_with_rules(self, user_input: str) -> Dict[str, Any]:
        """Enhanced rule-based parsing with more natural language support."""
        user_input = user_input.lower().strip()
        parsed_query = {"type": QueryType.UNSUPPORTED, "namespace": "default", "resource_name": None}
        
        # More flexible keyword matching
        action_keywords = ["list", "show", "get", "find", "display", "what", "which", "how many", "tell me"]
        resource_keywords = {
            "pods": QueryType.LIST_PODS,
            "pod": QueryType.LIST_PODS,
            "deployments": QueryType.LIST_DEPLOYMENTS,
            "deployment": QueryType.LIST_DEPLOYMENTS,
            "deploy": QueryType.LIST_DEPLOYMENTS,
            "services": QueryType.LIST_SERVICES,
            "service": QueryType.LIST_SERVICES,
            "svc": QueryType.LIST_SERVICES,
            "configmaps": QueryType.LIST_CONFIGMAPS,
            "configmap": QueryType.LIST_CONFIGMAPS,
            "config": QueryType.LIST_CONFIGMAPS,
            "namespaces": QueryType.LIST_NAMESPACES,
            "namespace": QueryType.LIST_NAMESPACES,
            "ns": QueryType.LIST_NAMESPACES,
            "nodes": QueryType.LIST_NODES,
            "node": QueryType.LIST_NODES
        }
        
        # Check for action + resource combinations
        has_action = any(keyword in user_input for keyword in action_keywords)
        if has_action or any(phrase in user_input for phrase in ["running", "available", "status", "health"]):
            for resource, query_type in resource_keywords.items():
                if resource in user_input:
                    parsed_query["type"] = query_type
                    break
        
        # Handle health/status queries
        if any(word in user_input for word in ["health", "healthy", "status", "working", "running"]):
            if any(word in user_input for word in ["cluster", "all", "everything"]):
                parsed_query["type"] = QueryType.CHECK_HEALTH
            elif "pod" in user_input:
                parsed_query["type"] = QueryType.LIST_PODS
            elif "service" in user_input:
                parsed_query["type"] = QueryType.LIST_SERVICES
            elif "deployment" in user_input:
                parsed_query["type"] = QueryType.LIST_DEPLOYMENTS
        
        # Handle problem/issue queries
        if any(word in user_input for word in ["problem", "issue", "error", "fail", "broken", "wrong", "trouble"]):
            parsed_query["type"] = QueryType.FIND_ISSUES
        
        # Handle event queries
        if any(word in user_input for word in ["event", "events", "recent", "happened", "activity"]):
            parsed_query["type"] = QueryType.LIST_EVENTS
        
        # Extract namespace if specified
        namespace_patterns = [
            r'namespace\s+(\w+)',
            r'ns\s+(\w+)',
            r'in\s+(\w+)\s+namespace',
            r'from\s+(\w+)\s+namespace',
            r'in\s+(\w+)'
        ]
        
        for pattern in namespace_patterns:
            namespace_match = re.search(pattern, user_input)
            if namespace_match:
                parsed_query["namespace"] = namespace_match.group(1)
                break
        
        # Check for "all namespaces" variations
        if any(phrase in user_input for phrase in ["all namespaces", "all-namespaces", "every namespace", "across namespaces", "everywhere"]):
            parsed_query["namespace"] = "all"
            
        return parsed_query
        
    def parse_natural_language(self, state: QueryState) -> QueryState:
        """Parse natural language input to determine query type and parameters."""
        user_input = state["user_input"]
        
        # Use Bedrock if available, otherwise fall back to rules
        parsed_query = self.parse_with_bedrock(user_input)
        
        state["parsed_query"] = parsed_query
        return state
    
    def validate_query(self, state: QueryState) -> QueryState:
        """Validate the query and check for potential security issues."""
        warnings = []
        parsed_query = state["parsed_query"]
        
        # Check if query type is supported
        if parsed_query["type"] == QueryType.UNSUPPORTED:
            warnings.append("⚠️ I couldn't understand that query. Try asking about pods, deployments, services, health checks, or finding issues.")
            # Suggest similar queries based on keywords
            user_input_lower = state["user_input"].lower()
            suggestions = []
            if any(word in user_input_lower for word in ["health", "status", "working"]):
                suggestions.append("Try: 'Check health of my cluster'")
            if any(word in user_input_lower for word in ["problem", "issue", "error", "wrong"]):
                suggestions.append("Try: 'Find problems in my cluster'")
            if any(word in user_input_lower for word in ["event", "recent", "happened"]):
                suggestions.append("Try: 'Show me recent events'")
            if suggestions:
                warnings.append(f"💡 Suggestions: {', '.join(suggestions)}")
        
        # Check for sensitive data requests
        user_input = state["user_input"].lower()
        for pattern in self.sensitive_patterns:
            if re.search(pattern, user_input, re.IGNORECASE):
                warnings.append(f"🔒 WARNING: Query may involve sensitive data ({pattern}). Proceed with caution.")
        
        # Special warnings for certain resource types
        if parsed_query["type"] == QueryType.LIST_CONFIGMAPS:
            warnings.append("🔐 ConfigMaps may contain sensitive configuration data.")
        
        state["warnings"] = warnings
        return state
    
    def execute_kubernetes_query(self, state: QueryState) -> QueryState:
        """Execute the Kubernetes query with better error handling."""
        if not self.k8s_connected:
            state["result"] = QueryResult(
                success=False,
                error=f"Not connected to Kubernetes cluster: {self.k8s_error}"
            )
            return state
        
        parsed_query = state["parsed_query"]
        query_type = parsed_query["type"]
        namespace = parsed_query.get("namespace", "default")
        
        try:
            result_data = []
            
            if query_type == QueryType.LIST_PODS:
                if namespace == "all":
                    pods = self.v1.list_pod_for_all_namespaces()
                else:
                    pods = self.v1.list_namespaced_pod(namespace=namespace)
                
                result_data = []
                for pod in pods.items:
                    # Calculate ready containers
                    ready_containers = 0
                    total_containers = len(pod.spec.containers) if pod.spec.containers else 0
                    restart_count = 0
                    
                    if pod.status.container_statuses:
                        ready_containers = sum(1 for c in pod.status.container_statuses if c.ready)
                        restart_count = sum(c.restart_count for c in pod.status.container_statuses)
                    
                    result_data.append({
                        "name": pod.metadata.name,
                        "namespace": pod.metadata.namespace,
                        "status": pod.status.phase or "Unknown",
                        "node": pod.spec.node_name or "Unknown",
                        "ready": f"{ready_containers}/{total_containers}",
                        "restarts": restart_count,
                        "age": self._calculate_age(pod.metadata.creation_timestamp)
                    })
                
            elif query_type == QueryType.LIST_DEPLOYMENTS:
                if namespace == "all":
                    deployments = self.apps_v1.list_deployment_for_all_namespaces()
                else:
                    deployments = self.apps_v1.list_namespaced_deployment(namespace=namespace)
                
                result_data = [{
                    "name": dep.metadata.name,
                    "namespace": dep.metadata.namespace,
                    "ready": f"{dep.status.ready_replicas or 0}/{dep.spec.replicas or 0}",
                    "up_to_date": dep.status.updated_replicas or 0,
                    "available": dep.status.available_replicas or 0,
                    "age": self._calculate_age(dep.metadata.creation_timestamp)
                } for dep in deployments.items]
                
            elif query_type == QueryType.LIST_SERVICES:
                if namespace == "all":
                    services = self.v1.list_service_for_all_namespaces()
                else:
                    services = self.v1.list_namespaced_service(namespace=namespace)
                
                result_data = []
                for svc in services.items:
                    # Handle external IP
                    external_ip = "<none>"
                    if svc.status.load_balancer and svc.status.load_balancer.ingress:
                        external_ips = []
                        for ingress in svc.status.load_balancer.ingress:
                            if ingress.ip:
                                external_ips.append(ingress.ip)
                            elif ingress.hostname:
                                external_ips.append(ingress.hostname)
                        external_ip = ', '.join(external_ips) if external_ips else "<pending>"
                    
                    # Handle ports
                    ports = []
                    if svc.spec.ports:
                        for port in svc.spec.ports:
                            port_str = f"{port.port}"
                            if port.target_port:
                                port_str += f":{port.target_port}"
                            if port.protocol:
                                port_str += f"/{port.protocol}"
                            ports.append(port_str)
                    
                    result_data.append({
                        "name": svc.metadata.name,
                        "namespace": svc.metadata.namespace,
                        "type": svc.spec.type or "ClusterIP",
                        "cluster_ip": svc.spec.cluster_ip or "<none>",
                        "external_ip": external_ip,
                        "ports": ', '.join(ports) if ports else "<none>",
                        "age": self._calculate_age(svc.metadata.creation_timestamp)
                    })
                
            elif query_type == QueryType.LIST_CONFIGMAPS:
                if namespace == "all":
                    configmaps = self.v1.list_config_map_for_all_namespaces()
                else:
                    configmaps = self.v1.list_namespaced_config_map(namespace=namespace)
                
                result_data = [{
                    "name": cm.metadata.name,
                    "namespace": cm.metadata.namespace,
                    "data_keys": ', '.join(list(cm.data.keys())) if cm.data else "None",
                    "age": self._calculate_age(cm.metadata.creation_timestamp)
                } for cm in configmaps.items]
                
            elif query_type == QueryType.LIST_NAMESPACES:
                namespaces = self.v1.list_namespace()
                result_data = [{
                    "name": ns.metadata.name,
                    "status": ns.status.phase or "Unknown",
                    "age": self._calculate_age(ns.metadata.creation_timestamp)
                } for ns in namespaces.items]
                
            elif query_type == QueryType.LIST_NODES:
                nodes = self.v1.list_node()
                result_data = []
                
                for node in nodes.items:
                    # Determine node status
                    status = "NotReady"
                    if node.status.conditions:
                        for condition in node.status.conditions:
                            if condition.type == "Ready" and condition.status == "True":
                                status = "Ready"
                                break
                    
                    # Get node roles
                    roles = []
                    if node.metadata.labels:
                        for label_key in node.metadata.labels.keys():
                            if 'node-role.kubernetes.io/' in label_key:
                                role = label_key.split('/')[-1]
                                if role:
                                    roles.append(role)
                    
                    if not roles:
                        roles = ["worker"]
                    
                    # Get internal IP
                    internal_ip = "Unknown"
                    if node.status.addresses:
                        for addr in node.status.addresses:
                            if addr.type == "InternalIP":
                                internal_ip = addr.address
                                break
                    
                    result_data.append({
                        "name": node.metadata.name,
                        "status": status,
                        "roles": ', '.join(roles),
                        "version": node.status.node_info.kubelet_version if node.status.node_info else "Unknown",
                        "internal_ip": internal_ip,
                        "age": self._calculate_age(node.metadata.creation_timestamp)
                    })
                    
            elif query_type == QueryType.LIST_EVENTS:
                if namespace == "all":
                    events = self.v1.list_event_for_all_namespaces()
                else:
                    events = self.v1.list_namespaced_event(namespace=namespace)
                
                # Get recent events (last 50)
                recent_events = sorted(events.items, 
                                     key=lambda x: x.last_timestamp or x.first_timestamp or datetime.now(), 
                                     reverse=True)[:50]
                
                result_data = [{
                    "namespace": event.namespace,
                    "type": event.type,
                    "reason": event.reason,
                    "object": f"{event.involved_object.kind}/{event.involved_object.name}",
                    "message": event.message[:100] + "..." if len(event.message) > 100 else event.message,
                    "time": self._calculate_age(event.last_timestamp or event.first_timestamp)
                } for event in recent_events]
                
            elif query_type == QueryType.CHECK_HEALTH:
                # Health check across multiple resources
                health_data = []
                
                # Check nodes
                nodes = self.v1.list_node()
                ready_nodes = sum(1 for node in nodes.items 
                                if any(cond.type == "Ready" and cond.status == "True" 
                                      for cond in node.status.conditions))
                health_data.append({
                    "component": "Nodes",
                    "total": len(nodes.items),
                    "ready": ready_nodes,
                    "status": "Healthy" if ready_nodes == len(nodes.items) else "Issues",
                })
                
                # Check pods
                pods = self.v1.list_pod_for_all_namespaces()
                running_pods = sum(1 for pod in pods.items if pod.status.phase == "Running")
                health_data.append({
                    "component": "Pods",
                    "total": len(pods.items),
                    "ready": running_pods,
                    "status": "Healthy" if running_pods > len(pods.items) * 0.8 else "Issues",
                })
                
                result_data = health_data
                
            elif query_type == QueryType.FIND_ISSUES:
                # Find problematic resources
                issues = []
                
                # Check for failed pods
                if namespace == "all":
                    pods = self.v1.list_pod_for_all_namespaces()
                else:
                    pods = self.v1.list_namespaced_pod(namespace=namespace)
                
                for pod in pods.items:
                    if pod.status.phase in ["Failed", "Pending", "Unknown"]:
                        issues.append({
                            "type": "Pod Issue",
                            "resource": f"{pod.metadata.namespace}/{pod.metadata.name}",
                            "status": pod.status.phase,
                            "message": pod.status.message or "No details available",
                            "age": self._calculate_age(pod.metadata.creation_timestamp)
                        })
                
                # Check for recent error events
                try:
                    if namespace == "all":
                        events = self.v1.list_event_for_all_namespaces()
                    else:
                        events = self.v1.list_namespaced_event(namespace=namespace)
                    
                    now = datetime.now()
                    error_events = []
                    for event in events.items:
                        if event.type == "Warning" and event.last_timestamp:
                            try:
                                # Handle timezone-aware datetime
                                if event.last_timestamp.tzinfo is not None:
                                    event_time = event.last_timestamp.replace(tzinfo=None)
                                else:
                                    event_time = event.last_timestamp
                                
                                time_diff = (now - event_time).total_seconds()
                                if time_diff < 3600:  # Last hour
                                    error_events.append(event)
                            except:
                                continue
                    
                    for event in error_events[:10]:  # Last 10 warnings
                        issues.append({
                            "type": "Warning Event",
                            "resource": f"{event.involved_object.kind}/{event.involved_object.name}",
                            "status": event.reason,
                            "message": event.message[:100] + "..." if len(event.message) > 100 else event.message,
                            "age": self._calculate_age(event.last_timestamp)
                        })
                except:
                    pass  # Skip events if there's an error
                
                result_data = issues
            
            state["result"] = QueryResult(success=True, data=result_data)
            
        except ApiException as e:
            error_msg = f"Kubernetes API error: {e.reason}"
            if e.status == 403:
                error_msg += " (Permission denied - check RBAC settings)"
            elif e.status == 404:
                error_msg += f" (Namespace '{namespace}' not found)"
            
            state["result"] = QueryResult(success=False, error=error_msg)
            
        except Exception as e:
            state["result"] = QueryResult(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
            
        return state
    
    def _calculate_age(self, creation_timestamp):
        """Calculate human-readable age from creation timestamp."""
        if not creation_timestamp:
            return "Unknown"
        
        try:
            # Handle timezone-aware datetime
            if creation_timestamp.tzinfo is not None:
                now = datetime.now(creation_timestamp.tzinfo)
            else:
                now = datetime.now()
            
            age = now - creation_timestamp
            
            if age.days > 0:
                return f"{age.days}d"
            elif age.seconds > 3600:
                return f"{age.seconds // 3600}h"
            elif age.seconds > 60:
                return f"{age.seconds // 60}m"
            else:
                return f"{age.seconds}s"
        except:
            return creation_timestamp.strftime('%Y-%m-%d %H:%M:%S') if creation_timestamp else "Unknown"
    
    def create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow."""
        workflow = StateGraph(QueryState)
        
        # Add nodes
        workflow.add_node("parse", self.parse_natural_language)
        workflow.add_node("validate", self.validate_query)
        workflow.add_node("execute", self.execute_kubernetes_query)
        
        # Add edges
        workflow.add_edge("parse", "validate")
        workflow.add_edge("validate", "execute")
        workflow.add_edge("execute", END)
        
        # Set entry point
        workflow.set_entry_point("parse")
        
        return workflow.compile()

def main():
    st.set_page_config(
        page_title="K8s NL Query System (AWS Bedrock)",
        page_icon="☸️",
        layout="wide"
    )
    
    st.title("☸️ Kubernetes Natural Language Query System")
    st.markdown("*Powered by AWS Bedrock and LangGraph*")
    
    # Initialize the processor
    if 'processor' not in st.session_state:
        with st.spinner("Initializing connections..."):
            st.session_state.processor = KubernetesNLProcessor()
            st.session_state.workflow = st.session_state.processor.create_workflow()
    
    # Status indicators with clean display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.processor.k8s_connected:
            st.success("✅ Kubernetes Connected")
        else:
            st.error("❌ Kubernetes Disconnected")
            if hasattr(st.session_state.processor, 'k8s_error'):
                with st.expander("🔍 Connection Details"):
                    st.error(f"Error: {st.session_state.processor.k8s_error}")
                    st.info("""
                    **Quick Fixes:**
                    - Ensure `kubectl get namespaces` works
                    - Check if ~/.kube/config exists
                    - Restart: `docker-compose restart`
                    """)
    
    with col2:
        if st.session_state.processor.bedrock_connected:
            st.success("✅ AWS Bedrock Connected")
        else:
            st.warning("⚠️ Rule-based Parsing")
            if hasattr(st.session_state.processor, 'bedrock_error') and st.session_state.processor.bedrock_error:
                with st.expander("🔍 Bedrock Details"):
                    st.warning(f"Bedrock Error: {st.session_state.processor.bedrock_error}")
                    st.info("""
                    **Bedrock Troubleshooting:**
                    - Check AWS credentials in .env file
                    - Verify Bedrock is available in your region
                    - Ensure proper IAM permissions
                    """)
    
    with col3:
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        st.info(f"🌍 Region: {aws_region}")
    
    # Only show the interface if we have Kubernetes connection
    if not st.session_state.processor.k8s_connected:
        st.error("⚠️ Cannot proceed without Kubernetes connection. Please fix the connection issues above.")
        
        # Refresh button
        if st.button("🔄 Retry Connection"):
            st.session_state.clear()
            st.experimental_rerun()
        
        return
    
    # Input section
    st.subheader("Ask a Question")
    
    # Example queries
    with st.expander("📝 Example Queries - Try These!"):
        st.write("""
        **Basic Queries:**
        - "List all pods"
        - "Show me deployments in kube-system"
        - "What services are running?"
        - "Show all namespaces"
        
        **Health & Status:**
        - "Check health of my cluster"
        - "Are my pods healthy?"
        - "What's the status of my deployments?"
        - "How many nodes are ready?"
        
        **Problem Finding:**
        - "Find problems in my cluster"
        - "What's wrong with my pods?"
        - "Show me any issues"
        - "Are there any failed pods?"
        
        **Recent Activity:**
        - "Show me recent events"
        - "What happened recently?"
        - "Any warnings in the last hour?"
        
        **Flexible Questions:**
        - "What pods are running in default namespace?"
        - "How many services do I have?"
        - "Which deployments are not ready?"
        - "Show me everything in kube-system"
        """)
    
    # Text input
    user_query = st.text_input(
        "Enter your query:",
        placeholder="e.g., 'What's wrong with my cluster?' or 'Check health of my pods'",
        help="Type your Kubernetes query in natural language"
    )
    
    if st.button("🔍 Execute Query", type="primary"):
        if not user_query.strip():
            st.warning("Please enter a query!")
            return
            
        # Execute workflow
        processing_method = "AWS Bedrock LLM" if st.session_state.processor.bedrock_connected else "Rule-based"
        with st.spinner(f"Processing query with {processing_method}..."):
            initial_state = QueryState(
                user_input=user_query,
                parsed_query={},
                k8s_command="",
                result=QueryResult(success=False),
                warnings=[]
            )
            
            final_state = st.session_state.workflow.invoke(initial_state)
        
        # Display parsed query info
        with st.expander("🧠 Query Analysis"):
            parsed = final_state["parsed_query"]
            st.write(f"**Query Type:** {parsed.get('type', 'Unknown')}")
            st.write(f"**Namespace:** {parsed.get('namespace', 'default')}")
            st.write(f"**Processing Method:** {processing_method}")
        
        # Display warnings
        if final_state["warnings"]:
            for warning in final_state["warnings"]:
                st.warning(warning)
        
        # Display results
        result = final_state["result"]
        
        if result.success:
            st.success(f"✅ Query executed successfully!")
            
            if result.data:
                st.subheader("Results")
                
                # Display as dataframe for better visualization
                df = pd.DataFrame(result.data)
                st.dataframe(df, use_container_width=True)
                
                # Show count
                st.info(f"Found {len(result.data)} items")
                
                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"k8s_query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # JSON view option
                with st.expander("🔍 View Raw JSON"):
                    st.json(result.data)
            else:
                st.info("No data returned from query")
        else:
            st.error(f"❌ Query failed: {result.error}")
    
    # Refresh connection button
    if st.button("🔄 Refresh Connections"):
        st.session_state.clear()
        st.experimental_rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **Security Notes:**
    - Only listing operations are supported for safety
    - Queries involving sensitive data will show warnings  
    - ConfigMaps and Secrets require extra caution
    - All queries are processed through AWS Bedrock for enhanced understanding
    """)

if __name__ == "__main__":
    main()
