"""
WAN Video Generation Debugging and Introspection Utilities

This module provides comprehensive debugging tools for WAN video generation,
including runtime introspection, performance monitoring, and troubleshooting helpers.
"""

import time
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from contextlib import contextmanager
from dataclasses import dataclass
import json
from datetime import datetime
import threading
import functools


@dataclass
class NodeCallInfo:
    """Information about a node call for debugging."""
    node_name: str
    method_name: str
    inputs: Dict[str, Any]
    outputs: Optional[Any] = None
    duration_ms: float = 0.0
    memory_before_mb: float = 0.0
    memory_after_mb: float = 0.0
    memory_peak_mb: float = 0.0
    error: Optional[str] = None
    timestamp: str = ""


class DebugTracker:
    """
    Comprehensive debugging tracker for WAN video generation.
    
    Tracks node calls, memory usage, timing, and provides introspection utilities.
    """
    
    def __init__(self, enabled: bool = True):
        """
        Initialize the debug tracker.
        
        Args:
            enabled: Whether debugging is enabled
        """
        self.enabled = enabled
        self.call_history: List[NodeCallInfo] = []
        self.performance_stats: Dict[str, List[float]] = {}
        self.memory_timeline: List[Dict[str, Any]] = []
        self.current_generation_id: Optional[str] = None
        self._lock = threading.Lock()
        
    def set_generation_id(self, generation_id: str) -> None:
        """Set current generation ID for tracking."""
        with self._lock:
            self.current_generation_id = generation_id
            self.memory_timeline.append({
                "timestamp": datetime.now().isoformat(),
                "event": f"generation_start",
                "generation_id": generation_id,
                "memory_mb": self._get_memory_usage()
            })
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = {}
        
        # System memory
        sys_mem = psutil.virtual_memory()
        memory_info["system_used_mb"] = sys_mem.used / 1024 / 1024
        memory_info["system_available_mb"] = sys_mem.available / 1024 / 1024
        memory_info["system_percent"] = sys_mem.percent
        
        # GPU memory
        if torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info["gpu_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024
        else:
            memory_info["gpu_allocated_mb"] = 0.0
            memory_info["gpu_reserved_mb"] = 0.0
            memory_info["gpu_max_allocated_mb"] = 0.0
            
        return memory_info
    
    def _analyze_inputs(self, inputs: Dict[str, Any]) -> Dict[str, str]:
        """Analyze inputs for debugging information."""
        analysis = {}
        
        for key, value in inputs.items():
            if hasattr(value, 'shape'):
                # Tensor or numpy array
                analysis[key] = f"shape={value.shape}, dtype={value.dtype}"
                if hasattr(value, 'device'):
                    analysis[key] += f", device={value.device}"
            elif isinstance(value, (list, tuple)):
                analysis[key] = f"length={len(value)}, type={type(value).__name__}"
                if value and hasattr(value[0], 'shape'):
                    analysis[key] += f", item_shape={value[0].shape}"
            elif isinstance(value, dict):
                analysis[key] = f"dict with keys: {list(value.keys())}"
            elif isinstance(value, (int, float)):
                analysis[key] = f"{type(value).__name__}={value}"
            elif isinstance(value, str):
                analysis[key] = f"str(len={len(value)}): '{value[:50]}...'" if len(value) > 50 else f"str: '{value}'"
            else:
                analysis[key] = f"{type(value).__name__}"
                
        return analysis
    
    @contextmanager
    def track_node_call(self, node_name: str, method_name: str, inputs: Dict[str, Any]):
        """
        Context manager to track a node call with comprehensive debugging.
        
        Args:
            node_name: Name of the node class
            method_name: Name of the method being called
            inputs: Input parameters to the method
        """
        if not self.enabled:
            yield
            return
            
        call_info = NodeCallInfo(
            node_name=node_name,
            method_name=method_name,
            inputs=self._analyze_inputs(inputs),
            timestamp=datetime.now().isoformat()
        )
        
        # Record memory before
        memory_before = self._get_memory_usage()
        call_info.memory_before_mb = memory_before.get("gpu_allocated_mb", 0.0)
        
        # Start timing
        start_time = time.perf_counter()
        
        try:
            print(f"🔍 [DEBUG] {node_name}.{method_name}:")
            for key, analysis in call_info.inputs.items():
                print(f"    {key}: {analysis}")
            
            yield call_info
            
            # Record successful completion
            end_time = time.perf_counter()
            call_info.duration_ms = (end_time - start_time) * 1000
            
            # Record memory after
            memory_after = self._get_memory_usage()
            call_info.memory_after_mb = memory_after.get("gpu_allocated_mb", 0.0)
            call_info.memory_peak_mb = memory_after.get("gpu_max_allocated_mb", 0.0)
            
            print(f"    ✅ Completed in {call_info.duration_ms:.1f}ms")
            print(f"    📊 Memory: {call_info.memory_before_mb:.1f}MB → {call_info.memory_after_mb:.1f}MB")
            
        except Exception as e:
            # Record error
            end_time = time.perf_counter()
            call_info.duration_ms = (end_time - start_time) * 1000
            call_info.error = str(e)
            
            print(f"    ❌ Failed in {call_info.duration_ms:.1f}ms: {e}")
            raise
            
        finally:
            # Store call info
            with self._lock:
                self.call_history.append(call_info)
                
                # Update performance stats
                key = f"{node_name}.{method_name}"
                if key not in self.performance_stats:
                    self.performance_stats[key] = []
                self.performance_stats[key].append(call_info.duration_ms)
                
                # Update memory timeline
                self.memory_timeline.append({
                    "timestamp": call_info.timestamp,
                    "event": f"{node_name}.{method_name}",
                    "duration_ms": call_info.duration_ms,
                    "memory_before_mb": call_info.memory_before_mb,
                    "memory_after_mb": call_info.memory_after_mb,
                    "generation_id": self.current_generation_id
                })
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {}
        
        for operation, times in self.performance_stats.items():
            summary[operation] = {
                "call_count": len(times),
                "total_time_ms": sum(times),
                "avg_time_ms": np.mean(times),
                "min_time_ms": np.min(times),
                "max_time_ms": np.max(times),
                "std_time_ms": np.std(times)
            }
            
        return summary
    
    def get_memory_analysis(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.memory_timeline:
            return {}
            
        gpu_usage = [entry["memory_after_mb"] for entry in self.memory_timeline 
                    if "memory_after_mb" in entry]
        
        if not gpu_usage:
            return {}
            
        return {
            "peak_memory_mb": max(gpu_usage),
            "avg_memory_mb": np.mean(gpu_usage),
            "memory_growth_mb": gpu_usage[-1] - gpu_usage[0] if len(gpu_usage) > 1 else 0,
            "timeline_entries": len(self.memory_timeline)
        }
    
    def find_bottlenecks(self, threshold_ms: float = 1000.0) -> List[Dict[str, Any]]:
        """Find performance bottlenecks above threshold."""
        bottlenecks = []
        
        for operation, times in self.performance_stats.items():
            avg_time = np.mean(times)
            if avg_time > threshold_ms:
                bottlenecks.append({
                    "operation": operation,
                    "avg_time_ms": avg_time,
                    "call_count": len(times),
                    "total_time_ms": sum(times)
                })
                
        return sorted(bottlenecks, key=lambda x: x["avg_time_ms"], reverse=True)
    
    def get_error_summary(self) -> List[Dict[str, Any]]:
        """Get summary of errors that occurred."""
        errors = []
        for call in self.call_history:
            if call.error:
                errors.append({
                    "node": call.node_name,
                    "method": call.method_name,
                    "error": call.error,
                    "timestamp": call.timestamp,
                    "duration_ms": call.duration_ms
                })
        return errors
    
    def save_debug_report(self, output_path: Path) -> None:
        """Save comprehensive debug report to file."""
        report = {
            "generation_id": self.current_generation_id,
            "timestamp": datetime.now().isoformat(),
            "call_history": [
                {
                    "node_name": call.node_name,
                    "method_name": call.method_name,
                    "duration_ms": call.duration_ms,
                    "memory_before_mb": call.memory_before_mb,
                    "memory_after_mb": call.memory_after_mb,
                    "error": call.error,
                    "timestamp": call.timestamp
                }
                for call in self.call_history
            ],
            "performance_summary": self.get_performance_summary(),
            "memory_analysis": self.get_memory_analysis(),
            "bottlenecks": self.find_bottlenecks(),
            "errors": self.get_error_summary(),
            "memory_timeline": self.memory_timeline
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Debug report saved to: {output_path}")
    
    def print_summary(self) -> None:
        """Print debugging summary to console."""
        print("\n" + "="*60)
        print("🔍 WAN VIDEO GENERATION DEBUG SUMMARY")
        print("="*60)
        
        # Performance summary
        print("\n📈 Performance Summary:")
        perf_summary = self.get_performance_summary()
        for operation, stats in perf_summary.items():
            print(f"  {operation}:")
            print(f"    Calls: {stats['call_count']}, Avg: {stats['avg_time_ms']:.1f}ms")
            print(f"    Total: {stats['total_time_ms']:.1f}ms, Range: {stats['min_time_ms']:.1f}-{stats['max_time_ms']:.1f}ms")
        
        # Memory analysis
        print("\n💾 Memory Analysis:")
        memory_analysis = self.get_memory_analysis()
        if memory_analysis:
            print(f"  Peak GPU Memory: {memory_analysis['peak_memory_mb']:.1f}MB")
            print(f"  Average GPU Memory: {memory_analysis['avg_memory_mb']:.1f}MB")
            print(f"  Memory Growth: {memory_analysis['memory_growth_mb']:.1f}MB")
        
        # Bottlenecks
        bottlenecks = self.find_bottlenecks()
        if bottlenecks:
            print("\n🐌 Performance Bottlenecks (>1000ms):")
            for bottleneck in bottlenecks[:5]:  # Top 5
                print(f"  {bottleneck['operation']}: {bottleneck['avg_time_ms']:.1f}ms avg")
        
        # Errors
        errors = self.get_error_summary()
        if errors:
            print("\n❌ Errors:")
            for error in errors:
                print(f"  {error['node']}.{error['method']}: {error['error']}")
        
        print("="*60)
    
    def clear(self) -> None:
        """Clear all debugging data."""
        with self._lock:
            self.call_history.clear()
            self.performance_stats.clear()
            self.memory_timeline.clear()
            self.current_generation_id = None


# Global debug tracker instance
debug_tracker = DebugTracker()


def debug_node_call(node_name: str, method_name: str):
    """
    Decorator to automatically track node calls.
    
    Args:
        node_name: Name of the node class
        method_name: Name of the method
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build inputs dict from args and kwargs
            inputs = kwargs.copy()
            if args:
                inputs["*args"] = f"{len(args)} positional args"
            
            with debug_tracker.track_node_call(node_name, method_name, inputs) as call_info:
                result = func(*args, **kwargs)
                call_info.outputs = f"{type(result).__name__}"
                return result
        return wrapper
    return decorator


class ModelStateInspector:
    """Utility for inspecting model states and configurations."""
    
    @staticmethod
    def inspect_model(model: Any, name: str = "model") -> Dict[str, Any]:
        """Inspect a model's properties and state."""
        info = {
            "name": name,
            "type": type(model).__name__,
            "device": None,
            "dtype": None,
            "parameters": 0,
            "trainable_parameters": 0,
            "memory_mb": 0.0
        }
        
        try:
            # Get device and dtype if available
            if hasattr(model, 'device'):
                info["device"] = str(model.device)
            if hasattr(model, 'dtype'):
                info["dtype"] = str(model.dtype)
            
            # Count parameters if it's a PyTorch model
            if hasattr(model, 'parameters'):
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                info["parameters"] = total_params
                info["trainable_parameters"] = trainable_params
                
                # Estimate memory usage (rough)
                param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
                info["memory_mb"] = param_memory / 1024 / 1024
            
        except Exception as e:
            info["error"] = str(e)
        
        return info
    
    @staticmethod
    def compare_models(model1: Any, model2: Any, name1: str = "model1", name2: str = "model2") -> Dict[str, Any]:
        """Compare two models."""
        info1 = ModelStateInspector.inspect_model(model1, name1)
        info2 = ModelStateInspector.inspect_model(model2, name2)
        
        comparison = {
            "model1": info1,
            "model2": info2,
            "differences": {}
        }
        
        # Compare key metrics
        for key in ["parameters", "trainable_parameters", "memory_mb"]:
            if key in info1 and key in info2:
                diff = info2[key] - info1[key]
                comparison["differences"][key] = {
                    "model1": info1[key],
                    "model2": info2[key],
                    "difference": diff,
                    "percent_change": (diff / info1[key] * 100) if info1[key] != 0 else 0
                }
        
        return comparison


class TensorAnalyzer:
    """Utility for analyzing tensor properties and distributions."""
    
    @staticmethod
    def analyze_tensor(tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """Analyze a tensor's properties and statistics."""
        if not isinstance(tensor, torch.Tensor):
            return {"error": f"Not a tensor: {type(tensor)}"}
        
        analysis = {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "requires_grad": tensor.requires_grad,
            "memory_mb": tensor.numel() * tensor.element_size() / 1024 / 1024
        }
        
        try:
            # Statistical analysis
            if tensor.numel() > 0:
                flat_tensor = tensor.flatten().float()
                analysis.update({
                    "min": float(torch.min(flat_tensor)),
                    "max": float(torch.max(flat_tensor)),
                    "mean": float(torch.mean(flat_tensor)),
                    "std": float(torch.std(flat_tensor)),
                    "zero_fraction": float(torch.sum(flat_tensor == 0) / flat_tensor.numel()),
                    "inf_count": int(torch.sum(torch.isinf(flat_tensor))),
                    "nan_count": int(torch.sum(torch.isnan(flat_tensor)))
                })
        except Exception as e:
            analysis["stats_error"] = str(e)
        
        return analysis
    
    @staticmethod
    def compare_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor, 
                       name1: str = "tensor1", name2: str = "tensor2", 
                       tolerance: float = 1e-6) -> Dict[str, Any]:
        """Compare two tensors."""
        analysis1 = TensorAnalyzer.analyze_tensor(tensor1, name1)
        analysis2 = TensorAnalyzer.analyze_tensor(tensor2, name2)
        
        comparison = {
            "tensor1": analysis1,
            "tensor2": analysis2,
            "compatible_shapes": tensor1.shape == tensor2.shape,
            "same_device": tensor1.device == tensor2.device,
            "same_dtype": tensor1.dtype == tensor2.dtype
        }
        
        # Detailed comparison if shapes match
        if tensor1.shape == tensor2.shape and tensor1.device == tensor2.device:
            try:
                diff = torch.abs(tensor1.float() - tensor2.float())
                comparison.update({
                    "max_difference": float(torch.max(diff)),
                    "mean_difference": float(torch.mean(diff)),
                    "close_elements_fraction": float(torch.sum(diff < tolerance) / diff.numel()),
                    "identical": torch.allclose(tensor1, tensor2, atol=tolerance)
                })
            except Exception as e:
                comparison["comparison_error"] = str(e)
        
        return comparison


def create_debug_session(session_name: str, output_dir: Path) -> Dict[str, Any]:
    """
    Create a debug session for comprehensive tracking.
    
    Args:
        session_name: Name for this debug session
        output_dir: Directory to save debug outputs
        
    Returns:
        Debug session context
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear previous debugging data
    debug_tracker.clear()
    debug_tracker.set_generation_id(session_name)
    
    return {
        "session_name": session_name,
        "output_dir": output_dir,
        "start_time": datetime.now(),
        "tracker": debug_tracker
    }


def finalize_debug_session(debug_session: Dict[str, Any]) -> None:
    """
    Finalize a debug session and save reports.
    
    Args:
        debug_session: Debug session context from create_debug_session
    """
    output_dir = debug_session["output_dir"]
    session_name = debug_session["session_name"]
    
    # Save comprehensive debug report
    report_path = output_dir / f"debug_report_{session_name}.json"
    debug_tracker.save_debug_report(report_path)
    
    # Print summary
    debug_tracker.print_summary()
    
    # Calculate total session time
    total_time = datetime.now() - debug_session["start_time"]
    print(f"\n⏱️  Total debug session time: {total_time.total_seconds():.1f} seconds")


# Convenience functions for common debugging tasks
def log_system_info():
    """Log comprehensive system information."""
    print("\n🖥️  System Information:")
    print(f"   Python: {torch.__version__}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # System memory
    sys_mem = psutil.virtual_memory()
    print(f"   System Memory: {sys_mem.total / 1024**3:.1f}GB total, {sys_mem.available / 1024**3:.1f}GB available")


def quick_tensor_check(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Quick tensor check with common issues."""
    print(f"\n🔍 Quick check: {name}")
    print(f"   Shape: {tensor.shape}, Device: {tensor.device}, Dtype: {tensor.dtype}")
    
    if tensor.numel() > 0:
        flat = tensor.flatten().float()
        nan_count = torch.sum(torch.isnan(flat))
        inf_count = torch.sum(torch.isinf(flat))
        
        if nan_count > 0:
            print(f"   ⚠️  WARNING: {nan_count} NaN values found!")
        if inf_count > 0:
            print(f"   ⚠️  WARNING: {inf_count} Inf values found!")
            
        print(f"   Range: [{torch.min(flat):.4f}, {torch.max(flat):.4f}]")
        print(f"   Mean: {torch.mean(flat):.4f}, Std: {torch.std(flat):.4f}")