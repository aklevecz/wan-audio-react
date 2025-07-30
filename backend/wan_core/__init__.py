"""
WAN Core Module

Provides typed interfaces, explicit imports, and utilities for WAN video generation
to improve code traceability and maintainability.

This module replaces the dynamic loading approach with explicit typed interfaces,
enabling IDE navigation, type checking, and comprehensive debugging capabilities.
"""

# Core registry and interfaces
from .node_interfaces import *
from .comfyui_nodes import ComfyUINodeRegistry, create_legacy_wan_module

# Debugging and introspection utilities
from .debugging_utils import (
    DebugTracker,
    ModelStateInspector, 
    TensorAnalyzer,
    debug_tracker,
    debug_node_call,
    create_debug_session,
    finalize_debug_session,
    log_system_info,
    quick_tensor_check
)

__all__ = [
    # Core registry
    'ComfyUINodeRegistry',
    'create_legacy_wan_module',
    
    # Node interface protocols
    'LoadWanVideoT5TextEncoderProtocol',
    'WanVideoModelLoaderProtocol',
    'WanVideoVAELoaderProtocol', 
    'WanVideoLoraSelectProtocol',
    'WanVideoVACEModelSelectProtocol',
    'WanVideoTextEncodeProtocol',
    'WanVideoBlockSwapProtocol',
    'WanVideoEnhanceAVideoProtocol',
    'WanVideoSLGProtocol',
    'WanVideoExperimentalArgsProtocol',
    'WanVideoVACEEncodeProtocol',
    'WanVideoSamplerProtocol',
    'WanVideoDecodeProtocol',
    'NodeRegistryProtocol',
    
    # Type aliases
    'VideoFramesTensor',
    'VideoMasksTensor',
    'LatentsTensor', 
    'ImageEmbedsTensor',
    'TextEmbedsTensor',
    'LoRAConfig',
    'BlockSwapConfig',
    'FETAConfig',
    'SLGConfig',
    'ExperimentalConfig',
    
    # Debugging utilities
    'DebugTracker',
    'ModelStateInspector',
    'TensorAnalyzer', 
    'debug_tracker',
    'debug_node_call',
    'create_debug_session',
    'finalize_debug_session',
    'log_system_info',
    'quick_tensor_check',
]

# Version info
__version__ = "1.0.0"
__author__ = "WAN Video Generation Team"
__description__ = "Typed interfaces and debugging utilities for WAN video generation"