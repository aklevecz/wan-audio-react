"""
ComfyUI Node Registry

This module provides explicit imports and typed access to all ComfyUI WAN video nodes,
replacing the dynamic module loading approach for better IDE support and traceability.
"""

import sys
import importlib.util
from pathlib import Path
from typing import Optional, Any
import types

from .node_interfaces import (
    LoadWanVideoT5TextEncoderProtocol,
    WanVideoModelLoaderProtocol, 
    WanVideoVAELoaderProtocol,
    WanVideoLoraSelectProtocol,
    WanVideoVACEModelSelectProtocol,
    WanVideoTextEncodeProtocol,
    WanVideoBlockSwapProtocol,
    WanVideoEnhanceAVideoProtocol,
    WanVideoSLGProtocol,
    WanVideoExperimentalArgsProtocol,
    WanVideoVACEEncodeProtocol,
    WanVideoSamplerProtocol,
    WanVideoDecodeProtocol,
    NodeRegistryProtocol
)


class ComfyUINodeRegistry:
    """
    Explicit node registry that provides typed access to all WAN video ComfyUI nodes.
    
    This replaces the dynamic `self.wan.*` access pattern with explicit, typed node instances
    that can be traced by IDEs and type checkers.
    """
    
    def __init__(self, comfy_root: str = "../ComfyUI"):
        """
        Initialize the node registry by loading ComfyUI nodes.
        
        Args:
            comfy_root: Path to ComfyUI root directory
        """
        self.comfy_root = Path(comfy_root)
        self._nodes_module = None
        self._setup_comfyui_environment()
        self._load_wan_nodes()
        self._initialize_node_instances()
    
    def _setup_comfyui_environment(self):
        """Setup ComfyUI environment for node loading."""
        COMFY_ROOT = self.comfy_root
        CUSTOM_NODES = COMFY_ROOT / "custom_nodes"
        
        # Add paths for imports
        sys.path.insert(0, str(COMFY_ROOT))
        sys.path.insert(0, str(CUSTOM_NODES))
        
        # Import ComfyUI's core systems or create stubs
        try:
            import comfy.model_management as mm
            import comfy.utils
            import folder_paths
            import server
            import app.frontend_management
            import app.user_manager
            import utils.install_util
            
            self.mm = mm
            print("[INFO] ComfyUI systems imported successfully")
            
        except ImportError as e:
            print(f"[WARNING] Creating stubs for missing ComfyUI systems: {e}")
            self._create_comfyui_stubs()
    
    def _create_comfyui_stubs(self):
        """Create stubs for missing ComfyUI modules."""
        # Minimal memory management stub
        class FallbackMM:
            @staticmethod
            def soft_empty_cache():
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            @staticmethod
            def unload_all_models():
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.mm = FallbackMM()
        
        # Create module stubs as needed
        if 'comfy.model_management' not in sys.modules:
            sys.modules['comfy.model_management'] = types.ModuleType('comfy.model_management')
        
        # Create other stubs if needed
        self._create_additional_stubs()
    
    def _create_additional_stubs(self):
        """Create additional module stubs for ComfyUI dependencies."""
        # Server stubs
        if 'server' not in sys.modules:
            server_module = types.ModuleType('server')
            class PromptServerStub:
                instance = None
                client_id = 0
                class BinaryEventTypes:
                    PREVIEW_IMAGE = 0
                def send_sync(self, *args, **kwargs):
                    return None
            
            setattr(server_module, 'PromptServer', PromptServerStub)
            PromptServerStub.instance = PromptServerStub()
            sys.modules['server'] = server_module
        
        # Utils stubs
        if 'utils.install_util' not in sys.modules:
            utils_install_util = types.ModuleType('utils.install_util')
            setattr(utils_install_util, 'get_missing_requirements_message', lambda *args, **kwargs: "")
            sys.modules['utils.install_util'] = utils_install_util
            
            utils_root = types.ModuleType('utils')
            setattr(utils_root, 'install_util', utils_install_util)
            sys.modules['utils'] = utils_root
    
    def _load_wan_nodes(self):
        """Load WAN video nodes from ComfyUI-WanVideoWrapper."""
        wan_nodes_py = self.comfy_root / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "nodes.py"
        wan_model_loading_py = self.comfy_root / "custom_nodes" / "ComfyUI-WanVideoWrapper" / "nodes_model_loading.py"
        
        if not wan_nodes_py.exists():
            raise FileNotFoundError(f"WAN nodes file not found: {wan_nodes_py}")
        if not wan_model_loading_py.exists():
            raise FileNotFoundError(f"WAN model loading nodes file not found: {wan_model_loading_py}")
        
        # Load main nodes.py
        spec = importlib.util.spec_from_file_location(
            "wan_nodes", wan_nodes_py, submodule_search_locations=[str(wan_nodes_py.parent)]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create spec for WAN nodes at {wan_nodes_py}")
        
        module = importlib.util.module_from_spec(spec)
        module.__package__ = "wan_nodes"
        module.__path__ = [str(wan_nodes_py.parent)]
        sys.modules["wan_nodes"] = module
        spec.loader.exec_module(module)
        
        # Load model loading nodes
        spec_model_loading = importlib.util.spec_from_file_location(
            "wan_nodes_model_loading", wan_model_loading_py, 
            submodule_search_locations=[str(wan_model_loading_py.parent)]
        )
        if spec_model_loading is None or spec_model_loading.loader is None:
            raise ImportError(f"Cannot create spec for WAN model loading nodes at {wan_model_loading_py}")
        
        model_loading_module = importlib.util.module_from_spec(spec_model_loading)
        model_loading_module.__package__ = "wan_nodes_model_loading"
        sys.modules["wan_nodes_model_loading"] = model_loading_module
        spec_model_loading.loader.exec_module(model_loading_module)
        
        # Merge model loading classes into main module
        for attr_name in dir(model_loading_module):
            if not attr_name.startswith('_'):
                attr = getattr(model_loading_module, attr_name)
                if isinstance(attr, type):  # Only copy classes
                    setattr(module, attr_name, attr)
        
        self._nodes_module = module
        
        # Disable latent preview to avoid server issues
        self._disable_latent_preview()
        print("[INFO] WAN video nodes loaded successfully")
    
    def _disable_latent_preview(self):
        """Disable latent preview to avoid server import issues."""
        try:
            lp = importlib.import_module("ComfyUI-WanVideoWrapper.latent_preview")
            def _disabled_callback(model, steps, x0_output_dict=None):
                def _cb(step, x0, x, total_steps):
                    return None
                return _cb
            setattr(lp, 'prepare_callback', _disabled_callback)
        except Exception:
            # Ignore if latent preview module doesn't exist
            pass
    
    def _initialize_node_instances(self):
        """Initialize typed node instances."""
        if self._nodes_module is None:
            raise RuntimeError("Nodes module not loaded")
        
        # Model loading nodes
        self.text_encoder_loader = self._nodes_module.LoadWanVideoT5TextEncoder
        self.model_loader = self._nodes_module.WanVideoModelLoader  
        self.vae_loader = self._nodes_module.WanVideoVAELoader
        self.lora_selector = self._nodes_module.WanVideoLoraSelect
        self.vace_model_selector = self._nodes_module.WanVideoVACEModelSelect
        
        # Text processing nodes
        self.text_encoder = self._nodes_module.WanVideoTextEncode
        
        # Configuration nodes
        self.block_swap = self._nodes_module.WanVideoBlockSwap
        self.enhance_a_video = self._nodes_module.WanVideoEnhanceAVideo
        self.slg = self._nodes_module.WanVideoSLG
        self.experimental_args = self._nodes_module.WanVideoExperimentalArgs
        
        # Generation pipeline nodes
        self.vace_encoder = self._nodes_module.WanVideoVACEEncode
        self.sampler = self._nodes_module.WanVideoSampler
        self.decoder = self._nodes_module.WanVideoDecode
        
        print("[INFO] Node instances initialized successfully")
    
    # =============================================================================
    # Convenience methods for node creation (maintains backward compatibility)
    # =============================================================================
    
    def create_text_encoder_loader(self) -> LoadWanVideoT5TextEncoderProtocol:
        """Create a new T5 text encoder loader instance."""
        return self.text_encoder_loader()
    
    def create_model_loader(self) -> WanVideoModelLoaderProtocol:
        """Create a new WAN video model loader instance."""
        return self.model_loader()
    
    def create_vae_loader(self) -> WanVideoVAELoaderProtocol:
        """Create a new VAE loader instance."""
        return self.vae_loader()
    
    def create_lora_selector(self) -> WanVideoLoraSelectProtocol:
        """Create a new LoRA selector instance."""
        return self.lora_selector()
    
    def create_vace_model_selector(self) -> WanVideoVACEModelSelectProtocol:
        """Create a new VACE model selector instance."""
        return self.vace_model_selector()
    
    def create_text_encoder(self) -> WanVideoTextEncodeProtocol:
        """Create a new text encoder instance."""
        return self.text_encoder()
    
    def create_block_swap(self) -> WanVideoBlockSwapProtocol:
        """Create a new block swap configuration instance."""
        return self.block_swap()
    
    def create_enhance_a_video(self) -> WanVideoEnhanceAVideoProtocol:
        """Create a new FETA configuration instance."""
        return self.enhance_a_video()
    
    def create_slg(self) -> WanVideoSLGProtocol:
        """Create a new SLG configuration instance."""
        return self.slg()
    
    def create_experimental_args(self) -> WanVideoExperimentalArgsProtocol:
        """Create a new experimental args configuration instance."""
        return self.experimental_args()
    
    def create_vace_encoder(self) -> WanVideoVACEEncodeProtocol:
        """Create a new VACE encoder instance."""
        return self.vace_encoder()
    
    def create_sampler(self) -> WanVideoSamplerProtocol:
        """Create a new sampler instance."""
        return self.sampler()
    
    def create_decoder(self) -> WanVideoDecodeProtocol:
        """Create a new decoder instance."""
        return self.decoder()
    
    # =============================================================================
    # Utility methods
    # =============================================================================
    
    def get_available_nodes(self) -> list[str]:
        """Get list of all available node class names."""
        if self._nodes_module is None:
            return []
        
        nodes = []
        for attr_name in dir(self._nodes_module):
            if not attr_name.startswith('_'):
                attr = getattr(self._nodes_module, attr_name)
                if isinstance(attr, type):
                    nodes.append(attr_name)
        return sorted(nodes)
    
    def get_node_class(self, class_name: str) -> Optional[type]:
        """Get a node class by name for dynamic access."""
        if self._nodes_module is None:
            return None
        
        return getattr(self._nodes_module, class_name, None)
    
    def __getattr__(self, name: str) -> Any:
        """
        Backward compatibility: Allow access to nodes via self.wan.NodeName() pattern.
        
        This enables gradual migration from the old dynamic access pattern.
        """
        if self._nodes_module is None:
            raise AttributeError(f"Nodes module not loaded, cannot access {name}")
        
        if hasattr(self._nodes_module, name):
            return getattr(self._nodes_module, name)
        
        raise AttributeError(f"Node '{name}' not found in WAN video nodes")


# =============================================================================
# Legacy compatibility helper
# =============================================================================

def create_legacy_wan_module(comfy_root: str = "../ComfyUI") -> Any:
    """
    Create a legacy-compatible WAN module object for backward compatibility.
    
    This allows existing code using `self.wan.NodeName()` to continue working
    while providing a migration path to the typed registry.
    
    Args:
        comfy_root: Path to ComfyUI root directory
        
    Returns:
        Object that behaves like the old dynamic module but with typed backing
    """
    
    class LegacyWanModule:
        def __init__(self, registry: ComfyUINodeRegistry):
            self._registry = registry
        
        def __getattr__(self, name: str):
            return getattr(self._registry, name)
    
    registry = ComfyUINodeRegistry(comfy_root)
    return LegacyWanModule(registry)