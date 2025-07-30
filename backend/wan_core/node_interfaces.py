"""
WAN Video Node Interface Definitions

This module provides typed Protocol interfaces for all ComfyUI WAN video nodes
to enable IDE navigation, type checking, and clear API contracts.

Based on analysis of ComfyUI-WanVideoWrapper nodes.py and nodes_model_loading.py
"""

from typing import Protocol, Tuple, Dict, Any, Optional, List, Union
import torch


# =============================================================================
# Model Loading Node Interfaces
# =============================================================================

class LoadWanVideoT5TextEncoderProtocol(Protocol):
    """
    Loads T5 text encoder models for text conditioning.
    
    Location: nodes.py lines 275-382
    Purpose: Load and initialize T5 text encoder with configurable precision
    """
    
    def loadmodel(self, 
                  model_name: str,
                  precision: str = "bf16",  # "fp32", "fp16", "bf16"
                  load_device: str = "offload_device", 
                  quantization: str = "disabled") -> Tuple[Any]:
        """
        Load T5 text encoder model.
        
        Args:
            model_name: Filename of T5 model (e.g., "umt5-xxl-enc-bf16.safetensors")
            precision: Model precision - "fp32", "fp16", "bf16"
            load_device: Device loading strategy
            quantization: Quantization mode - "disabled" typically
            
        Returns:
            Tuple containing loaded T5 text encoder model
        """
        ...


class WanVideoModelLoaderProtocol(Protocol):
    """
    Loads the main WAN video diffusion model.
    
    Location: nodes_model_loading.py lines 457-972
    Purpose: Load core diffusion transformer with optimizations
    """
    
    def loadmodel(self,
                  model: str,
                  base_precision: str = "bf16",
                  load_device: str = "offload_device",
                  quantization: str = "fp8_e4m3fn",
                  compile_args: Optional[Any] = None,
                  attention_mode: str = "sdpa",
                  block_swap_args: Optional[Any] = None,
                  lora: Optional[Any] = None,
                  vram_management_args: Optional[Any] = None,
                  vace_model: Optional[Any] = None,
                  **kwargs) -> Tuple[Any]:
        """
        Load main WAN video diffusion model.
        
        Args:
            model: Model filename (e.g., "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors")
            base_precision: Base precision - "bf16", "fp16", "fp32"
            quantization: Quantization mode - "fp8_e4m3fn", "disabled"
            attention_mode: Attention implementation - "sdpa", "flash_attention"
            block_swap_args: Memory optimization settings
            lora: LoRA configuration from WanVideoLoraSelect
            vace_model: VACE model path configuration
            
        Returns:
            Tuple containing loaded model with patcher
        """
        ...


class WanVideoVAELoaderProtocol(Protocol):
    """
    Loads Video VAE for pixel ↔ latent conversion.
    
    Location: nodes_model_loading.py lines 976-1018
    Purpose: Load VAE for encoding/decoding between pixel and latent space
    """
    
    def loadmodel(self,
                  model_name: str,
                  precision: str = "bf16") -> Tuple[Any]:
        """
        Load Video VAE model.
        
        Args:
            model_name: VAE model filename (e.g., "Wan2_1_VAE_bf16.safetensors")
            precision: Model precision - "fp16", "fp32", "bf16"
            
        Returns:
            Tuple containing loaded VAE model
        """
        ...


class WanVideoLoraSelectProtocol(Protocol):
    """
    Selects and configures LoRA models.
    
    Location: nodes_model_loading.py lines 245-331
    Purpose: Select LoRA models from ComfyUI/models/loras directory
    """
    
    def getlorapath(self,
                    lora: str,
                    strength: float = 1.0,
                    unique_id: Optional[Any] = None,
                    blocks: Dict = {},
                    prev_lora: Optional[Any] = None,
                    low_mem_load: bool = False) -> Tuple[Any]:
        """
        Select and configure LoRA model.
        
        Args:
            lora: LoRA filename from loras directory
            strength: LoRA application strength (0.0 - 2.0 typically)
            blocks: Block-specific LoRA application settings
            prev_lora: Previous LoRA config for chaining
            low_mem_load: Enable low memory loading
            
        Returns:
            Tuple containing LoRA configuration list
        """
        ...


class WanVideoVACEModelSelectProtocol(Protocol):
    """
    Selects VACE (Video-Aware Conditioning Enhancement) model.
    
    Location: nodes_model_loading.py lines 405-424  
    Purpose: Select VACE model when not included in main model
    """
    
    def getvacepath(self, vace_model: str) -> Tuple[str]:
        """
        Select VACE model path.
        
        Args:
            vace_model: VACE model filename (e.g., "Wan2_1-VACE_module_14B_bf16.safetensors")
            
        Returns:
            Tuple containing VACE model path configuration
        """
        ...


# =============================================================================
# Text Processing Node Interfaces  
# =============================================================================

class WanVideoTextEncodeProtocol(Protocol):
    """
    Encodes text prompts into embeddings using T5.
    
    Location: nodes.py lines 429-508
    Purpose: Convert text prompts to embeddings for conditioning
    """
    
    def process(self,
                t5: Any,  # T5 model from LoadWanVideoT5TextEncoder
                positive_prompt: str,
                negative_prompt: str = "",
                force_offload: bool = True,
                model_to_offload: Optional[Any] = None) -> Tuple[Dict[str, torch.Tensor]]:
        """
        Encode text prompts to embeddings.
        
        Args:
            t5: Loaded T5 text encoder model
            positive_prompt: Main prompt text
            negative_prompt: Negative prompt for guidance
            force_offload: Whether to offload model after encoding
            
        Returns:
            Tuple containing dict with 'prompt_embeds' and 'negative_prompt_embeds'
        """
        ...


# =============================================================================
# Configuration Node Interfaces
# =============================================================================

class WanVideoBlockSwapProtocol(Protocol):
    """
    Configures block swapping for memory optimization.
    
    Location: nodes.py lines 58-79
    Purpose: Configure transformer block swapping to CPU memory
    """
    
    def setargs(self,
                blocks_to_swap: int = 30,
                offload_img_emb: bool = False,
                offload_txt_emb: bool = False,
                use_non_blocking: bool = True,
                vace_blocks_to_swap: int = 0) -> Tuple[Dict[str, Any]]:
        """
        Configure block swapping parameters.
        
        Args:
            blocks_to_swap: Number of transformer blocks to swap to CPU
            offload_img_emb: Whether to offload image embeddings
            offload_txt_emb: Whether to offload text embeddings
            use_non_blocking: Use non-blocking memory transfers
            vace_blocks_to_swap: Number of VACE blocks to swap
            
        Returns:
            Tuple containing block swap configuration dict
        """
        ...


class WanVideoEnhanceAVideoProtocol(Protocol):
    """
    Configures FETA (Enhance-A-Video) parameters.
    
    Location: nodes.py lines 193-210
    Purpose: Configure video enhancement during generation
    """
    
    def setargs(self,
                weight: float = 2.0,
                start_percent: float = 0.0,
                end_percent: float = 1.0) -> Tuple[Dict[str, Any]]:
        """
        Configure FETA enhancement parameters.
        
        Args:
            weight: Enhancement strength
            start_percent: Start percentage of generation to apply
            end_percent: End percentage of generation to apply
            
        Returns:
            Tuple containing FETA configuration dict
        """
        ...


class WanVideoSLGProtocol(Protocol):
    """
    Configures SLG (Skip Layer Guidance) for speed optimization.
    
    Location: nodes.py lines 1247-1271
    Purpose: Skip unconditional computation on selected blocks
    """
    
    def process(self,
                blocks: str = "8",
                start_percent: float = 0.1,
                end_percent: float = 1.0) -> Tuple[Dict[str, Any]]:
        """
        Configure SLG parameters.
        
        Args:
            blocks: Comma-separated block indices to apply SLG
            start_percent: Start percentage for SLG application
            end_percent: End percentage for SLG application
            
        Returns:
            Tuple containing SLG configuration with parsed block list
        """
        ...


class WanVideoExperimentalArgsProtocol(Protocol):
    """
    Configures experimental features.
    
    Location: nodes.py lines 1705-1728
    Purpose: Enable experimental generation features
    """
    
    def process(self,
                video_attention_split_steps: str = "",
                cfg_zero_star: bool = True,
                use_zero_init: bool = False,
                zero_star_steps: int = 0,
                use_fresca: bool = False,
                fresca_scale_low: float = 1.0,
                fresca_scale_high: float = 1.2,
                fresca_freq_cutoff: int = 20) -> Tuple[Dict[str, Any]]:
        """
        Configure experimental features.
        
        Args:
            cfg_zero_star: Enable CFG-Zero-star optimization
            use_fresca: Enable FreSca frequency scaling
            fresca_scale_low: FreSca low frequency scale
            fresca_scale_high: FreSca high frequency scale
            fresca_freq_cutoff: FreSca frequency cutoff
            
        Returns:
            Tuple containing experimental configuration dict
        """
        ...


# =============================================================================
# Generation Pipeline Node Interfaces
# =============================================================================

class WanVideoVACEEncodeProtocol(Protocol):
    """
    Encodes video frames for VACE conditioning.
    
    Location: nodes.py lines 1274-1450+
    Purpose: Process video frames and reference images for conditioning
    """
    
    def process(self,
                vae: Any,  # VAE model from WanVideoVAELoader
                width: int,
                height: int,
                num_frames: int,
                strength: float = 1.0,
                vace_start_percent: float = 0.0,
                vace_end_percent: float = 1.0,
                input_frames: Optional[torch.Tensor] = None,
                ref_images: Optional[torch.Tensor] = None,
                input_masks: Optional[torch.Tensor] = None,
                prev_vace_embeds: Optional[Any] = None,
                tiled_vae: bool = False,
                **kwargs) -> Tuple[torch.Tensor]:
        """
        Encode video frames for VACE conditioning.
        
        Args:
            vae: Loaded VAE model
            width: Target width for generation
            height: Target height for generation
            num_frames: Number of frames to process
            strength: VACE conditioning strength
            input_frames: Input video frames tensor (F, H, W, C)
            ref_images: Reference images tensor (N, H, W, C)
            input_masks: Input masks tensor (F, H, W)
            
        Returns:
            Tuple containing VACE embeddings tensor
        """
        ...


class WanVideoSamplerProtocol(Protocol):
    """
    Core sampling engine for diffusion generation.
    
    Location: nodes.py lines 1753-2800+
    Purpose: Run diffusion sampling process with conditioning
    """
    
    def process(self,
                model: Any,  # Model from WanVideoModelLoader
                image_embeds: torch.Tensor,  # From WanVideoVACEEncode
                shift: float = 10.0,
                steps: int = 6,
                cfg: float = 1.0,
                seed: int = 0,
                scheduler: str = "euler",
                riflex_freq_index: int = 0,
                text_embeds: Optional[Dict[str, torch.Tensor]] = None,
                force_offload: bool = True,
                samples: Optional[torch.Tensor] = None,
                feta_args: Optional[Dict[str, Any]] = None,
                denoise_strength: float = 1.0,
                context_options: Optional[Any] = None,
                cache_args: Optional[Any] = None,
                teacache_args: Optional[Any] = None,
                flowedit_args: Optional[Any] = None,
                batched_cfg: bool = False,
                slg_args: Optional[Dict[str, Any]] = None,
                rope_function: str = "comfy",
                loop_args: Optional[Any] = None,
                experimental_args: Optional[Dict[str, Any]] = None,
                **kwargs) -> Tuple[torch.Tensor]:
        """
        Run diffusion sampling process.
        
        Args:
            model: Loaded WAN video model
            image_embeds: VACE embeddings from video/images
            shift: Timestep shifting parameter
            steps: Number of denoising steps
            cfg: Classifier-free guidance scale
            seed: Random seed for generation
            scheduler: Diffusion scheduler name
            text_embeds: Text embeddings from WanVideoTextEncode
            feta_args: FETA enhancement configuration
            slg_args: SLG optimization configuration  
            experimental_args: Experimental feature configuration
            
        Returns:
            Tuple containing generated latent samples
        """
        ...


class WanVideoDecodeProtocol(Protocol):
    """
    Decodes latents to images using VAE.
    
    Location: nodes.py lines 3484-3583+
    Purpose: Convert latent samples back to pixel images
    """
    
    def decode(self,
               vae: Any,  # VAE model from WanVideoVAELoader
               samples: torch.Tensor,  # Latents from WanVideoSampler
               enable_vae_tiling: bool = False,
               tile_x: int = 272,
               tile_y: int = 272,
               tile_stride_x: int = 144,
               tile_stride_y: int = 128,
               normalization: str = "default") -> Tuple[torch.Tensor]:
        """
        Decode latent samples to images.
        
        Args:
            vae: Loaded VAE model
            samples: Latent samples tensor from sampling
            enable_vae_tiling: Use tiled decoding for memory efficiency
            tile_x: Tile width for tiled decoding
            tile_y: Tile height for tiled decoding
            tile_stride_x: Tile stride in X direction
            tile_stride_y: Tile stride in Y direction
            normalization: Normalization mode
            
        Returns:
            Tuple containing decoded image tensor (F, H, W, C)
        """
        ...


# =============================================================================
# Node Registry Protocol
# =============================================================================

class NodeRegistryProtocol(Protocol):
    """
    Protocol for node registry that provides access to all WAN video nodes.
    """
    
    # Model loading nodes
    text_encoder_loader: LoadWanVideoT5TextEncoderProtocol
    model_loader: WanVideoModelLoaderProtocol
    vae_loader: WanVideoVAELoaderProtocol
    lora_selector: WanVideoLoraSelectProtocol
    vace_model_selector: WanVideoVACEModelSelectProtocol
    
    # Text processing nodes
    text_encoder: WanVideoTextEncodeProtocol
    
    # Configuration nodes
    block_swap: WanVideoBlockSwapProtocol
    enhance_a_video: WanVideoEnhanceAVideoProtocol
    slg: WanVideoSLGProtocol
    experimental_args: WanVideoExperimentalArgsProtocol
    
    # Generation pipeline nodes
    vace_encoder: WanVideoVACEEncodeProtocol
    sampler: WanVideoSamplerProtocol
    decoder: WanVideoDecodeProtocol


# =============================================================================
# Type Aliases for Common Data Types
# =============================================================================

# Common tensor shapes used in WAN video generation
VideoFramesTensor = torch.Tensor  # Shape: (num_frames, height, width, channels)
VideoMasksTensor = torch.Tensor   # Shape: (num_frames, height, width)
LatentsTensor = torch.Tensor      # Shape: (num_frames, latent_height, latent_width, latent_channels)
ImageEmbedsTensor = torch.Tensor  # Shape: Variable based on VACE encoding
TextEmbedsTensor = Dict[str, torch.Tensor]  # Keys: 'prompt_embeds', 'negative_prompt_embeds'

# Configuration types
LoRAConfig = List[Dict[str, Any]]
BlockSwapConfig = Dict[str, Any]
FETAConfig = Dict[str, Any] 
SLGConfig = Dict[str, Any]
ExperimentalConfig = Dict[str, Any]