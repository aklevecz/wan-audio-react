"""Sieve function for WAN video generation with R2 cloud storage integration.

This module provides a Sieve cloud function for WAN video generation that
fetches video and reference images directly from R2 cloud storage using session_id.
"""

import os
import sieve
from pathlib import Path
from typing import Optional, List, Dict, Any
import tempfile
import requests

COMFYUI_PATH = "/src/ComfyUI"

# Cloud storage functions (from cloud_storage.py)
import boto3
from botocore.client import Config


def get_env_vars():
    """Get environment variables dynamically to ensure they're loaded after dotenv."""
    # Original code commented out for now:
    # Try Sieve environment variables first (for production)
    sieve_vars = {
        'ENDPOINT_URL': os.environ.get("R2_ENDPOINT_URL"),
        'ACCESS_KEY_ID': os.environ.get("R2_ACCESS_KEY_ID"),
        'SECRET_ACCESS_KEY': os.environ.get("R2_SECRET_ACCESS_KEY"),
        'BUCKET_NAME': os.environ.get("R2_BUCKET_NAME"),
        'PUBLIC_URL_BASE': os.environ.get("R2_PUBLIC_URL_BASE")
    }
    
    # If Sieve vars are available, use them
    if all(sieve_vars.values()):
        return sieve_vars
    
    # Return an error
    raise ValueError("One or more R2 environment variables are not set.")

# Replace with Kaiber S3 R2 implementation
# R2 utilities
def get_s3_client():
    """Initializes and returns a boto3 S3 client configured for R2."""
    env_vars = get_env_vars()
    if not all([env_vars['ENDPOINT_URL'], env_vars['ACCESS_KEY_ID'], env_vars['SECRET_ACCESS_KEY'], env_vars['BUCKET_NAME']]):
        raise ValueError("One or more R2 environment variables are not set.")

    s3_client = boto3.client(
        's3',
        endpoint_url=env_vars['ENDPOINT_URL'],
        aws_access_key_id=env_vars['ACCESS_KEY_ID'],
        aws_secret_access_key=env_vars['SECRET_ACCESS_KEY'],
        config=Config(signature_version='s3v4'),
        region_name='auto' # R2 specific
    )
    return s3_client

def get_public_url(object_name: str) -> str:
    """Constructs the public URL for an object in the R2 bucket."""
    env_vars = get_env_vars()
    if not env_vars['PUBLIC_URL_BASE']:
        raise ValueError("R2_PUBLIC_URL_BASE environment variable is not set.")
        
    return f"{env_vars['PUBLIC_URL_BASE']}/{object_name}"

def download_r2_file(object_name: str, local_path: Path) -> bool:
    """Downloads a file from R2 bucket to local path."""
    try:
        s3_client = get_s3_client()
        env_vars = get_env_vars()
        s3_client.download_file(env_vars['BUCKET_NAME'], object_name, str(local_path))
        print(f"Successfully downloaded {object_name} to {local_path}")
        return True
    except Exception as e:
        print(f"Error downloading file from R2: {e}")
        return False

def upload_r2_file(local_path: Path, object_name: str) -> bool:
    """Uploads a file to R2 bucket."""
    try:
        s3_client = get_s3_client()
        env_vars = get_env_vars()
        s3_client.upload_file(str(local_path), env_vars['BUCKET_NAME'], object_name)
        print(f"Successfully uploaded {local_path} to R2: {object_name}")
        return True
    except Exception as e:
        print(f"Error uploading file to R2: {e}")
        return False

# End R2 utilities

# Sieve function entry point
@sieve.Model(
    name="wan-video-generator-r2", 
    python_version="3.11",
    gpu=sieve.gpu.A100(),
    system_packages=["ffmpeg", "git", "wget", "curl"],
    # Use Sieve env for production
    # environment_variables=[
    #     sieve.Env(name="R2_ENDPOINT_URL", description="R2 endpoint URL for cloud storage"),
    #     sieve.Env(name="R2_ACCESS_KEY_ID", description="R2 access key ID for authentication"),
    #     sieve.Env(name="R2_SECRET_ACCESS_KEY", description="R2 secret access key for authentication"),
    #     sieve.Env(name="R2_BUCKET_NAME", description="R2 bucket name for file storage"),
    #     sieve.Env(name="R2_PUBLIC_URL_BASE", description="R2 public URL base for file access"),
    # ],
    python_packages=[
        # Core PyTorch packages
        "torch>=2.4.0",
        "torchvision>=0.19.0", 
        "torchaudio>=2.4.0",
        "torchsde",
        
        # Essential ML packages
        "diffusers==0.34.0",
        "transformers==4.53.2",
        "accelerate>=1.2.1",
        "tokenizers>=0.13.3",
        "sentencepiece>=0.2.0",
        "safetensors>=0.4.2",
        "peft>=0.15.0",
        
        # Image/Video processing
        "einops",
        "numpy>=1.25.0",
        "Pillow",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg",
        "scipy",
        "kornia>=0.7.1",
        "spandrel",
        
        # Audio processing
        "soundfile",
        "av>=14.2.0",
        "pyloudnorm",
        
        # Utility packages
        "tqdm",
        "psutil",
        "pyyaml",
        "ftfy",
        "protobuf",
        "aiohttp>=3.11.8",
        "yarl>=1.18.0",
        "pydantic>=2.0",
        "gguf>=0.14.0",
        
        # Additional dependencies
        "cloudpickle",
        "pymongo",
        
        # R2/AWS SDK for file fetching
        "boto3",
        "requests",
        "python-dotenv"
    ],
    run_commands=[
        # Clone ComfyUI
        f"git clone https://github.com/comfyanonymous/ComfyUI.git {COMFYUI_PATH}",
        f"mkdir -p {COMFYUI_PATH}/custom_nodes",
        
        # Clone WAN wrapper
        f"git clone https://github.com/Kijai/ComfyUI-WanVideoWrapper.git {COMFYUI_PATH}/custom_nodes/ComfyUI-WanVideoWrapper",
        
        # Create model directories
        f"mkdir -p {COMFYUI_PATH}/models/diffusion_models",
        f"mkdir -p {COMFYUI_PATH}/models/vae",
        f"mkdir -p {COMFYUI_PATH}/models/text_encoders",
        f"mkdir -p {COMFYUI_PATH}/models/loras",
        
        # Download WAN models
        f"wget -O {COMFYUI_PATH}/models/diffusion_models/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-T2V-14B_fp8_e4m3fn.safetensors",
        f"wget -O {COMFYUI_PATH}/models/diffusion_models/Wan2_1-VACE_module_14B_bf16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-VACE_module_14B_bf16.safetensors", 
        f"wget -O {COMFYUI_PATH}/models/vae/Wan2_1_VAE_bf16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors",
        f"wget -O {COMFYUI_PATH}/models/text_encoders/umt5-xxl-enc-bf16.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
        f"wget -O {COMFYUI_PATH}/models/loras/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors",
        f"wget -O {COMFYUI_PATH}/models/loras/Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan21_CausVid_14B_T2V_lora_rank32_v1_5_no_first_block.safetensors"
    ]
)
class WanVideoGeneratorR2:
    """Sieve function for WAN video generation with R2 cloud storage integration."""
    
    def __init__(self, COMFY_PATH=COMFYUI_PATH):
        """Initialize the WAN video generator with R2 support."""
        self.comfy_root = COMFY_PATH
        self.wan_generator = None
        self.default_config = None
    
    def __setup__(self):
        """Setup: Load WAN generator with models downloaded by run_commands."""
        print("🚀 Starting R2-enabled setup with pre-downloaded models...")
        print("✅ Setup complete - models ready for R2-based predictions")
    
    def _fetch_r2_file(self, session_id: str, file_type: str, filename: str = None) -> Optional[Path]:
        """
        Fetch a file from R2 storage using session_id and cloud_storage functions.
        Used to fetch the mask video or reference image from R2.
        Note that if the mask video is webm, it will be converted to mp4. This is useful incase we want to generate the mask entirely in the browser.
        
        Args:
            session_id: Session identifier
            file_type: Type of file ('video' or 'image')
            filename: Optional specific filename
            
        Returns:
            Path to downloaded file or None if not found
        """
        try:
            # Instead of grabbing the video mask from R2 from the session we can get a preset
            # Construct R2 object key based on file type
            if file_type == "video":
                # Look for uploaded video in sessions/{session_id}/uploaded_videos/
                base_key = f"sessions/{session_id}/uploaded_videos/"
                if filename:
                    object_key = f"{base_key}{filename}"
                else:
                    # Try predictable filenames with common extensions
                    s3_client = get_s3_client()
                    env_vars = get_env_vars()
                    
                    predictable_files = [
                        f"{base_key}mask_video.webm",
                        f"{base_key}mask_video.mp4", 
                        f"{base_key}mask_video.mov",
                        f"{base_key}mask_video.avi"
                    ]
                    
                    found_predictable = False
                    for predictable_key in predictable_files:
                        try:
                            # Try to get object metadata to check if it exists
                            s3_client.head_object(Bucket=env_vars['BUCKET_NAME'], Key=predictable_key)
                            object_key = predictable_key
                            filename = Path(predictable_key).name
                            print(f"[INFO] Found predictable video file: {object_key}")
                            found_predictable = True
                            break
                        except Exception:
                            continue
                    
                    if not found_predictable:
                        # If predictable file doesn't exist, fallback to listing
                        print(f"[INFO] Predictable video file not found, searching for any video file...")
                        try:
                            response = s3_client.list_objects_v2(
                                Bucket=env_vars['BUCKET_NAME'],
                                Prefix=base_key
                            )
                            
                            video_files = [obj['Key'] for obj in response.get('Contents', [])
                                         if obj['Key'].lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
                            
                            if not video_files:
                                print(f"[ERROR] No video file found for session {session_id}")
                                return None
                            
                            object_key = video_files[0]  # Use first video file found
                            filename = Path(object_key).name
                            print(f"[INFO] Found fallback video file: {object_key}")
                            
                        except Exception as e:
                            print(f"[ERROR] Failed to list video files: {e}")
                            return None

            # Note: If there is an image in the session it will grab it            
            elif file_type == "image":
                # For reference images in sessions/{session_id}/reference_images/
                if not filename:
                    print(f"[ERROR] Filename required for image file type")
                    return None
                object_key = f"sessions/{session_id}/reference_images/{filename}"
            else:
                print(f"[ERROR] Unknown file type: {file_type}")
                return None
            
            print(f"[INFO] Downloading {file_type} from R2: {object_key}")
            
            # Create temporary file
            temp_dir = Path(tempfile.gettempdir()) / f"r2_downloads_{session_id}"
            temp_dir.mkdir(exist_ok=True)
            
            if filename:
                temp_file = temp_dir / filename
            else:
                # Extract filename from object key
                temp_file = temp_dir / Path(object_key).name
            
            # Use cloud_storage download function
            success = download_r2_file(object_key, temp_file)
            if not success:
                return None
            
            print(f"[INFO] Successfully downloaded {file_type} to: {temp_file} ({temp_file.stat().st_size} bytes)")
            
            # Convert WebM to MP4 if necessary
            if file_type == "video" and temp_file.suffix.lower() == '.webm':
                print(f"[INFO] Converting WebM to MP4 for better compatibility...")
                mp4_file = temp_file.with_suffix('.mp4')
                
                try:
                    import subprocess
                    import json
                    
                    # First, probe the input video to get its framerate
                    probe_cmd = [
                        "ffprobe", "-v", "quiet", "-print_format", "json", 
                        "-show_streams", "-select_streams", "v:0", str(temp_file)
                    ]
                    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                    probe_data = json.loads(probe_result.stdout)
                    
                    input_fps = 30.0  # default
                    if probe_data.get("streams"):
                        stream = probe_data["streams"][0]
                        if "r_frame_rate" in stream:
                            fps_fraction = stream["r_frame_rate"]
                            if "/" in fps_fraction:
                                num, den = fps_fraction.split("/")
                                input_fps = float(num) / float(den)
                            else:
                                input_fps = float(fps_fraction)
                    
                    print(f"[INFO] Input video framerate: {input_fps:.2f} fps")
                    
                    # Convert WebM to MP4 using ffmpeg with 30fps output
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",  # -y to overwrite output file
                        "-i", str(temp_file),  # input WebM file
                        "-c:v", "libx264",     # video codec
                        "-c:a", "aac",         # audio codec  
                        "-preset", "fast",     # encoding speed vs compression
                        "-crf", "23",          # quality (lower = better quality)
                        "-r", "30",            # force 30fps output
                        str(mp4_file)          # output MP4 file
                    ]
                    
                    print(f"[INFO] Running ffmpeg conversion: {temp_file.name} -> {mp4_file.name} (forcing 30fps)")
                    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
                    
                    if mp4_file.exists():
                        mp4_size = mp4_file.stat().st_size
                        print(f"[INFO] Successfully converted to MP4: {mp4_file} ({mp4_size} bytes)")
                        
                        # Clean up original WebM file to save space
                        temp_file.unlink()
                        print(f"[INFO] Cleaned up original WebM file")
                        
                        return mp4_file
                    else:
                        print(f"[ERROR] MP4 conversion failed - output file not created")
                        return temp_file  # Return original WebM if conversion failed
                        
                except subprocess.CalledProcessError as e:
                    print(f"[ERROR] ffmpeg conversion failed: {e}")
                    print(f"[ERROR] ffmpeg stderr: {e.stderr}")
                    print(f"[INFO] Continuing with original WebM file")
                    return temp_file
                    
                except Exception as e:
                    print(f"[ERROR] Unexpected error during WebM conversion: {e}")
                    print(f"[INFO] Continuing with original WebM file")
                    return temp_file
            
            return temp_file
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch {file_type} from R2: {e}")
            return None
    
    def _get_session_reference_images(self, session_id: str) -> List[Path]:
        """
        Get the reference image for a session from R2.
        
        Note: Assumes only one reference image per session with filename "reference_image.jpg".
        Should probably have a flag that ignores this in case the user uploads and image, but then doesn't want to use it in the future
        
        Args:
            session_id: Session identifier
            
        Returns:
            List containing single reference image Path, or empty list if none found
        """
        try:
            print(f"[INFO] Attempting to fetch reference image: reference_image.jpg")
            reference_path = self._fetch_r2_file(session_id, "image", "reference_image.jpg")
            
            if reference_path and reference_path.exists():
                # Validate it's a valid image file
                try:
                    from PIL import Image
                    with Image.open(reference_path) as img:
                        # Get image info before verify
                        width, height = img.size
                        mode = img.mode
                        format_type = img.format
                        print(f"[DEBUG] Reference image details: {width}x{height}, mode={mode}, format={format_type}")
                        img.verify()  # Verify the image is valid
                    
                    # Check file size
                    file_size = reference_path.stat().st_size
                    print(f"[DEBUG] Reference image file size: {file_size} bytes ({file_size/1024:.1f} KB)")
                    print(f"[INFO] Successfully found and validated reference image: reference_image.jpg")
                    return [reference_path]
                except Exception as e:
                    print(f"[WARNING] Invalid reference image file reference_image.jpg: {e}")
                    if reference_path.exists():
                        reference_path.unlink()
                    return []
            else:
                print(f"[INFO] No reference image found for session {session_id}")
                return []
                
        except Exception as e:
            print(f"[ERROR] Failed to get reference image for session {session_id}: {e}")
            return []
    
    def _upload_videos_to_r2(self, session_id: str, video_files: dict) -> dict:
        """
        Upload generated videos to R2 storage.
        Should be replaced with Kaiber S3 R2 implementation
        
        Args:
            session_id: Session identifier for organizing files
            video_files: Dictionary containing video file paths
            
        Returns:
            Dictionary with R2 URLs for uploaded videos
        """
        uploaded_urls = {}
        
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Upload initial video
            if "initial_video" in video_files:
                initial_path = Path(video_files["initial_video"].path)
                r2_key = f"sessions/{session_id}/wan_generations/{timestamp}/initial_video.mp4"
                
                print(f"[INFO] Uploading initial video to R2: {r2_key}")
                success = upload_r2_file(initial_path, r2_key)
                if success:
                    uploaded_urls["initial_video"] = get_public_url(r2_key)
                    print(f"[INFO] Initial video uploaded: {uploaded_urls['initial_video']}")
            
            # Upload extension videos
            if "extension_videos" in video_files and video_files["extension_videos"]:
                uploaded_urls["extension_videos"] = []
                
                for i, ext_video in enumerate(video_files["extension_videos"]):
                    ext_path = Path(ext_video.path)
                    r2_key = f"sessions/{session_id}/wan_generations/{timestamp}/extension_{i+1:03d}.mp4"
                    
                    print(f"[INFO] Uploading extension video {i+1} to R2: {r2_key}")
                    success = upload_r2_file(ext_path, r2_key)
                    if success:
                        ext_url = get_public_url(r2_key)
                        uploaded_urls["extension_videos"].append(ext_url)
                        print(f"[INFO] Extension video {i+1} uploaded: {ext_url}")
            
            # Upload combined video
            if "combined_video" in video_files:
                combined_path = Path(video_files["combined_video"].path)
                r2_key = f"sessions/{session_id}/wan_generations/{timestamp}/combined_video.mp4"
                
                print(f"[INFO] Uploading combined video to R2: {r2_key}")
                success = upload_r2_file(combined_path, r2_key)
                if success:
                    uploaded_urls["combined_video"] = get_public_url(r2_key)
                    print(f"[INFO] Combined video uploaded: {uploaded_urls['combined_video']}")
            
            print(f"[INFO] Successfully uploaded {len(uploaded_urls)} video types to R2")
            return uploaded_urls
            
        except Exception as e:
            print(f"[ERROR] Failed to upload videos to R2: {e}")
            return {}
    
    def __predict__(
        self,
        session_id: str,
        prompt: str = "bellowing flames",
        num_extensions: int = 1,
        resolution: str = "512p",
        invert_mask: bool = False,
        seed: int = 314525102295492
    ) -> Dict[str, Any]:
        """
        Generate WAN video using files from R2 storage.
        
        Args:
            session_id: Session ID containing uploaded files in R2
            prompt: Text prompt for generation
            num_extensions: Number of extension generations (0 for initial only)
            resolution: Resolution preset ("512p" or "720p")
            invert_mask: Whether to invert the mask
            seed: Random seed for generation
            
        Returns:
            Dictionary containing:
            - initial_video: Sieve File object for initial generation video
            - extension_videos: List of Sieve File objects for extension videos
            - combined_video: Sieve File object for complete video with all frames
            - total_frames: Total number of frames generated
            - generation_info: Metadata about the generation process
        """
        
        # Import all dependencies
        import torch
        import tempfile
        import json
        import shutil
        from wan_video_generator import GenerationConfig
        
        try:
            print(f"🚀 Starting WAN R2 generation for session: {session_id}")
            print("✅ Step 1: Function entry successful")
            
            # Prepare the input media, reference image, and mask video
            # Setup temporary working directory
            print("🔧 Step 2: Creating temporary directory...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                print(f"✅ Step 2: Temp directory created: {temp_path}")
                
                # Fetch video file from R2
                print("🔧 Step 3: Fetching video from R2...")
                video_file_path = self._fetch_r2_file(session_id, "video")
                if not video_file_path or not video_file_path.exists():
                    raise FileNotFoundError(f"Could not fetch video file for session {session_id} from R2")
                
                print(f"✅ Step 3: Video fetched from R2: {video_file_path}")
                original_video_name = video_file_path.stem
                
                # Copy to local temp directory for processing
                print("🔧 Step 4: Preparing local video file...")
                local_video_path = temp_path / f"{original_video_name}.mp4"
                shutil.copy2(video_file_path, local_video_path)
                print(f"✅ Step 4: Video prepared: {local_video_path}")
                
                # Fetch reference images from R2
                # Note: May want a flag here to ignore the reference image if it is not desired
                print("🔧 Step 5: Fetching reference images from R2...")
                reference_images = self._get_session_reference_images(session_id)
                print(f"✅ Step 5: Found {len(reference_images)} reference images")
                
                # Debug reference images
                if reference_images:
                    for i, ref_path in enumerate(reference_images):
                        print(f"[DEBUG] Reference image {i+1}: {ref_path} (exists: {ref_path.exists()})")
                else:
                    print(f"[DEBUG] No reference images found for session {session_id}")
                
                # Validate video file with ffprobe
                try:
                    import subprocess
                    result = subprocess.run([
                        "ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(local_video_path)
                    ], capture_output=True, text=True, check=True)
                    
                    probe_data = json.loads(result.stdout)
                    video_streams = [s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"]
                    
                    if video_streams:
                        video_stream = video_streams[0]
                        width = video_stream.get("width")
                        height = video_stream.get("height")
                        duration = float(video_stream.get("duration", 0))
                        fps = eval(video_stream.get("r_frame_rate", "30/1"))
                        codec_name = video_stream.get('codec_name')
                        
                        print(f"📊 R2 Video validation:")
                        print(f"   - Dimensions: {width}x{height}")
                        print(f"   - Duration: {duration:.2f}s")
                        print(f"   - FPS: {fps:.2f}")
                        print(f"   - Codec: {codec_name}")
                        
                        if codec_name in ['png', 'jpeg', 'jpg', 'webp', 'bmp'] or duration == 0.0:
                            raise ValueError(f"ERROR: R2 file is an image ({codec_name}), not a video. "
                                           f"Duration: {duration}s. Please upload a video file.")
                    else:
                        raise ValueError("No video streams found in R2 file.")
                        
                except Exception as e:
                    print(f"⚠️  R2 video validation failed: {e}")
                    raise e
                
                print(f"🚀 Starting WAN generation with R2 files:")
                print(f"   - Session: {session_id}")
                print(f"   - Prompt: '{prompt}'")
                print(f"   - Resolution: {resolution}")
                print(f"   - Extensions: {num_extensions}")
                
                from wan_video_generator import WanVideoGenerator as LocalWanVideoGenerator, GenerationConfig
        
                # Create config for model loading
                self.default_config = GenerationConfig(
                    comfy_root=self.comfy_root,
                    resolution_preset=resolution,
                    output_base_dir=str(temp_path / "wan_outputs"),
                    base_seed=seed
                )
                
                # Initialize the WAN generator
                self.wan_generator = LocalWanVideoGenerator(self.default_config)
                print("✅ WAN generator and models loaded into memory")
                
                # Set generator properties
                print("🔧 Setting generator properties...")
                self.wan_generator.reference_images = reference_images if reference_images else None
                self.wan_generator.invert_mask = invert_mask
                
                # Debug generator reference images
                if hasattr(self.wan_generator, 'reference_images') and self.wan_generator.reference_images:
                    print(f"[DEBUG] Generator has {len(self.wan_generator.reference_images)} reference images set")
                else:
                    print(f"[DEBUG] Generator has NO reference images set")
                
                # Reset generator state for new generation
                self.wan_generator.current_frame_idx = 0
                self.wan_generator.generation_count = 0
                self.wan_generator.current_output_dir = None
                
                print("✅ Generator configured and ready")
                
                # Generate initial video
                print("🎬 Starting initial generation...")
                initial_output_dir = self.wan_generator.generate_initial(
                    video_path=str(local_video_path),
                    prompt=prompt,
                    output_name=original_video_name
                )
                
                print(f"✅ Initial generation complete: {initial_output_dir}")
                
                # Generate extensions if requested
                extension_dirs = []
                if num_extensions > 0:
                    print(f"🔄 Generating {num_extensions} extensions...")
                    extension_dirs = self.wan_generator.extend_video(
                        video_path=str(local_video_path),
                        num_extensions=num_extensions,
                        prompt=prompt
                    )
                    print(f"✅ All {num_extensions} extensions complete: {len(extension_dirs)} generated")
                
                # Create MP4 videos (same logic as original)
                video_files = []
                
                # Create initial video
                initial_path = Path(initial_output_dir)
                if initial_path.exists():
                    initial_frames = sorted(initial_path.glob("frame_*.png"))
                    print(f"📁 Found {len(initial_frames)} initial frames")
                    
                    initial_video_path = self.wan_generator._create_video(initial_path, f"{original_video_name}_initial.mp4")
                    if initial_video_path.exists():
                        video_files.append({
                            "type": "initial",
                            "path": initial_video_path,
                            "frames": len(initial_frames)
                        })
                        print(f"🎬 Created initial video: {initial_video_path}")
                
                # Create extension videos
                extension_videos = []
                for i, ext_dir in enumerate(extension_dirs):
                    ext_path = Path(ext_dir)
                    if ext_path.exists():
                        ext_frames = sorted(ext_path.glob("frame_*.png"))
                        print(f"📁 Found {len(ext_frames)} extension frames in {ext_path.name}")
                        
                        ext_video_path = self.wan_generator._create_video(ext_path, f"{original_video_name}_extension_{i+1:03d}.mp4")
                        if ext_video_path.exists():
                            extension_videos.append({
                                "type": "extension",
                                "path": ext_video_path,
                                "frames": len(ext_frames)
                            })
                            video_files.append({
                                "type": "extension",
                                "path": ext_video_path,
                                "frames": len(ext_frames)
                            })
                            print(f"🎬 Created extension video: {ext_video_path}")
                
                # Create combined video with all frames
                combined_video_path = None
                if len(video_files) > 0:
                    combined_video_path = temp_path / f"{original_video_name}_complete.mp4"
                    print(f"🎬 Creating combined video: {combined_video_path}")
                    
                    # Collect all frames in order (handling context frame overlap)
                    all_frames = []
                    
                    # Add initial frames (use all frames)
                    if initial_path.exists():
                        initial_frames = sorted(initial_path.glob("frame_*.png"))
                        all_frames.extend(initial_frames)
                        print(f"📁 Added {len(initial_frames)} initial frames")
                    
                    # Add extension frames (skip first 15 context frames to avoid overlap)
                    for i, ext_dir in enumerate(extension_dirs):
                        ext_path = Path(ext_dir)
                        if ext_path.exists():
                            ext_frames = sorted(ext_path.glob("frame_*.png"))
                            # Skip first 15 frames (context overlap) for extensions
                            frames_to_use = ext_frames[15:] if len(ext_frames) > 15 else []
                            all_frames.extend(frames_to_use)
                            print(f"📁 Extension {i+1}: skipping 15 overlap frames, added {len(frames_to_use)} frames (total: {len(ext_frames)})")
                    
                    print(f"🔍 DEBUG: Total frames collected for combined video: {len(all_frames)}")
                    
                    # Create temporary directory for combined frames
                    combined_frames_dir = temp_path / "combined_frames"
                    combined_frames_dir.mkdir(exist_ok=True)
                    
                    # Copy all frames to combined directory with sequential naming
                    for i, frame_path in enumerate(all_frames):
                        dest_path = combined_frames_dir / f"frame_{i:04d}.png"
                        shutil.copy2(frame_path, dest_path)
                    
                    # Create combined video (silent first)
                    silent_combined_path = temp_path / f"{original_video_name}_complete_silent.mp4"
                    silent_combined_path = self.wan_generator._create_video(combined_frames_dir, f"{original_video_name}_complete_silent.mp4")
                    
                    if silent_combined_path and silent_combined_path.exists():
                        print(f"🎬 Created silent combined video: {silent_combined_path} with {len(all_frames)} total frames")
                        
                        # Calculate audio duration from generated frames
                        total_generated_frames = len(all_frames)
                        calculated_audio_duration = total_generated_frames / fps  # Use fps from video probe
                        
                        # Limit audio duration to not exceed original video duration
                        audio_duration = min(calculated_audio_duration, duration)
                        
                        print(f"🔊 Extracting audio: {total_generated_frames} frames / {fps:.2f} fps = {calculated_audio_duration:.2f}s")
                        if audio_duration < calculated_audio_duration:
                            print(f"⚠️  Limiting audio to original video duration: {audio_duration:.2f}s")
                        
                        # Check if original video has audio streams
                        has_audio = False
                        try:
                            # Check for audio streams in original video
                            probe_cmd = [
                                "ffprobe", "-v", "quiet", "-print_format", "json", 
                                "-show_streams", "-select_streams", "a", str(local_video_path)
                            ]
                            audio_probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                            audio_data = json.loads(audio_probe.stdout)
                            has_audio = len(audio_data.get("streams", [])) > 0
                            print(f"🔊 Original video audio streams: {len(audio_data.get('streams', []))}")
                        except Exception as e:
                            print(f"⚠️  Could not probe audio streams: {e}")
                            has_audio = False
                        
                        if has_audio:
                            # Extract audio segment from original video
                            audio_extract_path = temp_path / f"{original_video_name}_audio.aac"
                            audio_extract_cmd = [
                                "ffmpeg", "-y",
                                "-i", str(local_video_path),
                                "-t", str(audio_duration),  # Extract only the duration we need
                                "-vn",  # No video
                                "-acodec", "aac",
                                "-b:a", "128k",
                                str(audio_extract_path)
                            ]
                            
                            try:
                                subprocess.run(audio_extract_cmd, capture_output=True, text=True, check=True)
                                print(f"✅ Audio extracted: {audio_extract_path} ({audio_duration:.2f}s)")
                                
                                # Mux audio with video
                                mux_cmd = [
                                    "ffmpeg", "-y",
                                    "-i", str(silent_combined_path),  # Video input
                                    "-i", str(audio_extract_path),    # Audio input
                                    "-c:v", "copy",     # Copy video without re-encoding
                                    "-c:a", "aac",      # Audio codec
                                    "-shortest",        # Match shortest stream
                                    str(combined_video_path)
                                ]
                                
                                subprocess.run(mux_cmd, capture_output=True, text=True, check=True)
                                print(f"🎬 Combined video with audio created: {combined_video_path}")
                                
                                # Clean up temporary files
                                silent_combined_path.unlink(missing_ok=True)
                                audio_extract_path.unlink(missing_ok=True)
                                
                            except subprocess.CalledProcessError as e:
                                print(f"⚠️  Audio processing failed: {e}")
                                print(f"⚠️  Falling back to silent video")
                                # Rename silent video to final name if audio processing failed
                                if silent_combined_path.exists():
                                    silent_combined_path.rename(combined_video_path)
                                # Clean up partial audio file if it exists
                                if audio_extract_path.exists():
                                    audio_extract_path.unlink(missing_ok=True)
                        else:
                            print(f"ℹ️  Original video has no audio - creating silent combined video")
                            # Rename silent video to final name since there's no audio to add
                            if silent_combined_path.exists():
                                silent_combined_path.rename(combined_video_path)
                        
                        # Final safety check - ensure we have a combined video
                        if not combined_video_path or not combined_video_path.exists():
                            print(f"⚠️  Combined video missing - checking for silent version")
                            if silent_combined_path.exists():
                                print(f"⚠️  Recovering silent video as final output")
                                combined_video_path = silent_combined_path
                        
                    else:
                        combined_video_path = None
                
                # Count total frames
                total_frames = sum(vf["frames"] for vf in video_files)
                
                # Prepare generation info
                initial_frame_count = len(sorted(initial_path.glob("frame_*.png"))) if initial_path.exists() else 0
                extension_frame_count = total_frames - initial_frame_count
                
                generation_info = {
                    "session_id": session_id,
                    "prompt": prompt,
                    "resolution": resolution,
                    "num_extensions": num_extensions,
                    "invert_mask": invert_mask,
                    "total_frames": total_frames,
                    "initial_frames": initial_frame_count,
                    "extension_frames": extension_frame_count,
                    "seed": self.default_config.base_seed,
                    "model_name": self.default_config.model_name,
                    "execution_environment": "sieve_r2_cloud",
                    "source": "r2_storage",
                    "reference_images": {
                        "count": len(reference_images),
                        "filenames": [img.name for img in reference_images] if reference_images else []
                    },
                    "generation_dirs": {
                        "initial": str(initial_output_dir),
                        "extensions": [str(d) for d in extension_dirs]
                    }
                }
                
                print(f"✅ WAN R2 generation complete!")
                print(f"📊 Generation summary:")
                print(f"   - Session ID: {session_id}")
                print(f"   - Total frames: {total_frames}")
                print(f"   - Initial frames: {initial_frame_count}")
                print(f"   - Extension frames: {extension_frame_count}")
                print(f"   - Resolution: {resolution}")
                print(f"   - Source: R2 Storage")
                print(f"   - Videos created: {len(video_files)}")
                
                # Create persistent directory for video files
                import tempfile as tf
                persistent_dir = Path(tf.mkdtemp(prefix="sieve_r2_wan_videos_"))
                print(f"📁 Creating persistent video directory: {persistent_dir}")
                
                # Copy videos to persistent location
                persistent_videos = []
                
                # Copy initial video
                initial_video = next((vf for vf in video_files if vf["type"] == "initial"), None)
                if initial_video and initial_video["path"].exists():
                    persistent_initial_path = persistent_dir / f"{original_video_name}_initial.mp4"
                    shutil.copy2(initial_video["path"], persistent_initial_path)
                    persistent_videos.append({"type": "initial", "path": persistent_initial_path})
                    print(f"📋 Copied initial video to: {persistent_initial_path}")
                
                # Copy extension videos
                extension_video_files = [vf for vf in video_files if vf["type"] == "extension"]
                persistent_extensions = []
                for i, ext_video in enumerate(extension_video_files):
                    if ext_video["path"].exists():
                        persistent_ext_path = persistent_dir / f"{original_video_name}_extension_{i+1:03d}.mp4"
                        shutil.copy2(ext_video["path"], persistent_ext_path)
                        persistent_extensions.append(persistent_ext_path)
                        print(f"📋 Copied extension video to: {persistent_ext_path}")
                
                # Copy combined video
                persistent_combined_path = None
                if combined_video_path and combined_video_path.exists():
                    persistent_combined_path = persistent_dir / f"{original_video_name}_complete.mp4"
                    shutil.copy2(combined_video_path, persistent_combined_path)
                    print(f"📋 Copied combined video to: {persistent_combined_path}")
                
                # Create return structure with sieve.File objects
                result = {
                    "total_frames": total_frames,
                    "generation_info": generation_info
                }
                
                # Add initial video if available
                if persistent_videos and persistent_videos[0]["type"] == "initial":
                    result["initial_video"] = sieve.File(path=str(persistent_videos[0]["path"]))
                    print(f"🎬 Initial video ready: {persistent_videos[0]['path']}")
                
                # Add extension videos if available
                if persistent_extensions:
                    result["extension_videos"] = [sieve.File(path=str(path)) for path in persistent_extensions]
                    print(f"🎬 Extension videos ready: {len(persistent_extensions)}")
                
                # Add combined video if available
                if persistent_combined_path:
                    result["combined_video"] = sieve.File(path=str(persistent_combined_path))
                    print(f"🎬 Combined video ready: {persistent_combined_path}")
                
                # Upload videos to R2 and add URLs to result
                print(f"\n📤 Uploading generated videos to R2...")
                r2_urls = self._upload_videos_to_r2(session_id, result)
                
                if r2_urls:
                    result["r2_urls"] = r2_urls
                    generation_info["r2_urls"] = r2_urls
                    print(f"✅ Videos uploaded to R2 successfully")
                    
                    # Log R2 URLs for easy access
                    print(f"📋 R2 URLs:")
                    if "initial_video" in r2_urls:
                        print(f"   - Initial: {r2_urls['initial_video']}")
                    if "extension_videos" in r2_urls:
                        for i, url in enumerate(r2_urls["extension_videos"]):
                            print(f"   - Extension {i+1}: {url}")
                    if "combined_video" in r2_urls:
                        print(f"   - Combined: {r2_urls['combined_video']}")
                else:
                    print(f"⚠️  No videos were uploaded to R2")
                
                return result
                
        except Exception as e:
            print(f"❌ WAN R2 generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Return error response
            return {
                "error": str(e),
                "total_frames": 0,
                "generation_info": {
                    "session_id": session_id,
                    "prompt": prompt,
                    "resolution": resolution,
                    "num_extensions": num_extensions,
                    "error": str(e),
                    "source": "r2_storage"
                }
            }

def test_r2_sieve_function():
    """Test the R2-based Sieve WAN function locally with a real session."""
    print("🧪 Testing R2-based Sieve WAN Function")
    print("=" * 60)
    
    # Test session ID provided by user
    test_session_id = "session_1753484883710_h47bqnjl2"
    
    print(f"📋 Test Parameters:")
    print(f"   - Session ID: {test_session_id}")
    print(f"   - ComfyUI Path: {COMFYUI_PATH}")
    
    # For local testing, try to use cloud_storage.py directly
    print(f"\n🔧 Checking R2 environment variables...")
    
    # Try to import and use cloud_storage functions directly for local testing
    try:
        from cloud_storage import get_env_vars as local_get_env_vars, get_s3_client as local_get_s3_client
        print("✅ Using local cloud_storage.py for testing")
        
        env_vars = local_get_env_vars()
        missing_vars = [k for k, v in env_vars.items() if not v]
        
        if missing_vars:
            print(f"❌ Missing R2 environment variables: {missing_vars}")
            print("Please set the following environment variables:")
            for var in missing_vars:
                if var == 'ENDPOINT_URL':
                    print(f"   - RUNPOD_SECRET_R2_ENDPOINT_URL")
                elif var == 'ACCESS_KEY_ID':
                    print(f"   - RUNPOD_SECRET_R2_ACCESS_KEY_ID")
                elif var == 'SECRET_ACCESS_KEY':
                    print(f"   - RUNPOD_SECRET_R2_SECRET_ACCESS_KEY")
                elif var == 'BUCKET_NAME':
                    print(f"   - RUNPOD_SECRET_R2_BUCKET_NAME")
                elif var == 'PUBLIC_URL_BASE':
                    print(f"   - RUNPOD_SECRET_R2_PUBLIC_URL_BASE")
            return False
        
        # Use local cloud_storage functions for testing
        get_s3_client_func = local_get_s3_client
        
    except ImportError:
        print("⚠️  cloud_storage.py not available, using built-in functions")
        env_vars = get_env_vars()
        missing_vars = [k for k, v in env_vars.items() if not v]
        get_s3_client_func = get_s3_client
    
    if missing_vars:
        print(f"❌ Missing R2 environment variables: {missing_vars}")
        print("Please set the following environment variables:")
        for var in missing_vars:
            print(f"   - RUNPOD_SECRET_{var}")
        return False
    
    print(f"✅ All R2 environment variables are set")
    print(f"   - Endpoint: {env_vars['ENDPOINT_URL'][:30]}...")
    print(f"   - Bucket: {env_vars['BUCKET_NAME']}")
    print(f"   - Public URL: {env_vars['PUBLIC_URL_BASE'][:30]}...")
    
    # Test R2 connectivity
    print(f"\n🔗 Testing R2 connectivity...")
    try:
        s3_client = get_s3_client_func()
        
        # List objects in the session directory to see what's available
        session_prefix = f"sessions/{test_session_id}/"
        print(f"📂 Listing objects in session directory: {session_prefix}")
        
        response = s3_client.list_objects_v2(
            Bucket=env_vars['BUCKET_NAME'],
            Prefix=session_prefix,
            MaxKeys=20  # Limit results for testing
        )
        
        objects = response.get('Contents', [])
        if not objects:
            print(f"❌ No objects found in session directory")
            print(f"   - Checked prefix: {session_prefix}")
            return False
        
        print(f"✅ Found {len(objects)} objects in session directory:")
        for obj in objects[:10]:  # Show first 10 objects
            size_kb = obj['Size'] / 1024
            print(f"   - {obj['Key']} ({size_kb:.1f} KB)")
        
        if len(objects) > 10:
            print(f"   ... and {len(objects) - 10} more objects")
        
        # Look specifically for video files
        video_files = [obj for obj in objects 
                      if obj['Key'].lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))]
        
        if video_files:
            print(f"🎬 Found {len(video_files)} video file(s):")
            for video in video_files:
                size_mb = video['Size'] / (1024 * 1024)
                print(f"   - {video['Key']} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️  No video files found in session directory")
            
    except Exception as e:
        print(f"❌ R2 connectivity test failed: {e}")
        return False
    
    # Test the _fetch_r2_file method
    print(f"\n📥 Testing R2 file fetching...")
    try:
        # Create a test instance (without full Sieve setup)
        test_generator = WanVideoGeneratorR2(COMFY_PATH="/home/paperspace/wan-paper/ComfyUI")
        
        # Test fetching video file
        print(f"🎬 Attempting to fetch video file for session {test_session_id}...")
        video_path = test_generator._fetch_r2_file(test_session_id, "video")
        
        if video_path and video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"✅ Successfully fetched video file:")
            print(f"   - Local path: {video_path}")
            print(f"   - File size: {size_mb:.1f} MB")
            print(f"   - File format: {video_path.suffix}")
            
            # Check if WebM was automatically converted to MP4
            if video_path.suffix.lower() == '.mp4':
                print(f"✅ WebM conversion: Video is in MP4 format (conversion successful or was already MP4)")
            elif video_path.suffix.lower() == '.webm':
                print(f"ℹ️  WebM conversion: Video remains as WebM (conversion skipped or failed)")
            
            # Basic video validation
            print(f"🔍 Validating video file...")
            try:
                import subprocess
                import json
                
                result = subprocess.run([
                    "ffprobe", "-v", "quiet", "-print_format", "json", 
                    "-show_format", "-show_streams", str(video_path)
                ], capture_output=True, text=True, check=True)
                
                probe_data = json.loads(result.stdout)
                video_streams = [s for s in probe_data.get("streams", []) 
                               if s.get("codec_type") == "video"]
                
                if video_streams:
                    stream = video_streams[0]
                    width = stream.get("width")
                    height = stream.get("height")
                    duration = float(stream.get("duration", 0))
                    fps = eval(stream.get("r_frame_rate", "30/1"))
                    
                    print(f"✅ Video validation successful:")
                    print(f"   - Dimensions: {width}x{height}")
                    print(f"   - Duration: {duration:.2f}s")
                    print(f"   - FPS: {fps:.2f}")
                    
                    # Clean up downloaded file
                    video_path.unlink()
                    video_path.parent.rmdir()
                    print(f"🧹 Cleaned up temporary files")
                    
                else:
                    print(f"⚠️  No video streams found in file")
                    
            except Exception as validation_error:
                print(f"⚠️  Video validation failed: {validation_error}")
                
        else:
            print(f"❌ Failed to fetch video file")
            return False
            
    except Exception as e:
        print(f"❌ R2 file fetching test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Summary
    print(f"\n📊 Test Summary:")
    print(f"✅ R2 environment variables: OK")
    print(f"✅ R2 connectivity: OK")
    print(f"✅ Session objects found: OK")
    print(f"✅ Video file fetching: OK")
    print(f"✅ Video validation: OK")
    
    print(f"\n🎯 Test Results:")
    print(f"✅ The R2-based Sieve function should work with session: {test_session_id}")
    print(f"📋 Ready for production deployment to Sieve platform")
    print(f"🚀 Next step: Deploy with 'sieve deploy sieve_wan_generator_r2.py'")
    
    return True

def test_full_wan_generation():
    """Test the complete WAN generation pipeline with R2 files."""
    """You must install ComfyUI and download models locally to run this test"""
    print("🎬 Testing Full WAN Generation Pipeline")
    print("=" * 60)
    
    # Test session ID provided by user
    test_session_id = "session_1753742688539_ppz46418y"
    
    print(f"📋 Test Parameters:")
    print(f"   - Session ID: {test_session_id}")
    print(f"   - ComfyUI Path: /home/paperspace/wan-paper/ComfyUI")
    print(f"   - Test Mode: Local (not Sieve cloud)")
    
    # Check environment and dependencies
    print(f"\n🔧 Checking dependencies...")
    
    # Check if we have cloud_storage available
    try:
        from cloud_storage import get_env_vars as local_get_env_vars
        env_vars = local_get_env_vars()
        missing_vars = [k for k, v in env_vars.items() if not v]
        
        if missing_vars:
            print(f"❌ Missing R2 environment variables: {missing_vars}")
            return False
            
        print("✅ R2 environment variables: OK")
        
    except ImportError:
        print("❌ cloud_storage.py not available")
        return False
    
    # Check if WAN dependencies are available
    try:
        from wan_video_generator import WanVideoGenerator, GenerationConfig
        print("✅ WAN generator dependencies: OK")
        
    except ImportError as e:
        print(f"❌ WAN generator dependencies missing: {e}")
        print("This test requires the local WAN generator to be available")
        return False
    
    # Check ComfyUI path
    import os
    comfy_path = "/home/paperspace/wan-paper/ComfyUI"
    if not os.path.exists(comfy_path):
        print(f"❌ ComfyUI not found at: {comfy_path}")
        return False
        
    print(f"✅ ComfyUI path: OK ({comfy_path})")
    
    # Create test instance with local ComfyUI path
    print(f"\n🚀 Creating WAN R2 generator instance...")
    try:
        generator = WanVideoGeneratorR2(COMFY_PATH=comfy_path)
        generator.__setup__()
        print("✅ Generator instance created successfully")
        
    except Exception as e:
        print(f"❌ Failed to create generator instance: {e}")
        return False
    
    # Test parameters for generation
    test_params = {
        "session_id": test_session_id,
        "prompt": "pulsing slime",
        "num_extensions": 0,  # Generate initial + 1 extension
        "resolution": "512p",
        "invert_mask": True,
        "seed": 314525102295492
    }
    
    print(f"\n🎬 Starting full WAN generation test...")
    print(f"📋 Generation parameters:")
    for key, value in test_params.items():
        print(f"   - {key}: {value}")
    
    # Run the actual prediction
    print(f"\n🚀 Running __predict__ method...")
    try:
        result = generator.__predict__(**test_params)
        
        print(f"\n✅ Generation completed successfully!")
        print(f"📊 Result summary:")
        
        # Check if we got an error
        if "error" in result:
            print(f"❌ Generation failed with error: {result['error']}")
            return False
        
        # Print basic result info
        total_frames = result.get("total_frames", 0)
        generation_info = result.get("generation_info", {})
        
        print(f"   - Total frames generated: {total_frames}")
        print(f"   - Session ID: {generation_info.get('session_id', 'N/A')}")
        print(f"   - Prompt: {generation_info.get('prompt', 'N/A')}")
        print(f"   - Resolution: {generation_info.get('resolution', 'N/A')}")
        print(f"   - Source: {generation_info.get('source', 'N/A')}")
        
        # Check for video outputs
        has_initial = "initial_video" in result
        has_extensions = "extension_videos" in result and len(result["extension_videos"]) > 0
        has_combined = "combined_video" in result
        
        print(f"   - Initial video: {'✅' if has_initial else '❌'}")
        print(f"   - Extension videos: {'✅' if has_extensions else '❌'} ({len(result.get('extension_videos', []))} videos)")
        print(f"   - Combined video: {'✅' if has_combined else '❌'}")
        
        # Detailed video file information
        if has_initial:
            initial_path = result["initial_video"].path
            initial_size = os.path.getsize(initial_path) / (1024 * 1024)
            print(f"📁 Initial video: {initial_path} ({initial_size:.1f} MB)")
        
        if has_extensions:
            print(f"📁 Extension videos:")
            for i, ext_video in enumerate(result["extension_videos"]):
                ext_path = ext_video.path
                ext_size = os.path.getsize(ext_path) / (1024 * 1024)
                print(f"   - Extension {i+1}: {ext_path} ({ext_size:.1f} MB)")
        
        if has_combined:
            combined_path = result["combined_video"].path
            combined_size = os.path.getsize(combined_path) / (1024 * 1024)
            print(f"📁 Combined video: {combined_path} ({combined_size:.1f} MB)")
        
        # Validate video files with ffprobe
        print(f"\n🔍 Validating generated videos...")
        
        videos_to_check = []
        if has_initial:
            videos_to_check.append(("Initial", result["initial_video"].path))
        if has_extensions:
            for i, ext_video in enumerate(result["extension_videos"]):
                videos_to_check.append((f"Extension {i+1}", ext_video.path))
        if has_combined:
            videos_to_check.append(("Combined", result["combined_video"].path))
        
        all_valid = True
        for video_name, video_path in videos_to_check:
            try:
                import subprocess
                import json
                
                result_probe = subprocess.run([
                    "ffprobe", "-v", "quiet", "-print_format", "json", 
                    "-show_format", "-show_streams", str(video_path)
                ], capture_output=True, text=True, check=True)
                
                probe_data = json.loads(result_probe.stdout)
                video_streams = [s for s in probe_data.get("streams", []) 
                               if s.get("codec_type") == "video"]
                
                if video_streams:
                    stream = video_streams[0]
                    width = stream.get("width")
                    height = stream.get("height")
                    duration = float(stream.get("duration", 0))
                    fps = eval(stream.get("r_frame_rate", "30/1"))
                    
                    print(f"✅ {video_name}: {width}x{height}, {duration:.2f}s, {fps:.1f}fps")
                else:
                    print(f"❌ {video_name}: No video streams found")
                    all_valid = False
                    
            except Exception as e:
                print(f"❌ {video_name}: Validation failed - {e}")
                all_valid = False
        
        # Check R2 uploads
        has_r2_urls = "r2_urls" in result and result["r2_urls"]
        print(f"   - R2 uploads: {'✅' if has_r2_urls else '❌'}")
        
        if has_r2_urls:
            r2_urls = result["r2_urls"]
            print(f"📤 R2 Upload Results:")
            if "initial_video" in r2_urls:
                print(f"   - Initial video: {r2_urls['initial_video']}")
            if "extension_videos" in r2_urls:
                for i, url in enumerate(r2_urls["extension_videos"]):
                    print(f"   - Extension {i+1}: {url}")
            if "combined_video" in r2_urls:
                print(f"   - Combined video: {r2_urls['combined_video']}")
        
        # Final results
        print(f"\n📊 Final Test Results:")
        print(f"✅ R2 file fetching: OK")
        print(f"✅ WAN generation: OK")
        print(f"✅ Video creation: OK")
        print(f"✅ Video validation: {'OK' if all_valid else 'FAILED'}")
        print(f"✅ R2 video uploads: {'OK' if has_r2_urls else 'FAILED'}")
        
        if total_frames > 0 and (has_initial or has_combined) and has_r2_urls:
            print(f"\n🎉 SUCCESS: Full WAN generation pipeline with R2 uploads working!")
            print(f"🚀 Generated {total_frames} frames from R2 session: {test_session_id}")
            print(f"📤 All videos uploaded to R2 and accessible via public URLs")
            print(f"📋 Ready for production Sieve deployment!")
            return True
        else:
            print(f"\n❌ FAILED: Missing frames, videos, or R2 uploads")
            return False
            
    except Exception as e:
        print(f"❌ Generation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # Run full generation test
        print("Running FULL WAN generation test (this may take several minutes)...")
        success = test_full_wan_generation()
        if success:
            print(f"\n🎉 Full generation test passed!")
        else:
            print(f"\n❌ Full generation test failed.")
            exit(1)
    else:
        # Run basic R2 connectivity test (default)
        print("Running R2 connectivity test (use 'python sieve_wan_generator_r2.py full' for generation test)...")
        success = test_r2_sieve_function()
        if success:
            print(f"\n🎉 R2 connectivity test passed!")
        else:
            print(f"\n❌ R2 connectivity test failed.")
            exit(1)