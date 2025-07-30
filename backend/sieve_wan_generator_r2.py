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
from datetime import datetime
import uuid

COMFYUI_PATH = "/src/ComfyUI"

# URL parsing for downloading files
from urllib.parse import urlparse

def generate_unique_id():
    """
    Generates a unique identifier combining UUID and timestamp.
    Returns a string in format: timestamp-uuid
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_uuid = str(uuid.uuid4())
    return f"{timestamp}_{unique_uuid}"

@sieve.Model(
    name="wan-video-generator-r2", 
    python_version="3.11",
    gpu=sieve.gpu.A100(),
    system_packages=["ffmpeg", "git", "wget", "curl"],
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
        
        # URL downloading
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
    
    def _fetch_video_from_url(self, video_url: str) -> Optional[Path]:
        """
        Fetch a video file from a URL.
        
        Args:
            video_url: URL to the video file
            
        Returns:
            Path to downloaded file or None if not found
        """
        try:
            print(f"[INFO] Downloading video from URL: {video_url}")
            
            # Create temporary directory
            temp_dir = Path(tempfile.gettempdir()) / f"url_downloads_video"
            temp_dir.mkdir(exist_ok=True)
            
            # Extract filename from URL or generate one
            from urllib.parse import urlparse
            parsed_url = urlparse(video_url)
            filename = Path(parsed_url.path).name
            if not filename or not any(filename.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                filename = "downloaded_video.mp4"
            
            temp_file = temp_dir / filename
            
            # Download file using requests
            response = requests.get(video_url, stream=True)
            response.raise_for_status()
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[INFO] Successfully downloaded video to: {temp_file} ({temp_file.stat().st_size} bytes)")
            
            # Convert WebM to MP4 if necessary
            if temp_file.suffix.lower() == '.webm':
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
            print(f"[ERROR] Failed to fetch video from URL: {e}")
            return None
    
    def _fetch_image_from_url(self, img_url: str) -> List[Path]:
        """
        Fetch a reference image from a URL.
        
        Args:
            img_url: URL to the image file
            
        Returns:
            List containing single reference image Path, or empty list if none found
        """
        try:
            print(f"[INFO] Attempting to fetch reference image from URL: {img_url}")
            
            # Create temporary directory
            temp_dir = Path(tempfile.gettempdir()) / f"url_downloads_image"
            temp_dir.mkdir(exist_ok=True)
            
            # Extract filename from URL or generate one
            from urllib.parse import urlparse
            parsed_url = urlparse(img_url)
            filename = Path(parsed_url.path).name
            if not filename or not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
                filename = "reference_image.jpg"
            
            reference_path = temp_dir / filename
            
            # Download file using requests
            response = requests.get(img_url, stream=True)
            response.raise_for_status()
            
            with open(reference_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"[INFO] Successfully downloaded image to: {reference_path} ({reference_path.stat().st_size} bytes)")
            
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
                    print(f"[INFO] Successfully found and validated reference image from URL")
                    return [reference_path]
                except Exception as e:
                    print(f"[WARNING] Invalid reference image file from URL: {e}")
                    if reference_path.exists():
                        reference_path.unlink()
                    return []
            else:
                print(f"[INFO] No reference image downloaded from URL")
                return []
                
        except Exception as e:
            print(f"[ERROR] Failed to get reference image from URL: {e}")
            return []
    
    
    def __predict__(
        self,
        session_id: str,
        prompt: str = "bellowing flames",
        num_extensions: int = 1,
        resolution: str = "512p",
        invert_mask: bool = False,
        seed: int = 314525102295492,
        img_url: str = None,
        video_url: str = None
    ) -> Dict[str, Any]:
        """
        Generate WAN video using files from URLs.
        
        Args:
            session_id: Session ID for tracking (legacy parameter)
            prompt: Text prompt for generation
            num_extensions: Number of extension generations (0 for initial only)
            resolution: Resolution preset ("512p" or "720p")
            invert_mask: Whether to invert the mask
            seed: Random seed for generation
            img_url: URL to the reference image
            video_url: URL to the input video
            
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
            print(f"🚀 Starting WAN generation with URLs for session: {session_id}")
            print(f"   - Video URL: {video_url}")
            print(f"   - Image URL: {img_url}")
            print("✅ Step 1: Function entry successful")
            
            # Setup temporary working directory
            print("🔧 Step 2: Creating temporary directory...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                print(f"✅ Step 2: Temp directory created: {temp_path}")
                
                # We now fetch a video file from a URL
                print("🔧 Step 3: Fetching video from URL...")
                video_file_path = self._fetch_video_from_url(video_url)
                if not video_file_path or not video_file_path.exists():
                    raise FileNotFoundError(f"Could not fetch video file from URL: {video_url}")
                
                print(f"✅ Step 3: Video fetched from URL: {video_file_path}")
                original_video_name = video_file_path.stem
                
                # Copy to local temp directory for processing
                print("🔧 Step 4: Preparing local video file...")
                local_video_path = temp_path / f"{original_video_name}.mp4"
                shutil.copy2(video_file_path, local_video_path)
                print(f"✅ Step 4: Video prepared: {local_video_path}")
                
                # Fetch reference images from URL
                print("🔧 Step 5: Fetching reference images from URL...")
                reference_images = self._fetch_image_from_url(img_url)
                print(f"✅ Step 5: Found {len(reference_images)} reference images")
                
                # Debug reference images
                if reference_images:
                    for i, ref_path in enumerate(reference_images):
                        print(f"[DEBUG] Reference image {i+1}: {ref_path} (exists: {ref_path.exists()})")
                else:
                    print(f"[DEBUG] No reference images found from URL")
                
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
                        
                        print(f"📊 Video validation:")
                        print(f"   - Dimensions: {width}x{height}")
                        print(f"   - Duration: {duration:.2f}s")
                        print(f"   - FPS: {fps:.2f}")
                        print(f"   - Codec: {codec_name}")
                        
                        if codec_name in ['png', 'jpeg', 'jpg', 'webp', 'bmp'] or duration == 0.0:
                            raise ValueError(f"ERROR: Downloaded file is an image ({codec_name}), not a video. "
                                           f"Duration: {duration}s. Please provide a video URL.")
                    else:
                        raise ValueError("No video streams found in downloaded file.")
                        
                except Exception as e:
                    print(f"⚠️  Video validation failed: {e}")
                    raise e
                
                print(f"🚀 Starting WAN generation with URL files:")
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
                    "execution_environment": "sieve_url_cloud",
                    "source": "url_download",
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
                print(f"   - Source: URL Download")
                print(f"   - Videos created: {len(video_files)}")
                
                # Create persistent directory for video files
                import tempfile as tf
                persistent_dir = Path(tf.mkdtemp(prefix="sieve_url_wan_videos_"))
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

                final_output_video = result["combined_video"]
                # Begin upload of combined file -- probably don't need initial and extension ultimately
                upload_name = f"{generate_unique_id()}_wan_audio_reactive.mp4"
                r2_upload_path = f"init-images/{upload_name}"

                print(f"Preparing for postgen upload.\n Media: {final_output_video.path}\n Upload path: {r2_upload_path}")
                postgen = sieve.function.get("kaiber/kaiber-postgen")
                future = postgen.push(file=final_output_video, s3_path=r2_upload_path)
                postgen_jobid = future.job["id"]

                yield {
                    "postgen_jobid": postgen_jobid,
                }
                yield final_output_video
                
        except Exception as e:
            print(f"❌ WAN URL generation failed: {str(e)}")
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
                    "source": "url_download"
                }
            }
