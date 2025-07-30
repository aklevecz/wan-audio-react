# WAN Video Generation Pipeline

This README maps the complete flow from the `/upload_video_and_image` endpoint in `main.py` to the WAN video generation process.

## Flow Overview

```
FastAPI Endpoint → Session Management → R2 Upload → Sieve Function → WAN Generation → R2 Storage
```

## 1. FastAPI Endpoint (`main.py:47-251`)

**Endpoint**: `POST /session/{session_id}/upload_video_and_image`

**Input Parameters**:
- `session_id`: Unique session identifier
- `video_file`: Uploaded video file (MP4, AVI, MOV, MKV, WebM)
- `image_file`: Optional reference image (PNG, JPG, JPEG, BMP, WebP)
- `prompt`: Text prompt for generation (default: "")
- `num_extensions`: Number of extension generations (default: 1)
- `invert_mask`: Whether to invert the mask (default: False)
- `resolution`: Resolution preset (default: "720p")
- `use_distorted_video`: Whether to use distorted video (default: False)
- `seed`: Random seed (default: 42069)

**Process Flow**:

### 1.1 Session Management (`main.py:62-66`)
- Retrieves or creates session using `session_manager.py`
- Session tracks metadata, files, and generation history

### 1.2 Video Processing (`main.py:69-125`)
- Validates video file format and filename
- Uploads video to R2 storage at `sessions/{session_id}/uploaded_videos/mask_video{ext}`
- Generates public R2 URL for video access

### 1.3 Image Processing (`main.py:127-182`)
- Optional reference image processing
- Converts any format to JPG (RGB, quality=95)
- Uploads to R2 at `sessions/{session_id}/reference_images/reference_image.jpg`
- Updates session metadata with image URL

### 1.4 Sieve Job Invocation (`main.py:194-234`)
- Calls Sieve function `kaiber/wan-video-generator-r2`
- Passes session_id and generation parameters
- Returns job ID for async processing

## 2. Cloud Storage (`cloud_storage.py`)

**Purpose**: Handles R2 (Cloudflare) storage operations

**Key Functions**:
- `upload_file()`: Uploads files to R2 bucket
- `get_public_url()`: Generates public URLs for R2 objects
- `download_r2_file()`: Downloads files from R2 (used by Sieve)

**Environment Variables**:
- `RUNPOD_SECRET_R2_ENDPOINT_URL`
- `RUNPOD_SECRET_R2_ACCESS_KEY_ID`
- `RUNPOD_SECRET_R2_SECRET_ACCESS_KEY`
- `RUNPOD_SECRET_R2_BUCKET_NAME`
- `RUNPOD_SECRET_R2_PUBLIC_URL_BASE`

## 3. Session Management (`session_manager.py`)

**Purpose**: Tracks generation sessions and metadata

**Key Components**:
- `SessionMetadata`: Dataclass storing session information
- `SessionManager`: Manages session lifecycle
- Session tracking includes files, settings, generations, and reference images

**Storage Structure**:
```
sessions/{session_id}/
├── uploaded_videos/mask_video.{ext}
├── reference_images/reference_image.jpg
└── wan_generations/{timestamp}/
    ├── initial_video.mp4
    ├── extension_*.mp4
    └── combined_video.mp4
```

## 4. Sieve Function (`sieve_wan_generator_r2.py`)

**Purpose**: Cloud-based WAN video generation using R2 storage

**Class**: `WanVideoGeneratorR2`

### 4.1 Setup (`__setup__`)
- Initializes on A100 GPU with pre-downloaded WAN models
- Models include: WAN 2.1 T2V, VAE, text encoders, and LoRA adapters

### 4.2 File Fetching (`_fetch_r2_file`)
- Downloads video and reference image from R2 using session_id
- Converts WebM to MP4 if needed (for better compatibility)
- Validates downloaded files

### 4.3 WAN Generation (`__predict__`)
**Parameters**:
- `session_id`: Session identifier for R2 file access
- `prompt`: Text prompt for generation
- `num_extensions`: Number of extension generations
- `resolution`: Resolution preset ("512p" or "720p")
- `invert_mask`: Whether to invert the mask
- `seed`: Random seed for generation

**Process**:
1. Fetch video and reference image from R2
2. Initialize WAN generator with ComfyUI models
3. Generate initial video (81 frames)
4. Generate extensions if requested (66 frames each, 15 frame overlap)
5. Create combined video with all frames
6. Extract and sync audio from original video
7. Upload all generated videos back to R2
8. Return Sieve File objects and R2 URLs

### 4.4 Video Output Types
- **Initial Video**: First 81 frames of generation
- **Extension Videos**: Additional 66-frame segments (minus 15 overlap frames)
- **Combined Video**: All frames concatenated with original audio

## 5. WAN Video Generator (`wan_video_generator.py`)

**Purpose**: Core WAN video generation using ComfyUI as a library

**Key Features**:
- Uses ComfyUI models without server mode
- Handles initial generation and extensions
- Memory management and GPU optimization
- Reference image integration
- Audio synchronization

**Generation Process**:
1. Load WAN models (diffusion, VAE, text encoder)
2. Process input video and reference images
3. Generate frames using WAN diffusion model
4. Create MP4 videos from generated frames
5. Sync audio with generated video

## 6. Job Status Endpoint (`main.py:253-288`)

**Endpoint**: `GET /sieve/job/{job_id}`

**Purpose**: Check Sieve job status and retrieve results

**Returns**:
- Job status (running, completed, failed)
- Generated video URLs from R2 storage
- Generation metadata

## Key File Locations

- **FastAPI Server**: `main.py`
- **Session Management**: `session_manager.py`
- **Cloud Storage**: `cloud_storage.py`
- **Sieve Function**: `sieve_wan_generator_r2.py`
- **WAN Generator**: `wan_video_generator.py`

## Generation Output Structure

```
R2 Bucket:
sessions/{session_id}/
├── uploaded_videos/
│   └── mask_video.{ext}          # Original uploaded video
├── reference_images/
│   └── reference_image.jpg       # Reference image (if provided)
└── wan_generations/{timestamp}/
    ├── initial_video.mp4         # First 81 frames
    ├── extension_001.mp4         # Extension 1 (66 frames)
    ├── extension_002.mp4         # Extension 2 (66 frames)
    └── combined_video.mp4        # All frames with original audio
```

## Dependencies

- **FastAPI**: Web server and API endpoints
- **Sieve**: Cloud compute platform for GPU processing
- **ComfyUI**: WAN model integration and processing
- **R2 (Cloudflare)**: Object storage for videos and images
- **FFmpeg**: Video processing and audio extraction
- **PyTorch**: Deep learning framework for WAN models