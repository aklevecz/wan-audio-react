from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
import tempfile
from pathlib import Path
import os
import io
import sys
import torch

# Add "./backend" to path so we can import wan_integration
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add this directory to the sys.path
if current_dir not in sys.path:
    print("Adding", current_dir, "to sys.path")
    sys.path.append(current_dir)
# Import WAN integration and session manager
try:
    from session_manager import session_manager
    from filename_utils import sanitize_filename, create_session_directory_name
    from cloud_storage import upload_file, get_public_url
    WAN_AVAILABLE = True
    if torch.cuda.is_available():
        print("CUDA IS AVAILABLE")
    print("[OK] WAN integration available")
except ImportError as e:
    print(f"[WARNING] WAN integration not available: {e}")
    WAN_AVAILABLE = False

# Sieve function name constant
SIEVE_R2_FUNCTION_NAME = "kaiber/wan-video-generator-r2"

app = FastAPI(title="Audio Mask API", description="Endpoints for audio analysis and rendering.")

# Allow local dev front-end on http://localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5176", "https://audio-reactive-fe.raptorz.workers.dev", "https://audio-reactive-fe.raptorz.workers.dev/", "https://reactor.raptorz.workers.dev", "https://reactor.raptorz.workers.dev/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Entry point for Generation
# Video may be preset instead of uploaded
@app.post("/session/{session_id}/upload_video_and_image")
async def upload_video_and_image(
    session_id: str,
    video_file: UploadFile = File(...),
    image_file: Optional[UploadFile] = File(None),
    prompt: str = Form(""),
    num_extensions: int = Form(1),
    invert_mask: bool = Form(False),
    resolution: str = Form("720p"),
    use_distorted_video: bool = Form(False),
    seed: int = 42069
):
    """Upload both a video file and image file with processing options."""
    print("calling upload_video_and_image")
    try:
        # Session management -- should be adapted to Kaiber session usage
        session = session_manager.get_session(session_id)
        if not session:
            # Naively creating a session for testing
            session_manager.create_session_raw(session_id)
        #     raise HTTPException(status_code=404, detail="Session not found")

        # Video file is the masked audio reactive video
        # Note: Probably providing a preset video file instead of uploading from the frontend
        if not video_file.filename:
            raise HTTPException(status_code=400, detail="No video filename provided")
        video_filename = video_file.filename.lower()
        if not video_filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
            raise HTTPException(status_code=400, detail="Unsupported video type. Use MP4, AVI, MOV, MKV, or WebM")
        
        # Initialize image variables
        image_status = "not_provided"
        relative_image_path = None
        
        # Image file is not required
        if not image_file or not image_file.filename:
            # raise HTTPException(status_code=400, detail="No image filename provided")
            image_filename = None
        else:
            image_filename = image_file.filename.lower()
            if not image_filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                raise HTTPException(status_code=400, detail="Unsupported image type. Use PNG, JPG, JPEG, BMP, or WebP")
        
        # Handle video file upload to R2
        video_data = await video_file.read()
        # Use predictable filename - preserve original extension (typically .webm)
        original_extension = Path(video_file.filename).suffix.lower()
        video_safe_name = f"mask_video{original_extension}"
        
        # Create temporary file for R2 upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=original_extension) as temp_video:
            temp_video.write(video_data)
            temp_video_path = Path(temp_video.name)
        
        
        try:
            # Might not be necessary if we are using preset videos
            # Upload to R2
            r2_video_key = f"sessions/{session_id}/uploaded_videos/{video_safe_name}"
            video_upload_success = upload_file(temp_video_path, r2_video_key)
            
            if not video_upload_success:
                raise HTTPException(status_code=500, detail=f"Failed to upload video '{video_file.filename}' to cloud storage")
            
            # Get R2 public URL
            video_r2_url = get_public_url(r2_video_key)
            print(f"[SUCCESS] Video uploaded to R2: {video_r2_url}")
            
        except ValueError as e:
            # Handle R2 configuration issues
            print(f"[ERROR] R2 configuration error during video upload: {e}")
            raise HTTPException(status_code=503, detail="Cloud storage configuration error")
        except Exception as e:
            # Handle other R2 upload errors
            print(f"[ERROR] Failed to upload video to R2: {e}")
            raise HTTPException(status_code=500, detail=f"Cloud storage upload failed: {str(e)}")
        finally:
            # Clean up temporary file
            temp_video_path.unlink(missing_ok=True)
        
        # Handle image file upload to R2 (only if image provided)
        if image_filename is not None:
            image_data = await image_file.read()
            
            try:
                from PIL import Image
                import io
                
                # Open image from bytes and convert to RGB (removes alpha channel if present)
                pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Convert to JPG bytes
                jpg_buffer = io.BytesIO()
                pil_image.save(jpg_buffer, format='JPEG', quality=95, optimize=True)
                jpg_data = jpg_buffer.getvalue()
                
            except Exception as e:
                print(f"[ERROR] Failed to convert image to JPG: {e}")
                raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")
            
            # Use predictable filename - always overwrite with reference_image.jpg
            image_safe_name = "reference_image.jpg"
            
            # Create temporary file for R2 upload with converted JPG data
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_image:
                temp_image.write(jpg_data)
                temp_image_path = Path(temp_image.name)
            
            try:
                # Upload to R2
                r2_image_key = f"sessions/{session_id}/reference_images/{image_safe_name}"
                image_upload_success = upload_file(temp_image_path, r2_image_key)
                
                if not image_upload_success:
                    raise HTTPException(status_code=500, detail=f"Failed to upload image '{image_file.filename}' to cloud storage")
                
                # Get R2 public URL
                image_r2_url = get_public_url(r2_image_key)
                print(f"[SUCCESS] Image uploaded to R2: {image_r2_url}")
                
                # Add to session metadata (store R2 URL instead of local path)
                session_manager.add_reference_image(session_id, image_r2_url)
                image_status = "uploaded"
                relative_image_path = image_r2_url
                
            except ValueError as e:
                # Handle R2 configuration issues
                print(f"[ERROR] R2 configuration error during image upload: {e}")
                raise HTTPException(status_code=503, detail="Cloud storage configuration error")
            except Exception as e:
                # Handle other R2 upload errors
                print(f"[ERROR] Failed to upload image to R2: {e}")
                raise HTTPException(status_code=500, detail=f"Cloud storage upload failed: {str(e)}")
            finally:
                # Clean up temporary file
                temp_image_path.unlink(missing_ok=True)
        
        # Store processing options in session metadata
        processing_options = {
            "prompt": prompt,
            "num_extensions": num_extensions,
            "invert_mask": invert_mask,
            "resolution": resolution,
            "use_distorted_video": use_distorted_video
        }

        # Import Sieve at the function level to handle cases where it's not available
        try:
            import sieve
        except ImportError:
            raise HTTPException(status_code=503, detail="Sieve SDK not available")
        
        try:
            print(f"[INFO] Processing WAN R2 for session {session_id}")
            print(f"[INFO] Sieve will fetch R2 files directly using session_id")
            
            # Call Sieve function directly with session_id
            # Sieve function will handle R2 file fetching internally
            print(f"[INFO] Calling Sieve R2 function: {SIEVE_R2_FUNCTION_NAME}")
            
            wan_generator_r2 = sieve.function.get(SIEVE_R2_FUNCTION_NAME)
            
            # Debug parameter values
            print(f"[DEBUG] Sieve R2 function parameters:")
            print(f"   - session_id: {session_id}")
            print(f"   - prompt: {prompt} (type: {type(prompt)})")
            print(f"   - num_extensions: {num_extensions} (type: {type(num_extensions)})")
            print(f"   - resolution: {resolution} (type: {type(resolution)})")
            print(f"   - invert_mask: {invert_mask} (type: {type(invert_mask)})")

            # Call Sieve function asynchronously - returns job ID immediately
            sieve_job = wan_generator_r2.push(
                session_id=str(session_id),
                prompt=str(prompt) if prompt is not None else "bellowing flames",
                num_extensions=int(num_extensions),
                resolution=str(resolution) if resolution is not None else "720p",
                invert_mask=bool(invert_mask),
                seed=seed if seed is not None else 42069 # Use fixed seed for now
            )
            sieve_job_id = sieve_job.job['id']
            print(f"[SUCCESS] Sieve job started asynchronously: {sieve_job_id}")


        except Exception as e:
            print(f"[ERROR] WAN R2 processing failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))
        
        return {
            "session_id": session_id,
            "sieve_job_id": sieve_job_id,
            "video_path": video_r2_url,
            "video_filename": video_file.filename,
            "video_upload_status": "uploaded_to_r2",
            "image_path": relative_image_path,
            "image_filename": image_file.filename if image_file else None,
            "image_status": image_status,
            "processing_options": processing_options,
            "message": "Video and image uploaded successfully to cloud storage. Sieve processing started."
        }
        
    except Exception as e:
        print(f"Error uploading video and image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/sieve/job/{job_id}")
async def get_sieve_job_status(job_id: str):
    """Get Sieve job status and results using Sieve HTTP API."""
    try:
        import os
        import requests
        
        # Get Sieve API key from environment
        api_key = os.environ.get("SIEVE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="SIEVE_API_KEY not configured")
        
        # Call Sieve API directly
        url = f"https://mango.sievedata.com/v2/jobs/{job_id}"
        headers = {"X-API-Key": api_key}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Job not found")
        elif response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=f"Sieve API error: {response.text}")
        
        job_data = response.json()
        
        return {
            "job_id": job_id,
            "status": job_data.get("status"),
            "message": f"Sieve job status: {job_data.get('status')}",
            "job_data": job_data
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to Sieve API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get Sieve job status: {str(e)}")