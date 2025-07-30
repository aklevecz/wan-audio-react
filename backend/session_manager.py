import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from filename_utils import create_session_directory_name, validate_safe_path


@dataclass
class SessionMetadata:
    session_id: str
    project_name: str
    timestamp: str
    audio_file: Optional[str] = None
    video_settings: Optional[Dict] = None
    wan_settings: Optional[Dict] = None
    files: Optional[Dict] = None
    generations: Optional[Dict] = None  # Track WAN generations with prompts
    reference_images: Optional[List[str]] = None  # Track reference image paths
    status: str = "active"


class SessionManager:
    def __init__(self, base_output_dir: str = "generations"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.sessions: Dict[str, SessionMetadata] = {}
        self._load_existing_sessions()
    
    def create_session_raw(self, session_id: str):
        """Create a session with only the session ID, used for raw uploads."""
        if session_id in self.sessions:
            raise ValueError(f"Session {session_id} already exists")
        
        timestamp = datetime.now().isoformat()
        project_name = session_id
        metadata = SessionMetadata(
            session_id=session_id,
            project_name=project_name,
            timestamp=timestamp,
            files={}
        )
        
        self.sessions[session_id] = metadata
        # self._save_session_metadata(session_id)
        
        return session_id

    def create_session(self, project_name: str, audio_file: Optional[str] = None) -> str:
        """Create a new generation session."""
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Create session directory structure
        session_dir = self.get_session_dir(session_id, project_name)
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (session_dir / "wan_generations").mkdir(exist_ok=True)
        (session_dir / "reference_images").mkdir(exist_ok=True)
        
        # Create session metadata
        metadata = SessionMetadata(
            session_id=session_id,
            project_name=project_name,
            timestamp=timestamp,
            audio_file=audio_file,
            files={}
        )
        
        self.sessions[session_id] = metadata
        self._save_session_metadata(session_id)
        
        return session_id
    
    def get_session_dir(self, session_id: str, project_name: Optional[str] = None) -> Path:
        """Get the directory path for a session using sanitized naming."""
        if project_name is None and session_id in self.sessions:
            project_name = self.sessions[session_id].project_name
        
        if project_name is None:
            raise ValueError(f"Project name not found for session {session_id}")
        
        # Get timestamp for the session
        if session_id in self.sessions:
            timestamp = self.sessions[session_id].timestamp
        else:
            timestamp = datetime.now().isoformat()
        
        # Use sanitized directory naming
        dirname = create_session_directory_name(project_name, timestamp, session_id)
        
        # Security check
        if not validate_safe_path(dirname):
            raise ValueError(f"Invalid session directory name: {dirname}")
        
        return self.base_output_dir / dirname
    
    def update_session_file(self, session_id: str, file_type: str, file_path: str):
        """Update session with a new file."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if self.sessions[session_id].files is None:
            self.sessions[session_id].files = {}
        
        self.sessions[session_id].files[file_type] = file_path
        self._save_session_metadata(session_id)
    
    def update_session_settings(self, session_id: str, video_settings: Optional[Dict] = None, 
                               wan_settings: Optional[Dict] = None):
        """Update session with rendering settings."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if video_settings:
            self.sessions[session_id].video_settings = video_settings
        if wan_settings:
            self.sessions[session_id].wan_settings = wan_settings
        
        self._save_session_metadata(session_id)
    
    def add_reference_image(self, session_id: str, image_path: str) -> str:
        """Add a reference image to the session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Initialize reference_images list if it doesn't exist
        if self.sessions[session_id].reference_images is None:
            self.sessions[session_id].reference_images = []
        
        # Add image path to session
        self.sessions[session_id].reference_images.append(image_path)
        try:
            self._save_session_metadata(session_id)
        except Exception as e:
            print(f"Warning: Could not save session metadata for {session_id}: {e}")
        
        return image_path
    
    def get_reference_images(self, session_id: str) -> List[str]:
        """Get all reference images for a session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        return self.sessions[session_id].reference_images or []
    
    def remove_reference_image(self, session_id: str, image_path: str):
        """Remove a reference image from the session."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        if self.sessions[session_id].reference_images:
            try:
                self.sessions[session_id].reference_images.remove(image_path)
                self._save_session_metadata(session_id)
                
                # Also delete the physical file
                full_path = Path(image_path)
                if full_path.exists():
                    full_path.unlink()
            except ValueError:
                pass  # Image not in list

    def add_wan_generation(self, session_id: str, generation_name: str, prompt: str, 
                          generation_type: str, job_id: str, settings: Optional[Dict] = None,
                          source_generation: Optional[str] = None):
        """Add a WAN generation to the session registry."""
        if session_id not in self.sessions:
            raise ValueError(f"Session {session_id} not found")
        
        # Initialize generations dict if it doesn't exist
        if self.sessions[session_id].generations is None:
            self.sessions[session_id].generations = {}
        
        # Add to session registry
        generation_info = {
            "prompt": prompt,
            "type": generation_type,
            "timestamp": datetime.now().isoformat(),
            "job_id": job_id
        }
        
        if source_generation:
            generation_info["source"] = source_generation
            
        self.sessions[session_id].generations[generation_name] = generation_info
        
        # Save individual generation metadata file
        session_dir = self.get_session_dir(session_id)
        generation_dir = session_dir / "wan_generations" / generation_name
        if generation_dir.exists():
            generation_metadata = {
                "generation_name": generation_name,
                "generation_type": generation_type,
                "prompt": prompt,
                "timestamp": generation_info["timestamp"],
                "job_id": job_id,
                "session_id": session_id,
                "settings": settings or {},
                "source_generation": source_generation
            }
            
            metadata_file = generation_dir / "generation_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(generation_metadata, f, indent=2)
        
        self._save_session_metadata(session_id)
    
    def get_session(self, session_id: str) -> Optional[SessionMetadata]:
        """Get session metadata."""
        return self.sessions.get(session_id)
    
    def list_sessions(self, project_name: Optional[str] = None) -> List[SessionMetadata]:
        """List all sessions, optionally filtered by project name."""
        sessions = list(self.sessions.values())
        if project_name:
            sessions = [s for s in sessions if s.project_name == project_name]
        return sorted(sessions, key=lambda x: x.timestamp, reverse=True)
    
    def close_session(self, session_id: str):
        """Mark session as completed."""
        if session_id in self.sessions:
            self.sessions[session_id].status = "completed"
            self._save_session_metadata(session_id)
    
    def _save_session_metadata(self, session_id: str):
        """Save session metadata to file."""
        if session_id not in self.sessions:
            return
        
        session_dir = self.get_session_dir(session_id)
        metadata_file = session_dir / "metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(asdict(self.sessions[session_id]), f, indent=2)
    
    def _load_existing_sessions(self):
        """Load existing sessions from metadata files."""
        if not self.base_output_dir.exists():
            return
        
        for session_dir in self.base_output_dir.iterdir():
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        data = json.load(f)
                    metadata = SessionMetadata(**data)
                    self.sessions[metadata.session_id] = metadata
                except Exception as e:
                    print(f"Warning: Could not load session metadata from {metadata_file}: {e}")


# Global session manager instance
session_manager = SessionManager()