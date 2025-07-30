import { useState } from 'react';
import ImageUpload from './ImageUpload';

interface VideoUploadFormProps {
  videoBlob: Blob;
  onUpload: (formData: UploadFormData) => Promise<void>;
  onCancel: () => void;
  isUploading: boolean;
  uploadProgress: number;
}

export interface UploadFormData {
  videoBlob: Blob;
  imageFile: File | null;
  sessionId: string;
  options: {
    prompt: string;
    num_extensions: number;
    invert_mask: boolean;
    resolution: string;
    use_distorted_video: boolean;
  };
}

export default function VideoUploadForm({ 
  videoBlob, 
  onUpload, 
  onCancel, 
  isUploading, 
  uploadProgress 
}: VideoUploadFormProps) {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [sessionId, setSessionId] = useState(() => {
    // Generate a session ID if one doesn't exist
    return localStorage.getItem('audioVisualizationSessionId') || 
           `session_${Date.now()}_${Math.random().toString(36).substring(2, 11)}`;
  });
  
  // Processing options with defaults from server API
  const [prompt, setPrompt] = useState('');
  const [numExtensions, setNumExtensions] = useState(1);
  const [invertMask, setInvertMask] = useState(false);
  const [resolution, setResolution] = useState('720p');
  const [useDistortedVideo, setUseDistortedVideo] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();


    // Store session ID for future use
    localStorage.setItem('audioVisualizationSessionId', sessionId);

    const formData: UploadFormData = {
      videoBlob,
      imageFile,
      sessionId,
      options: {
        prompt,
        num_extensions: numExtensions,
        invert_mask: invertMask,
        resolution,
        use_distorted_video: useDistortedVideo
      }
    };

    await onUpload(formData);
  };

  return (
    <div className="upload-form-overlay">
      <div className="upload-form-modal">
        <div className="form-header">
          <h3>Upload Video with Thumbnail</h3>
          <button 
            type="button" 
            onClick={onCancel}
            className="close-button"
            disabled={isUploading}
          >
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-section">
            <ImageUpload 
              onImageSelect={setImageFile}
              selectedImage={imageFile}
            />
          </div>

          <div className="form-section">
            <label className="form-label">Session ID</label>
            <input
              type="text"
              value={sessionId}
              onChange={(e) => setSessionId(e.target.value)}
              className="form-input"
              placeholder="Enter session ID"
              disabled={isUploading}
              required
            />
            <p className="form-help">Session ID for organizing your uploads on the server</p>
          </div>

          <div className="form-section">
            <label className="form-label">Prompt (Optional)</label>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              className="form-input"
              placeholder="e.g., bellowing flames"
              disabled={isUploading}
            />
            <p className="form-help">Text prompt for video processing</p>
          </div>

          <div className="form-row">
            <div className="form-group">
              <label className="form-label">Extensions</label>
              <input
                type="number"
                min="1"
                max="10"
                value={numExtensions}
                onChange={(e) => setNumExtensions(parseInt(e.target.value))}
                className="form-input"
                disabled={isUploading}
              />
            </div>

            <div className="form-group">
              <label className="form-label">Resolution</label>
              <select
                value={resolution}
                onChange={(e) => setResolution(e.target.value)}
                className="form-select"
                disabled={isUploading}
              >
                <option value="480p">480p</option>
                <option value="720p">720p</option>
                <option value="1080p">1080p</option>
              </select>
            </div>
          </div>

          <div className="form-section">
            <div className="checkbox-group">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={invertMask}
                  onChange={(e) => setInvertMask(e.target.checked)}
                  disabled={isUploading}
                />
                Invert Mask
              </label>
              
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useDistortedVideo}
                  onChange={(e) => setUseDistortedVideo(e.target.checked)}
                  disabled={isUploading}
                />
                Use Distorted Video
              </label>
            </div>
          </div>

          {isUploading && (
            <div className="upload-progress-section">
              <div className="progress-label">Uploading... {uploadProgress.toFixed(0)}%</div>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${uploadProgress}%` }}
                />
              </div>
            </div>
          )}

          <div className="form-actions">
            <button
              type="button"
              onClick={onCancel}
              className="cancel-button"
              disabled={isUploading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="upload-button"
              disabled={isUploading}
            >
              {isUploading ? 'Uploading...' : 'Upload Video'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}