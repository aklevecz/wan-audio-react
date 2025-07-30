import { useState, useRef } from 'react';

interface ImageUploadProps {
  onImageSelect: (file: File | null) => void;
  selectedImage: File | null;
}

export default function ImageUpload({ onImageSelect, selectedImage }: ImageUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isValidImageFile = (file: File): boolean => {
    const validTypes = ['image/png', 'image/jpg', 'image/jpeg', 'image/bmp', 'image/webp'];
    return validTypes.includes(file.type) || 
           file.name.match(/\.(png|jpg|jpeg|bmp|webp)$/i) !== null;
  };

  const handleFile = (file: File) => {
    if (isValidImageFile(file)) {
      onImageSelect(file);
      
      // Create preview
      const url = URL.createObjectURL(file);
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
      setPreviewUrl(url);
    } else {
      alert('Please select a valid image file (PNG, JPG, JPEG, BMP, WebP)');
    }
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  const clearImage = () => {
    onImageSelect(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="image-upload">
      <label className="image-upload-label">Thumbnail Image</label>
      
      <div
        className={`image-dropzone ${isDragOver ? 'drag-over' : ''} ${selectedImage ? 'has-image' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".png,.jpg,.jpeg,.bmp,.webp"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        
        {previewUrl ? (
          <div className="image-preview">
            <img src={previewUrl} alt="Preview" className="preview-image" />
            <div className="image-overlay">
              <button 
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  clearImage();
                }}
                className="clear-image-button"
              >
                ✕
              </button>
            </div>
          </div>
        ) : (
          <div className="image-placeholder">
            <div className="upload-icon">🖼️</div>
            <p>Drop an image here or click to browse</p>
            <p className="supported-formats">Supports: PNG, JPG, JPEG, BMP, WebP</p>
          </div>
        )}
      </div>
      
      {selectedImage && (
        <div className="image-info">
          <div className="info-row">
            <span>File:</span>
            <span>{selectedImage.name}</span>
          </div>
          <div className="info-row">
            <span>Size:</span>
            <span>{(selectedImage.size / 1024 / 1024).toFixed(2)} MB</span>
          </div>
        </div>
      )}
    </div>
  );
}