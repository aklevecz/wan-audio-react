import { useState, useRef } from 'react';
import { AudioAnalyzer } from './lib/audio/analyzer';
import { AudioPlayer } from './lib/audio/player';
import AudioVisualization from './AudioVisualization';
import type { RMSAnalysisResult } from './lib/audio/analyzer';
import type { AudioSegment } from './lib/audio/player';

interface AudioFileDropzoneProps {
  onFileSelect?: (file: File) => void;
  onAnalysisComplete?: (result: RMSAnalysisResult) => void;
}

interface Selection {
  startIndex: number;
  endIndex: number;
  startTime: number;
  endTime: number;
}

export default function AudioFileDropzone({ onFileSelect, onAnalysisComplete }: AudioFileDropzoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [analysisResult, setAnalysisResult] = useState<RMSAnalysisResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selection, setSelection] = useState<Selection | null>(null);
  const [isSelecting, setIsSelecting] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const analyzerRef = useRef<AudioAnalyzer | null>(null);
  const playerRef = useRef<AudioPlayer | null>(null);

  const audioFileTypes = ['audio/mp3', 'audio/wav', 'audio/ogg', 'audio/m4a', 'audio/aac', 'audio/flac'];

  const isValidAudioFile = (file: File): boolean => {
    return audioFileTypes.includes(file.type) || 
           file.name.match(/\.(mp3|wav|ogg|m4a|aac|flac)$/i) !== null;
  };

  const handleFile = async (file: File) => {
    if (isValidAudioFile(file)) {
      setSelectedFile(file);
      setAnalysisResult(null);
      setSelection(null);
      setIsAnalyzing(true);
      onFileSelect?.(file);

      try {
        // Initialize analyzer and player
        if (!analyzerRef.current) {
          analyzerRef.current = new AudioAnalyzer();
        }
        if (!playerRef.current) {
          playerRef.current = new AudioPlayer();
        }
        
        // Load file into player
        await playerRef.current.loadAudioFile(file);
        
        // Set up audio event listeners
        const audio = (playerRef.current as any).audioElement;
        if (audio) {
          audio.addEventListener('ended', () => setIsPlaying(false));
          audio.addEventListener('pause', () => setIsPlaying(false));
          audio.addEventListener('play', () => setIsPlaying(true));
        }
        
        // Analyze audio with higher temporal resolution (20ms segments = 50 FPS)
        const result = await analyzerRef.current.analyzeRMS(file, 0.02);
        setAnalysisResult(result);
        onAnalysisComplete?.(result);
      } catch (error) {
        console.error('Audio analysis failed:', error);
        alert(`Failed to analyze audio: ${error instanceof Error ? error.message : 'Unknown error'}`);
      } finally {
        setIsAnalyzing(false);
      }
    } else {
      alert('Please select a valid audio file (MP3, WAV, OGG, M4A, AAC, FLAC)');
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

  const handleWaveformMouseDown = (index: number) => {
    if (!analysisResult) return;
    
    setIsSelecting(true);
    const timePerPoint = analysisResult.duration / analysisResult.waveformData.length;
    const startTime = index * timePerPoint;
    
    setSelection({
      startIndex: index,
      endIndex: index,
      startTime,
      endTime: startTime
    });
  };

  const handleWaveformMouseMove = (index: number) => {
    if (!isSelecting || !selection || !analysisResult) return;
    
    const timePerPoint = analysisResult.duration / analysisResult.waveformData.length;
    const endTime = index * timePerPoint;
    
    setSelection({
      ...selection,
      endIndex: index,
      endTime
    });
  };

  const handleWaveformMouseUp = () => {
    setIsSelecting(false);
  };

  const clearSelection = () => {
    setSelection(null);
  };

  const playSelection = async () => {
    if (!selection || !playerRef.current) return;
    
    try {
      const segment: AudioSegment = {
        startTime: Math.min(selection.startTime, selection.endTime),
        endTime: Math.max(selection.startTime, selection.endTime)
      };
      
      await playerRef.current.playSegment(segment);
    } catch (error) {
      console.error('Playback failed:', error);
    }
  };

  const playFull = async () => {
    if (!playerRef.current) return;
    
    try {
      await playerRef.current.playFull();
    } catch (error) {
      console.error('Playback failed:', error);
    }
  };

  const pausePlayback = () => {
    if (playerRef.current) {
      playerRef.current.pause();
    }
  };

  const resumePlayback = async () => {
    if (!playerRef.current) return;
    
    try {
      // Resume from current position by calling play on the audio element
      const audio = (playerRef.current as any).audioElement;
      if (audio) {
        await audio.play();
      }
    } catch (error) {
      console.error('Resume failed:', error);
    }
  };

  const stopPlayback = () => {
    if (playerRef.current) {
      playerRef.current.stop();
    }
  };

  return (
    <>
      <div
        className={`dropzone ${isDragOver ? 'drag-over' : ''} ${selectedFile ? 'has-file' : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onClick={handleClick}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*,.mp3,.wav,.ogg,.m4a,.aac,.flac"
          onChange={handleFileInput}
          style={{ display: 'none' }}
        />
        
        <div className="dropzone-content">
          {selectedFile ? (
            <div className="dropzone-status">
              <p>File selected: {selectedFile.name}</p>
              {isAnalyzing && <p>Analyzing...</p>}
            </div>
          ) : (
            <div className="upload-placeholder">
              <p>Drop audio files here or click to browse</p>
              <p className="supported-formats">Supports: MP3, WAV, OGG, M4A, AAC, FLAC</p>
            </div>
          )}
        </div>
      </div>

      {selectedFile && (
        <div className="file-info">
          <table className="info-table">
            <tbody>
              <tr>
                <td>File</td>
                <td>{selectedFile.name}</td>
              </tr>
              <tr>
                <td>Size</td>
                <td>{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</td>
              </tr>
              {isAnalyzing && (
                <tr>
                  <td>Status</td>
                  <td>Analyzing...</td>
                </tr>
              )}
              {analysisResult && !isAnalyzing && (
                <>
                  <tr>
                    <td>Duration</td>
                    <td>{analysisResult.duration.toFixed(2)}s</td>
                  </tr>
                  <tr>
                    <td>Sample Rate</td>
                    <td>{analysisResult.sampleRate.toLocaleString()} Hz</td>
                  </tr>
                  <tr>
                    <td>Channels</td>
                    <td>{analysisResult.channels}</td>
                  </tr>
                  <tr>
                    <td>Overall RMS</td>
                    <td>{analysisResult.rmsData.overall.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>Peak RMS</td>
                    <td>{analysisResult.rmsData.peak.toFixed(4)}</td>
                  </tr>
                  <tr>
                    <td>Segments</td>
                    <td>{analysisResult.rmsData.segments.length}</td>
                  </tr>
                </>
              )}
            </tbody>
          </table>
          
          {analysisResult && !isAnalyzing && (
            <>
              <div className="waveform-table">
                <div className="waveform-header">
                  Waveform ({analysisResult.waveformData.length} points)
                  {selection && (
                    <span className="selection-info">
                      {' | Selection: '}
                      {Math.min(selection.startTime, selection.endTime).toFixed(2)}s - 
                      {Math.max(selection.startTime, selection.endTime).toFixed(2)}s
                    </span>
                  )}
                </div>
                <div 
                  className="waveform-container"
                  onMouseUp={handleWaveformMouseUp}
                  onMouseLeave={handleWaveformMouseUp}
                >
                  {analysisResult.waveformData.map((amplitude, index) => {
                    const isSelected = selection && 
                      index >= Math.min(selection.startIndex, selection.endIndex) && 
                      index <= Math.max(selection.startIndex, selection.endIndex);
                    
                    return (
                      <div
                        key={index}
                        className={`waveform-bar ${isSelected ? 'selected' : ''}`}
                        style={{
                          height: `${Math.max(10, amplitude * 70)}px`,
                          backgroundColor: isSelected ? '#4CAF50' : (amplitude > 0.1 ? '#333' : '#ccc')
                        }}
                        onMouseDown={() => handleWaveformMouseDown(index)}
                        onMouseMove={() => handleWaveformMouseMove(index)}
                        title={`${index}: ${amplitude.toFixed(3)} | ${(index * analysisResult.duration / analysisResult.waveformData.length).toFixed(2)}s`}
                      />
                    );
                  })}
                </div>
              </div>
              
              <div className="player-controls">
                {isPlaying ? (
                  <>
                    <button onClick={pausePlayback}>
                      Pause
                    </button>
                    <button onClick={stopPlayback}>
                      Stop
                    </button>
                  </>
                ) : (
                  <>
                    <button onClick={playFull}>
                      Play Full
                    </button>
                    {selection && (
                      <button onClick={playSelection}>
                        Play Selection
                      </button>
                    )}
                    <button onClick={resumePlayback}>
                      Resume
                    </button>
                  </>
                )}
                
                {selection && (
                  <button onClick={clearSelection} disabled={isPlaying}>
                    Clear Selection
                  </button>
                )}
              </div>
              
              <AudioVisualization
                analysisResult={analysisResult}
                audioPlayer={playerRef.current}
                isPlaying={isPlaying}
              />
            </>
          )}
        </div>
      )}
    </>
  );
}