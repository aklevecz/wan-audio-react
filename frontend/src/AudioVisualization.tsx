import { useRef, useEffect, useState } from 'react';
import VideoUploadForm, { type UploadFormData } from './VideoUploadForm';
import WebGLAudioVisualization from './WebGLAudioVisualization';
import { WebGLRenderer } from './lib/webgl/renderer';
import type { RMSAnalysisResult } from './lib/audio/analyzer';
import type { AudioPlayer } from './lib/audio/player';
import { BACKEND_URL } from './constants';

interface AudioVisualizationProps {
  analysisResult: RMSAnalysisResult | null;
  audioPlayer: AudioPlayer | null;
  isPlaying: boolean;
}

export default function AudioVisualization({
  analysisResult,
  audioPlayer,
  isPlaying
}: AudioVisualizationProps) {
  // Rendering mode selection
  const [useWebGL, setUseWebGL] = useState(false);
  const [webGLSupported, setWebGLSupported] = useState(false);

  // User-adjustable parameters
  const width = 720;
  const height = 720;
  const [minRadius, setMinRadius] = useState(10);
  const [maxRadius, setMaxRadius] = useState(150);
  const [feedback, setFeedback] = useState(0.1);
  const [smoothWindow, setSmoothWindow] = useState(3);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number>(0);
  const previousFrameRef = useRef<ImageData | null>(null);
  const [isEnabled, setIsEnabled] = useState(true);

  // Video recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordingProgress, setRecordingProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [showUploadForm, setShowUploadForm] = useState(false);
  const [recordedVideoBlob, setRecordedVideoBlob] = useState<Blob | null>(null);

  // Job polling state
  const [activeJobs, setActiveJobs] = useState<Array<{
    jobId: string;
    sessionId: string;
    status: string;
    videoFilename: string;
    startTime: number;
  }>>(() => {
    // Load active jobs from localStorage on component mount
    try {
      const saved = localStorage.getItem('activeJobs');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [completedJobs, setCompletedJobs] = useState<Array<{
    jobId: string;
    sessionId: string;
    videoFilename: string;
    completedAt: number;
    result: any;
  }>>([]);
  const [pollingInterval, setPollingInterval] = useState<number | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const recordedChunksRef = useRef<Blob[]>([]);
  const recordingStartTimeRef = useRef<number>(0);


  // Get current intensity based on audio playback time
  const getCurrentIntensity = (): number => {
    if (!analysisResult || !audioPlayer || !isPlaying) {
      return 0;
    }

    const currentTime = audioPlayer.getCurrentTime();
    const segments = analysisResult.rmsData.segments;

    // Find the segment that contains the current time
    const currentSegment = segments.find(segment =>
      currentTime >= segment.startTime && currentTime <= segment.endTime
    );

    return currentSegment ? currentSegment.rms : 0;
  };

  // Map intensity to radius
  const intensityToRadius = (intensity: number): number => {
    if (!analysisResult) return minRadius;

    const maxIntensity = analysisResult.rmsData.peak;
    if (maxIntensity === 0) return minRadius;

    const normalizedIntensity = Math.min(intensity / maxIntensity, 1);
    return minRadius + (normalizedIntensity * (maxRadius - minRadius));
  };

  // Render a single frame
  const renderFrame = () => {
    const canvas = canvasRef.current;
    if (!canvas || !isEnabled) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const centerX = width / 2;
    const centerY = height / 2;
    const currentIntensity = getCurrentIntensity();
    const radius = intensityToRadius(currentIntensity);

    // Apply feedback effect (trails)
    if (feedback > 0 && previousFrameRef.current) {
      const imageData = ctx.createImageData(width, height);
      const prevData = previousFrameRef.current.data;
      const currentData = imageData.data;

      // Fade previous frame
      for (let i = 0; i < prevData.length; i += 4) {
        currentData[i] = prevData[i] * feedback;     // R
        currentData[i + 1] = prevData[i + 1] * feedback; // G
        currentData[i + 2] = prevData[i + 2] * feedback; // B
        currentData[i + 3] = 255; // A
      }

      ctx.putImageData(imageData, 0, 0);
    } else {
      // Clear canvas to black
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, width, height);
    }

    // Draw white circle
    ctx.fillStyle = 'white';
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.fill();

    // Store current frame for feedback effect
    if (feedback > 0) {
      previousFrameRef.current = ctx.getImageData(0, 0, width, height);
    }
  };

  // Animation loop
  const animate = () => {
    renderFrame();

    if (isPlaying && isEnabled) {
      animationRef.current = requestAnimationFrame(animate);
    }
  };

  // Start/stop animation based on playback state
  useEffect(() => {
    if (isPlaying && isEnabled && analysisResult) {
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      // Render one final frame when stopped
      if (!isPlaying && analysisResult) {
        renderFrame();
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, isEnabled, analysisResult, audioPlayer]);

  // Video recording functions
  const startRecording = async () => {
    const canvas = canvasRef.current;
    if (!canvas || !audioPlayer || !analysisResult) return;

    try {
      // Get canvas stream at 30fps
      const stream = canvas.captureStream(30);

      // Add audio track if available
      const audioElement = (audioPlayer as any).audioElement;
      if (audioElement && audioElement.captureStream) {
        const audioStream = audioElement.captureStream();
        audioStream.getAudioTracks().forEach((track: MediaStreamTrack) => {
          stream.addTrack(track);
        });
      }

      // Setup MediaRecorder
      recordedChunksRef.current = [];
      mediaRecorderRef.current = new MediaRecorder(stream, {
        mimeType: 'video/webm;codecs=vp9,opus'
      });

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = () => {
        setIsRecording(false);
        setRecordingProgress(0);

        // Create video blob from recorded chunks
        if (recordedChunksRef.current.length > 0) {
          const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
          setRecordedVideoBlob(blob);
          setShowUploadForm(true);
        }
      };

      // Start recording
      mediaRecorderRef.current.start(100); // Collect data every 100ms
      setIsRecording(true);
      recordingStartTimeRef.current = Date.now();

      // Update progress based on audio duration
      const progressInterval = setInterval(() => {
        if (!isRecording) {
          clearInterval(progressInterval);
          return;
        }

        const elapsed = (Date.now() - recordingStartTimeRef.current) / 1000;
        const progress = (elapsed / analysisResult.duration) * 100;
        setRecordingProgress(Math.min(progress, 100));

        // Auto-stop when audio duration reached
        if (elapsed >= analysisResult.duration) {
          stopRecording();
          clearInterval(progressInterval);
        }
      }, 100);

    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to start recording. Please check browser permissions.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
    }
  };

  const downloadVideo = () => {
    if (!recordedVideoBlob) return;

    const url = URL.createObjectURL(recordedVideoBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audio_visualization_${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleUploadFormSubmit = async (formData: UploadFormData) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      const serverFormData = new FormData();

      // Add required files
      serverFormData.append('video_file', formData.videoBlob, `visualization_${Date.now()}.webm`);
      if (formData.imageFile) {
        serverFormData.append('image_file', formData.imageFile);
      }

      // Add processing options
      serverFormData.append('prompt', formData.options.prompt);
      serverFormData.append('num_extensions', formData.options.num_extensions.toString());
      serverFormData.append('invert_mask', formData.options.invert_mask.toString());
      serverFormData.append('resolution', formData.options.resolution);
      serverFormData.append('use_distorted_video', formData.options.use_distorted_video.toString());

      const response = await fetch(`${BACKEND_URL}/session/${formData.sessionId}/upload_video_and_image`, {
        method: 'POST',
        body: serverFormData
      });

      setIsUploading(false);
      setUploadProgress(100);
      setShowUploadForm(false);

      if (response.ok) {
        const result = await response.json();
        // alert(`Upload successful!\n\n` +
        //       `Session ID: ${result.session_id}\n` +
        //       `Sieve Job ID: ${result.sieve_job_id}\n` +
        //       `Video: ${result.video_filename} (${result.video_upload_status})\n` +
        //       `Image: ${result.image_filename} (${result.image_status})\n` +
        //       `Video URL: ${result.video_path}\n\n` +
        //       `${result.message}\n\nStarting job polling...`);
        console.log(`Upload successful!`, result);
        // Start polling the Sieve job
        const jobId = result.sieve_job_id;
        if (jobId) {
          console.log(`Starting job polling for Sieve Job ID: ${jobId}`);
          addJobToPolling(jobId, result.session_id, result.video_filename);
        } else {
          console.warn(`No valid Sieve Job ID found.`);
        }

        // Clear recorded video
        setRecordedVideoBlob(null);
        recordedChunksRef.current = [];
      } else {
        console.error('Upload failed with status:', response.status, response.statusText);
        try {
          const error = await response.json();
          console.error('Server error details:', error);
          alert(`Upload failed: ${error.detail || 'Unknown error'}`);
        } catch (parseError) {
          console.error('Failed to parse error response:', parseError);
          alert(`Upload failed with status ${response.status}: ${response.statusText}`);
        }
      }

    } catch (error: any) {
      console.error('Upload error:', error);
      setIsUploading(false);
      setUploadProgress(0);
      alert(`Upload failed: ${error.message || 'Please try again.'}`);
    }
  };

  const handleUploadCancel = () => {
    setShowUploadForm(false);
    setIsUploading(false);
    setUploadProgress(0);
  };

  // Job polling functions
  const pollJobStatus = async (jobId: string) => {
    try {
      const response = await fetch(`${BACKEND_URL}/sieve/job/${jobId}`);

      if (response.ok) {
        const result = await response.json();
        return result;
      } else {
        console.error('Failed to poll job status:', response.statusText);
        return null;
      }
    } catch (error) {
      console.error('Error polling job status:', error);
      return null;
    }
  };

  const updateJobStatus = (jobId: string, newStatus: string, result?: any) => {
    setActiveJobs(prevJobs => {
      const updatedJobs = prevJobs.map(job =>
        job.jobId === jobId
          ? {
            ...job,
            status: newStatus,
            // Update startTime when job moves from queued to processing
            startTime: newStatus === 'processing' && job.status === 'queued'
              ? Date.now()
              : job.startTime
          }
          : job
      );
      // Update localStorage
      localStorage.setItem('activeJobs', JSON.stringify(updatedJobs));
      return updatedJobs;
    });

    // Handle completed jobs
    if (newStatus === 'finished') {
      // Find the job to get additional info
      const job = activeJobs.find(j => j.jobId === jobId);
      if (job) {
        // Add to completed jobs list with full result data
        console.log('Adding completed job with result:', { jobId, result });
        const completedJob = {
          jobId,
          sessionId: job.sessionId,
          videoFilename: job.videoFilename,
          completedAt: Date.now(),
          result: result
        };
        setCompletedJobs(prev => [...prev, completedJob]);
      }

      const videoUrls = [];

      // Check both result structure possibilities
      const videoData = result.result || (result.job_data?.outputs?.[0]?.data);

      if (videoData?.combined_video?.url) {
        videoUrls.push(`Combined Video: ${videoData.combined_video.url}`);
      }
      if (videoData?.r2_urls?.combined_video) {
        videoUrls.push(`R2 Combined: ${videoData.r2_urls.combined_video}`);
      }

      console.log(`Job completed successfully! Job ID: ${jobId}`, { result, videoData });
      removeCompletedJob(jobId);
    } else if (newStatus === 'failed') {
      alert(`Job failed!\nJob ID: ${jobId}`);
      removeCompletedJob(jobId);
    }
  };

  const removeCompletedJob = (jobId: string) => {
    setActiveJobs(prevJobs => {
      const updatedJobs = prevJobs.filter(job => job.jobId !== jobId);
      // Update localStorage
      localStorage.setItem('activeJobs', JSON.stringify(updatedJobs));
      return updatedJobs;
    });
  };

  const addJobToPolling = (jobId: string, sessionId: string, videoFilename: string) => {
    const newJob = {
      jobId,
      sessionId,
      status: 'running',
      videoFilename,
      startTime: Date.now()
    };

    setActiveJobs(prevJobs => {
      const updatedJobs = [...prevJobs, newJob];
      // Save to localStorage
      localStorage.setItem('activeJobs', JSON.stringify(updatedJobs));
      return updatedJobs;
    });
  };

  // Polling effect
  useEffect(() => {
    if (activeJobs.length > 0) {
      const interval = setInterval(async () => {
        for (const job of activeJobs) {
          if (job.status === 'processing' || job.status === 'queued' || job.status === 'running') {
            const result = await pollJobStatus(job.jobId);
            console.log(`Polling job ${job.jobId}:`, result);
            if (result && result.status !== job.status) {
              console.log('Full polling result structure:', JSON.stringify(result, null, 2));
              console.log('result.result:', result.result);
              console.log('result.job_data:', result.job_data);
              console.log('result.job_data?.outputs?.[0]?.data:', result.job_data?.outputs?.[0]?.data);
              updateJobStatus(job.jobId, result.status, result);
            }
          }
        }
      }, 3000); // Poll every 3 seconds

      setPollingInterval(interval);

      return () => {
        clearInterval(interval);
        setPollingInterval(null);
      };
    } else {
      if (pollingInterval) {
        clearInterval(pollingInterval);
        setPollingInterval(null);
      }
    }
  }, [activeJobs]);

  // Check WebGL support on component mount
  useEffect(() => {
    setWebGLSupported(WebGLRenderer.isSupported());
  }, []);

  // Initialize canvas when component mounts
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.width = width;
    canvas.height = height;

    // Render initial black frame
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'black';
      ctx.fillRect(0, 0, width, height);
    }
  }, [width, height]);

  // Render WebGL version if enabled and supported
  if (useWebGL && webGLSupported) {
    return (
      <WebGLAudioVisualization
        analysisResult={analysisResult}
        audioPlayer={audioPlayer}
        isPlaying={isPlaying}
      />
    );
  }

  return (
    <div className="audio-visualization">
      <div className="visualization-header">
        <span>Audio Visualization</span>
        <div className="header-controls">
          <label className="visualization-toggle">
            <input
              type="checkbox"
              checked={isEnabled}
              onChange={(e) => setIsEnabled(e.target.checked)}
            />
            Enable
          </label>

          {webGLSupported && (
            <label className="visualization-toggle">
              <input
                type="checkbox"
                checked={useWebGL}
                onChange={(e) => setUseWebGL(e.target.checked)}
              />
              WebGL Distortion
            </label>
          )}

          {isRecording && (
            <div className="recording-indicator">
              <span className="recording-dot"></span>
              Recording {recordingProgress.toFixed(0)}%
            </div>
          )}
        </div>
      </div>

      <div className="visualization-controls">

        <div className="control-group">
          <label>Min Radius: {minRadius}px</label>
          <input
            type="range"
            min="5"
            max="50"
            value={minRadius}
            onChange={(e) => setMinRadius(parseInt(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Max Radius: {maxRadius}px</label>
          <input
            type="range"
            min="50"
            max="250"
            value={maxRadius}
            onChange={(e) => setMaxRadius(parseInt(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Feedback: {(feedback * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0"
            max="90"
            value={feedback * 100}
            onChange={(e) => setFeedback(parseInt(e.target.value) / 100)}
          />
        </div>

        <div className="control-group">
          <label>Smoothing: {smoothWindow}</label>
          <input
            type="range"
            min="1"
            max="10"
            value={smoothWindow}
            onChange={(e) => setSmoothWindow(parseInt(e.target.value))}
          />
        </div>
      </div>

      <div className="video-controls">
        <div className="video-buttons">
          {!isRecording ? (
            <button
              onClick={startRecording}
              disabled={!analysisResult || !isPlaying}
              className="record-button"
            >
              Record Video
            </button>
          ) : (
            <button
              onClick={stopRecording}
              className="stop-button"
            >
              Stop Recording
            </button>
          )}

          {recordedVideoBlob && !isRecording && (
            <>
              <button onClick={downloadVideo} className="download-button">
                Download Video
              </button>
              <button
                onClick={() => setShowUploadForm(true)}
                disabled={isUploading}
                className="upload-button"
              >
                Upload to Server
              </button>
            </>
          )}
        </div>

        {isRecording && (
          <div className="recording-progress">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{ width: `${recordingProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {isUploading && (
          <div className="upload-progress">
            <div className="progress-bar">
              <div
                className="progress-fill upload-fill"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      <canvas
        ref={canvasRef}
        className="visualization-canvas"
        style={{
          border: '1px solid #ddd',
          display: 'block'
        }}
      />

      <div className="visualization-info">
        <div>
          Intensity: {analysisResult ? getCurrentIntensity().toFixed(4) : '0.0000'}
        </div>
        <div>
          Radius: {analysisResult ? intensityToRadius(getCurrentIntensity()).toFixed(1) : '0.0'}px
        </div>
        <div>
          Time: {audioPlayer ? audioPlayer.getCurrentTime().toFixed(2) : '0.00'}s
        </div>
      </div>

      {activeJobs.length > 0 && (
        <div className="job-status-section">
          <div className="job-status-header">Processing Jobs</div>
          {activeJobs.map((job) => (
            <div key={job.jobId} className="job-status-item">
              <div className="job-info">
                <div className="job-filename">{job.videoFilename}</div>
                <div className="job-details">
                  <span className="job-id">Job: {String(job.jobId).substring(0, 8)}...</span>
                  <span className={`job-status ${job.status}`}>{job.status}</span>
                </div>
              </div>
              <div className="job-duration">
                {Math.floor((Date.now() - job.startTime) / 1000)}s
              </div>
            </div>
          ))}
        </div>
      )}

      {completedJobs.length > 0 && (
        <div className="completed-jobs-section">
          <div className="job-status-header">
            Completed Jobs
            <button
              onClick={() => setCompletedJobs([])}
              className="clear-completed-button"
              style={{ marginLeft: '10px', fontSize: '12px', padding: '4px 8px' }}
            >
              Clear All
            </button>
          </div>
          {completedJobs.map((job) => {
            // Extract video data from result (same logic as manual polling)
            console.log('Rendering completed job:', { jobId: job.jobId, result: job.result });
            const videoData = job.result.result || (job.result.job_data?.outputs?.[0]?.data);
            console.log('Extracted videoData:', videoData);

            return (
              <div key={job.jobId} className="completed-job-item">
                <div className="job-header">
                  <div className="job-info">
                    <div className="job-filename">{job.videoFilename}</div>
                    <div className="job-details">
                      <span className="job-id">Job: {String(job.jobId).substring(0, 8)}...</span>
                      <span className="job-completed-time">
                        Completed: {new Date(job.completedAt).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                  <button
                    onClick={() => setCompletedJobs(prev => prev.filter(j => j.jobId !== job.jobId))}
                    className="remove-job-button"
                    style={{ fontSize: '16px', padding: '4px 8px' }}
                  >
                    ✕
                  </button>
                </div>

                {videoData && (
                  <div className="job-results">
                    {/* Display videos if available */}
                    {videoData.combined_video?.url && (
                      <div className="video-section">
                        <h5>Combined Video:</h5>
                        <video
                          controls
                          className="result-video"
                          style={{ width: '100%', maxWidth: '600px', height: 'auto' }}
                        >
                          <source src={videoData.combined_video.url} type="video/mp4" />
                          Your browser does not support the video tag.
                        </video>
                      </div>
                    )}

                    {videoData.initial_video?.url && (
                      <div className="video-section">
                        <h5>Initial Video:</h5>
                        <video
                          controls
                          className="result-video"
                          style={{ width: '100%', maxWidth: '400px', height: 'auto' }}
                        >
                          <source src={videoData.initial_video.url} type="video/mp4" />
                          Your browser does not support the video tag.
                        </video>
                      </div>
                    )}

                    {videoData.extension_videos && videoData.extension_videos.length > 0 && (
                      <div className="video-section">
                        <h5>Extension Videos:</h5>
                        {videoData.extension_videos.map((video: any, index: number) => (
                          <div key={index} className="extension-video">
                            <h6>Extension {index + 1}:</h6>
                            <video
                              controls
                              className="result-video"
                              style={{ width: '100%', maxWidth: '400px', height: 'auto' }}
                            >
                              <source src={video.url} type="video/mp4" />
                              Your browser does not support the video tag.
                            </video>
                          </div>
                        ))}
                      </div>
                    )}

                    {/* R2 URLs if available */}
                    {videoData.r2_urls && (
                      <div className="r2-urls-section">
                        <h5>R2 Storage URLs:</h5>
                        <div className="url-list">
                          {videoData.r2_urls.combined_video && (
                            <div className="url-item">
                              <strong>Combined:</strong>
                              <a href={videoData.r2_urls.combined_video} target="_blank" rel="noopener noreferrer">
                                {videoData.r2_urls.combined_video}
                              </a>
                            </div>
                          )}
                          {videoData.r2_urls.initial_video && (
                            <div className="url-item">
                              <strong>Initial:</strong>
                              <a href={videoData.r2_urls.initial_video} target="_blank" rel="noopener noreferrer">
                                {videoData.r2_urls.initial_video}
                              </a>
                            </div>
                          )}
                          {videoData.r2_urls.extension_videos && videoData.r2_urls.extension_videos.map((url: string, index: number) => (
                            <div key={index} className="url-item">
                              <strong>Extension {index + 1}:</strong>
                              <a href={url} target="_blank" rel="noopener noreferrer">
                                {url}
                              </a>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {showUploadForm && recordedVideoBlob && (
        <VideoUploadForm
          videoBlob={recordedVideoBlob}
          onUpload={handleUploadFormSubmit}
          onCancel={handleUploadCancel}
          isUploading={isUploading}
          uploadProgress={uploadProgress}
        />
      )}
    </div>
  );
}