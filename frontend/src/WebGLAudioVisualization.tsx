import { useRef, useEffect, useState } from 'react';
import VideoUploadForm, { type UploadFormData } from './VideoUploadForm';
import type { RMSAnalysisResult } from './lib/audio/analyzer';
import type { AudioPlayer } from './lib/audio/player';
import { WebGLRenderer, type DistortionParams, type RenderParams } from './lib/webgl/renderer';
import { BACKEND_URL } from './constants';

interface WebGLAudioVisualizationProps {
  analysisResult: RMSAnalysisResult | null;
  audioPlayer: AudioPlayer | null;
  isPlaying: boolean;
}

export default function WebGLAudioVisualization({
  analysisResult,
  audioPlayer,
  isPlaying
}: WebGLAudioVisualizationProps) {
  // Canvas and rendering
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<WebGLRenderer | null>(null);
  const animationRef = useRef<number>(0);
  const startTimeRef = useRef<number>(0);
  const intensityHistoryRef = useRef<number[]>([]);
  const lastFrameTimeRef = useRef<number>(0);
  const recordingModeRef = useRef<boolean>(false);
  const recordingIntervalRef = useRef<number>(0);
  const recordingCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const radiusHistoryRef = useRef<number[]>([]);

  // Basic parameters
  const width = 720;
  const height = 720;
  const [minRadius, setMinRadius] = useState(10);
  const [maxRadius, setMaxRadius] = useState(150);
  const [feedback, setFeedback] = useState(0.1);
  const [smoothWindow, setSmoothWindow] = useState(3);
  const [isEnabled, setIsEnabled] = useState(true);

  // Distortion parameters
  const [distortionType, setDistortionType] = useState(0); // 0: wave, 1: ripple, 2: noise, 3: bulge
  const [distortionIntensity, setDistortionIntensity] = useState(5.0);
  const [distortionScale, setDistortionScale] = useState(0.02);
  const [distortionSpeed, setDistortionSpeed] = useState(1.0);
  const [audioReactivity, setAudioReactivity] = useState(1.0);

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
    // Load active WebGL jobs from localStorage on component mount
    try {
      const saved = localStorage.getItem('activeWebGLJobs');
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

  // Initialize WebGL renderer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    try {
      const renderer = new WebGLRenderer(canvas);
      if (renderer.initialize(width, height)) {
        rendererRef.current = renderer;
        startTimeRef.current = Date.now();
      } else {
        console.error('Failed to initialize WebGL renderer');
      }
    } catch (error) {
      console.error('WebGL not supported:', error);
    }

    return () => {
      if (rendererRef.current) {
        rendererRef.current.dispose();
        rendererRef.current = null;
      }
    };
  }, []);

  // Handle canvas resize
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rendererRef.current) return;

    canvas.width = width;
    canvas.height = height;
    rendererRef.current.resize(width, height);
  }, [width, height]);


  // Get current intensity with interpolation between segments
  const getCurrentIntensity = (): number => {
    if (!analysisResult || !audioPlayer || !isPlaying) {
      return 0;
    }

    const currentTime = audioPlayer.getCurrentTime();
    const segments = analysisResult.rmsData.segments;

    if (segments.length === 0) return 0;

    // Find current and next segments for interpolation
    let currentIndex = -1;
    for (let i = 0; i < segments.length; i++) {
      if (currentTime >= segments[i].startTime && currentTime <= segments[i].endTime) {
        currentIndex = i;
        break;
      }
    }

    // If we're past all segments, use the last one
    if (currentIndex === -1) {
      if (currentTime > segments[segments.length - 1].endTime) {
        return segments[segments.length - 1].rms;
      }
      // If we're before all segments, use the first one
      return segments[0].rms;
    }

    const currentSegment = segments[currentIndex];
    const nextSegment = segments[currentIndex + 1];

    // If no next segment, just return current value
    if (!nextSegment) {
      return currentSegment.rms;
    }

    // Calculate interpolation factor based on position within current segment
    const segmentDuration = currentSegment.endTime - currentSegment.startTime;
    const timeInSegment = currentTime - currentSegment.startTime;
    const progress = Math.min(timeInSegment / segmentDuration, 1);

    // Apply much smoother interpolation - use cubic ease-in-out for very smooth transitions
    const smoothProgress = progress < 0.5
      ? 4 * progress * progress * progress
      : 1 - Math.pow(-2 * progress + 2, 3) / 2;

    // Interpolate between current and next segment values
    const interpolatedValue = currentSegment.rms + (nextSegment.rms - currentSegment.rms) * smoothProgress;

    // Apply much more aggressive smoothing using larger history window
    intensityHistoryRef.current.push(interpolatedValue);

    // Use much larger smoothing window for ultra-smooth results
    const extendedSmoothingWindow = Math.max(smoothWindow * 3, 10);
    if (intensityHistoryRef.current.length > extendedSmoothingWindow) {
      intensityHistoryRef.current = intensityHistoryRef.current.slice(-extendedSmoothingWindow);
    }

    // Apply exponential smoothing with much stronger weighting toward recent values
    const weights = intensityHistoryRef.current.map((_, i, arr) => Math.pow(1.5, i - arr.length + 1));
    const weightedSum = intensityHistoryRef.current.reduce((sum, val, i) => sum + val * weights[i], 0);
    const totalWeight = weights.reduce((sum, w) => sum + w, 0);

    // Apply additional low-pass filtering
    const smoothedResult = weightedSum / totalWeight;

    // Final smoothing pass using simple moving average
    if (intensityHistoryRef.current.length >= 3) {
      const lastThree = intensityHistoryRef.current.slice(-3);
      const avgOfRecent = lastThree.reduce((sum, val) => sum + val, 0) / lastThree.length;
      return (smoothedResult + avgOfRecent) / 2;
    }

    return smoothedResult;
  };

  // Map intensity to radius with heavy smoothing
  const intensityToRadius = (intensity: number): number => {
    if (!analysisResult) return minRadius;

    const maxIntensity = analysisResult.rmsData.peak;
    if (maxIntensity === 0) return minRadius;

    const normalizedIntensity = Math.min(intensity / maxIntensity, 1);
    const rawRadius = minRadius + (normalizedIntensity * (maxRadius - minRadius));

    // Apply heavy smoothing to radius changes
    radiusHistoryRef.current.push(rawRadius);

    // Keep a large history for ultra-smooth radius changes
    const radiusSmoothingWindow = 20;
    if (radiusHistoryRef.current.length > radiusSmoothingWindow) {
      radiusHistoryRef.current = radiusHistoryRef.current.slice(-radiusSmoothingWindow);
    }

    // Apply exponential moving average for very smooth radius transitions
    if (radiusHistoryRef.current.length === 1) {
      return rawRadius;
    }

    // Heavy weighting toward previous values for very smooth changes
    const alpha = 0.15; // Low alpha = more smoothing
    let smoothedRadius = radiusHistoryRef.current[0];
    for (let i = 1; i < radiusHistoryRef.current.length; i++) {
      smoothedRadius = alpha * radiusHistoryRef.current[i] + (1 - alpha) * smoothedRadius;
    }

    return smoothedRadius;
  };

  // Render a single frame
  const renderFrame = () => {
    if (!rendererRef.current || !isEnabled) return;

    const currentTime = (Date.now() - startTimeRef.current) / 1000;
    const currentIntensity = getCurrentIntensity();
    const radius = intensityToRadius(currentIntensity);

    // Normalize audio intensity for shader
    const normalizedAudioIntensity = analysisResult
      ? Math.min(currentIntensity / analysisResult.rmsData.peak, 1) * audioReactivity
      : 0;

    const distortionParams: DistortionParams = {
      type: distortionType,
      intensity: distortionIntensity,
      scale: distortionScale,
      speed: distortionSpeed
    };

    const renderParams: RenderParams = {
      centerX: width / 2,
      centerY: height / 2,
      radius: radius,
      audioIntensity: normalizedAudioIntensity,
      feedback: feedback,
      distortion: distortionParams,
      time: currentTime
    };

    rendererRef.current.render(renderParams);
  };

  // Animation loop at full refresh rate for maximum smoothness
  const animate = () => {
    if (!isPlaying || !isEnabled) return;

    renderFrame();
    animationRef.current = requestAnimationFrame(animate);
  };

  // Start/stop animation based on playback state
  useEffect(() => {
    if (isPlaying && isEnabled && analysisResult && rendererRef.current) {
      intensityHistoryRef.current = []; // Reset intensity history
      radiusHistoryRef.current = []; // Reset radius history
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      // Render one final frame when stopped
      if (!isPlaying && analysisResult && rendererRef.current) {
        renderFrame();
      }
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, isEnabled, analysisResult, audioPlayer, distortionType, distortionIntensity, distortionScale, distortionSpeed, audioReactivity, feedback]);

  // Video recording functions (same as original)
  const startRecording = async () => {
    const canvas = canvasRef.current;
    if (!canvas || !audioPlayer || !analysisResult) return;

    try {
      // Get canvas stream at 30fps
      const stream = canvas.captureStream(30);

      const audioElement = (audioPlayer as any).audioElement;
      if (audioElement && audioElement.captureStream) {
        const audioStream = audioElement.captureStream();
        audioStream.getAudioTracks().forEach((track: MediaStreamTrack) => {
          stream.addTrack(track);
        });
      }

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

        if (recordedChunksRef.current.length > 0) {
          const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
          setRecordedVideoBlob(blob);
          setShowUploadForm(true);
        }
      };

      mediaRecorderRef.current.start(100);
      recordingModeRef.current = true; // Enable recording mode
      setIsRecording(true);
      recordingStartTimeRef.current = Date.now();


      const progressInterval = setInterval(() => {
        if (!isRecording) {
          clearInterval(progressInterval);
          return;
        }

        const elapsed = (Date.now() - recordingStartTimeRef.current) / 1000;
        const progress = (elapsed / analysisResult.duration) * 100;
        setRecordingProgress(Math.min(progress, 100));

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
      recordingModeRef.current = false; // Disable recording mode
      mediaRecorderRef.current.stop();
    }
  };

  const downloadVideo = () => {
    if (!recordedVideoBlob) return;

    const url = URL.createObjectURL(recordedVideoBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `webgl_audio_visualization_${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Upload and job polling functions (same as original)
  const handleUploadFormSubmit = async (formData: UploadFormData) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      const serverFormData = new FormData();

      serverFormData.append('video_file', formData.videoBlob, `webgl_visualization_${Date.now()}.webm`);
      if (formData.imageFile) {
        serverFormData.append('image_file', formData.imageFile);
      }

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
        console.log(`WebGL Upload successful!`, result);

        const jobId = result.sieve_job_id;
        if (jobId) {
          console.log(`Starting job polling for Sieve Job ID: ${jobId}`);
          addJobToPolling(jobId, result.session_id, result.video_filename);
        } else {
          console.warn(`No valid Sieve Job ID found.`);
        }

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

  // Job polling functions (same as original)
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
      localStorage.setItem('activeWebGLJobs', JSON.stringify(updatedJobs));
      return updatedJobs;
    });

    if (newStatus === 'finished') {
      // Find the job to get additional info
      const job = activeJobs.find(j => j.jobId === jobId);
      if (job) {
        // Add to completed jobs list with full result data
        console.log('Adding WebGL completed job with result:', { jobId, result });
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

      const videoData = result.result || (result.job_data?.outputs?.[0]?.data);

      if (videoData?.combined_video?.url) {
        videoUrls.push(`Combined Video: ${videoData.combined_video.url}`);
      }
      if (videoData?.r2_urls?.combined_video) {
        videoUrls.push(`R2 Combined: ${videoData.r2_urls.combined_video}`);
      }

      console.log(`WebGL Job completed successfully! Job ID: ${jobId}`, { result, videoData });
      removeCompletedJob(jobId);
    } else if (newStatus === 'failed') {
      alert(`WebGL Job failed!\nJob ID: ${jobId}`);
      removeCompletedJob(jobId);
    }
  };

  const removeCompletedJob = (jobId: string) => {
    setActiveJobs(prevJobs => {
      const updatedJobs = prevJobs.filter(job => job.jobId !== jobId);
      // Update localStorage
      localStorage.setItem('activeWebGLJobs', JSON.stringify(updatedJobs));
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
      localStorage.setItem('activeWebGLJobs', JSON.stringify(updatedJobs));
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
            console.log(`Polling WebGL job ${job.jobId}:`, result);
            if (result && result.status !== job.status) {
              console.log('Full polling result structure:', JSON.stringify(result, null, 2));
              console.log('result.result:', result.result);
              console.log('result.job_data:', result.job_data);
              console.log('result.job_data?.outputs?.[0]?.data:', result.job_data?.outputs?.[0]?.data);
              updateJobStatus(job.jobId, result.status, result);
            }
          }
        }
      }, 3000);

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

  // Get distortion type name for display
  const getDistortionTypeName = (type: number): string => {
    switch (type) {
      case 0: return 'Wave';
      case 1: return 'Ripple';
      case 2: return 'Noise';
      case 3: return 'Bulge';
      default: return 'Unknown';
    }
  };

  return (
    <div className="audio-visualization">
      <div className="visualization-header">
        <span>WebGL Audio Visualization</span>
        <div className="header-controls">
          <label className="visualization-toggle">
            <input
              type="checkbox"
              checked={isEnabled}
              onChange={(e) => setIsEnabled(e.target.checked)}
            />
            Enable
          </label>

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

        {/* Distortion Controls */}
        <div className="control-group">
          <label>Distortion: {getDistortionTypeName(distortionType)}</label>
          <input
            type="range"
            min="0"
            max="3"
            value={distortionType}
            onChange={(e) => setDistortionType(parseInt(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Intensity: {distortionIntensity.toFixed(1)}</label>
          <input
            type="range"
            min="0"
            max="20"
            step="0.5"
            value={distortionIntensity}
            onChange={(e) => setDistortionIntensity(parseFloat(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Scale: {distortionScale.toFixed(3)}</label>
          <input
            type="range"
            min="0.001"
            max="0.1"
            step="0.001"
            value={distortionScale}
            onChange={(e) => setDistortionScale(parseFloat(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Speed: {distortionSpeed.toFixed(1)}x</label>
          <input
            type="range"
            min="0.1"
            max="5"
            step="0.1"
            value={distortionSpeed}
            onChange={(e) => setDistortionSpeed(parseFloat(e.target.value))}
          />
        </div>

        <div className="control-group">
          <label>Audio Reactivity: {audioReactivity.toFixed(1)}x</label>
          <input
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={audioReactivity}
            onChange={(e) => setAudioReactivity(parseFloat(e.target.value))}
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
              Record WebGL Video
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
        <div>
          Distortion: {getDistortionTypeName(distortionType)} ({distortionIntensity.toFixed(1)})
        </div>
      </div>

      {activeJobs.length > 0 && (
        <div className="job-status-section">
          <div className="job-status-header">WebGL Processing Jobs</div>
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
            WebGL Completed Jobs
            <button
              onClick={() => setCompletedJobs([])}
              className="clear-completed-button"
              style={{ marginLeft: '10px', fontSize: '12px', padding: '4px 8px' }}
            >
              Clear All
            </button>
          </div>
          {completedJobs.map((job) => {
            // Extract video data from result
            console.log('Rendering WebGL completed job:', { jobId: job.jobId, result: job.result });
            const videoData = job.result.result || (job.result.job_data?.outputs?.[0]?.data);
            console.log('Extracted WebGL videoData:', videoData);

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