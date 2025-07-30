import { useState } from 'react';
import { BACKEND_URL } from './constants';

interface JobStatus {
  job_id: string;
  status: string;
  message: string;
  job_data?: any;
  result?: any;
  error?: string;
}

export default function JobPollingForm() {
  const [jobId, setJobId] = useState('');
  const [isPolling, setIsPolling] = useState(false);
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pollJob = async () => {
    if (!jobId.trim()) {
      setError('Please enter a job ID');
      return;
    }

    setIsPolling(true);
    setError(null);

    try {
      const response = await fetch(`${BACKEND_URL}/sieve/job/${jobId.trim()}`);
      
      if (response.ok) {
        const result = await response.json();
        console.log('Job polling result:', result);
        setJobStatus(result);
        setError(null);
      } else {
        const errorData = await response.json();
        setError(`Failed to poll job: ${errorData.detail || response.statusText}`);
        setJobStatus(null);
      }
    } catch (err) {
      setError(`Network error: ${err instanceof Error ? err.message : 'Unknown error'}`);
      setJobStatus(null);
    } finally {
      setIsPolling(false);
    }
  };

  const clearResults = () => {
    setJobStatus(null);
    setError(null);
  };

  const getStatusColor = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'queued': return '#0c5460';
      case 'processing': return '#856404';
      case 'finished': return '#155724';
      case 'failed': return '#721c24';
      default: return '#666';
    }
  };

  const getStatusBackground = (status: string) => {
    switch (status?.toLowerCase()) {
      case 'queued': return '#d1ecf1';
      case 'processing': return '#fff3cd';
      case 'finished': return '#d4edda';
      case 'failed': return '#f8d7da';
      default: return '#f8f9fa';
    }
  };

  return (
    <div className="job-polling-form">
      <div className="form-header">
        <h3>Manual Job Polling</h3>
      </div>

      <div className="polling-form">
        <div className="input-group">
          <label htmlFor="jobId">Sieve Job ID:</label>
          <input
            id="jobId"
            type="text"
            value={jobId}
            onChange={(e) => setJobId(e.target.value)}
            placeholder="e.g., 8538b6b9-8c5c-471d-9094-6490023692d8"
            className="job-id-input"
            disabled={isPolling}
          />
        </div>

        <div className="button-group">
          <button 
            onClick={pollJob} 
            disabled={isPolling || !jobId.trim()}
            className="poll-button"
          >
            {isPolling ? 'Polling...' : 'Poll Job Status'}
          </button>
          
          {(jobStatus || error) && (
            <button 
              onClick={clearResults}
              className="clear-button"
            >
              Clear Results
            </button>
          )}
        </div>

        {error && (
          <div className="error-message">
            <strong>Error:</strong> {error}
          </div>
        )}

        {jobStatus && (
          <div className="job-status-result">
            <div className="status-header">
              <span className="job-id-display">Job ID: {jobStatus.job_id}</span>
              <span 
                className="status-badge"
                style={{
                  backgroundColor: getStatusBackground(jobStatus.status),
                  color: getStatusColor(jobStatus.status)
                }}
              >
                {jobStatus.status.toUpperCase()}
              </span>
            </div>

            <div className="status-message">
              {jobStatus.message}
            </div>

            {jobStatus.job_data && (
              <div className="job-details">
                <h4>Job Details:</h4>
                <div className="detail-grid">
                  <div className="detail-item">
                    <span className="label">Function:</span>
                    <span className="value">{jobStatus.job_data.function?.name || 'N/A'}</span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Created:</span>
                    <span className="value">
                      {jobStatus.job_data.created_at ? 
                        new Date(jobStatus.job_data.created_at).toLocaleString() : 'N/A'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Started:</span>
                    <span className="value">
                      {jobStatus.job_data.started_at && jobStatus.job_data.started_at !== '1970-01-01T00:00:00' ? 
                        new Date(jobStatus.job_data.started_at).toLocaleString() : 'Not started'}
                    </span>
                  </div>
                  <div className="detail-item">
                    <span className="label">Completed:</span>
                    <span className="value">
                      {jobStatus.job_data.completed_at && jobStatus.job_data.completed_at !== '1970-01-01T00:00:00' ? 
                        new Date(jobStatus.job_data.completed_at).toLocaleString() : 'Not completed'}
                    </span>
                  </div>
                </div>

                {jobStatus.job_data.inputs && (
                  <div className="inputs-section">
                    <h5>Inputs:</h5>
                    <div className="inputs-grid">
                      {Object.entries(jobStatus.job_data.inputs).map(([key, input]: [string, any]) => (
                        <div key={key} className="input-item">
                          <span className="input-label">{key}:</span>
                          <span className="input-value">{String(input.data)}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

{(() => {
              // Extract video data from either result or job_data.outputs
              const videoData = jobStatus.result || 
                                (jobStatus.job_data?.outputs?.[0]?.data);
              
              return videoData && (
                <div className="job-result">
                  <h4>Result:</h4>
                  
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
                  
                  <details className="raw-result">
                    <summary>Raw JSON Result</summary>
                    <pre className="result-json">
                      {JSON.stringify(videoData, null, 2)}
                    </pre>
                  </details>
                </div>
              );
            })()}

            {jobStatus.error && (
              <div className="job-error">
                <h4>Error:</h4>
                <div className="error-content">{jobStatus.error}</div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}