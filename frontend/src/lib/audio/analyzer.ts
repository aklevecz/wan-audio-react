export interface RMSAnalysisResult {
	filename: string;
	duration: number;
	sampleRate: number;
	channels: number;
	rmsData: {
		overall: number;
		peak: number;
		segments: Array<{
			startTime: number;
			endTime: number;
			rms: number;
		}>;
	};
	waveformData: number[];
	analyzedAt: string;
}

export class AudioAnalyzer {
	private audioContext: AudioContext;

	constructor() {
		this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
	}

	async analyzeRMS(file: File, segmentDuration: number = 0.1): Promise<RMSAnalysisResult> {
		try {
			// Decode audio file
			const arrayBuffer = await file.arrayBuffer();
			const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

			// Calculate RMS for each segment
			const segments = this.calculateRMSSegments(audioBuffer, segmentDuration);
			
			// Calculate overall and peak RMS
			const rmsValues = segments.map(s => s.rms);
			const overallRMS = Math.sqrt(rmsValues.reduce((sum, rms) => sum + rms * rms, 0) / rmsValues.length);
			const peakRMS = Math.max(...rmsValues);

			// Generate waveform data (downsample for visualization)
			const waveformData = this.generateWaveformData(audioBuffer, 800);

			return {
				filename: file.name,
				duration: audioBuffer.duration,
				sampleRate: audioBuffer.sampleRate,
				channels: audioBuffer.numberOfChannels,
				rmsData: {
					overall: overallRMS,
					peak: peakRMS,
					segments
				},
				waveformData,
				analyzedAt: new Date().toISOString()
			};
		} catch (error) {
			throw new Error(`Failed to analyze audio: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	private calculateRMSSegments(audioBuffer: AudioBuffer, segmentDuration: number) {
		const sampleRate = audioBuffer.sampleRate;
		const segmentSamples = Math.floor(segmentDuration * sampleRate);
		const totalSamples = audioBuffer.length;
		const segments = [];

		for (let start = 0; start < totalSamples; start += segmentSamples) {
			const end = Math.min(start + segmentSamples, totalSamples);
			const startTime = start / sampleRate;
			const endTime = end / sampleRate;
			
			let sumSquares = 0;
			let sampleCount = 0;

			// Process all channels
			for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
				const channelData = audioBuffer.getChannelData(channel);
				
				for (let i = start; i < end; i++) {
					sumSquares += channelData[i] * channelData[i];
					sampleCount++;
				}
			}

			const rms = Math.sqrt(sumSquares / sampleCount);
			
			segments.push({
				startTime,
				endTime,
				rms
			});
		}

		return segments;
	}

	private generateWaveformData(audioBuffer: AudioBuffer, targetPoints: number): number[] {
		const channelData = audioBuffer.getChannelData(0); // Use first channel
		const samplesPerPoint = Math.floor(audioBuffer.length / targetPoints);
		const waveformData: number[] = [];

		for (let i = 0; i < targetPoints; i++) {
			const startSample = i * samplesPerPoint;
			const endSample = Math.min(startSample + samplesPerPoint, audioBuffer.length);
			
			let max = 0;
			for (let j = startSample; j < endSample; j++) {
				max = Math.max(max, Math.abs(channelData[j]));
			}
			
			waveformData.push(max);
		}

		return waveformData;
	}

	dispose() {
		if (this.audioContext.state !== 'closed') {
			this.audioContext.close();
		}
	}
}