export interface AudioSegment {
	startTime: number;
	endTime: number;
}

export class AudioPlayer {
	private audioElement: HTMLAudioElement | null = null;
	private audioBuffer: AudioBuffer | null = null;
	private audioContext: AudioContext | null = null;
	private source: AudioBufferSourceNode | null = null;
	private isPlaying = false;
	private currentSegment: AudioSegment | null = null;

	async loadAudioFile(file: File): Promise<void> {
		try {
			// Create audio element for playback
			this.audioElement = new Audio();
			this.audioElement.src = URL.createObjectURL(file);
			
			// Also decode for more precise control
			this.audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
			const arrayBuffer = await file.arrayBuffer();
			this.audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);
		} catch (error) {
			throw new Error(`Failed to load audio: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	async playSegment(segment: AudioSegment): Promise<void> {
		if (!this.audioElement) {
			throw new Error('No audio file loaded');
		}

		try {
			this.currentSegment = segment;
			this.audioElement.currentTime = segment.startTime;
			
			// Set up event listener to stop at end time
			const handleTimeUpdate = () => {
				if (this.audioElement && this.audioElement.currentTime >= segment.endTime) {
					this.stop();
					this.audioElement.removeEventListener('timeupdate', handleTimeUpdate);
				}
			};

			this.audioElement.addEventListener('timeupdate', handleTimeUpdate);
			await this.audioElement.play();
			this.isPlaying = true;
		} catch (error) {
			throw new Error(`Failed to play segment: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	async playFull(): Promise<void> {
		if (!this.audioElement) {
			throw new Error('No audio file loaded');
		}

		try {
			this.currentSegment = null;
			this.audioElement.currentTime = 0;
			await this.audioElement.play();
			this.isPlaying = true;
		} catch (error) {
			throw new Error(`Failed to play audio: ${error instanceof Error ? error.message : 'Unknown error'}`);
		}
	}

	pause(): void {
		if (this.audioElement) {
			this.audioElement.pause();
			this.isPlaying = false;
		}
	}

	stop(): void {
		if (this.audioElement) {
			this.audioElement.pause();
			this.audioElement.currentTime = 0;
			this.isPlaying = false;
		}
	}

	getCurrentTime(): number {
		return this.audioElement?.currentTime || 0;
	}

	getDuration(): number {
		return this.audioElement?.duration || 0;
	}

	getIsPlaying(): boolean {
		return this.isPlaying;
	}

	getCurrentSegment(): AudioSegment | null {
		return this.currentSegment;
	}

	dispose(): void {
		this.stop();
		if (this.audioElement) {
			URL.revokeObjectURL(this.audioElement.src);
			this.audioElement = null;
		}
		if (this.audioContext && this.audioContext.state !== 'closed') {
			this.audioContext.close();
		}
	}
}