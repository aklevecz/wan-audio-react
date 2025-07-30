// WebGL Renderer for distorted audio visualization

import { ShaderManager, type ShaderProgram, FULLSCREEN_VERTEX_SHADER, CIRCLE_MASK_FRAGMENT_SHADER, DISTORTION_FRAGMENT_SHADER, FEEDBACK_FRAGMENT_SHADER } from './shaders';
import { TextureManager } from './textures';

export interface DistortionParams {
  type: number; // 0: wave, 1: ripple, 2: noise, 3: bulge
  intensity: number;
  scale: number;
  speed: number;
}

export interface RenderParams {
  centerX: number;
  centerY: number;
  radius: number;
  audioIntensity: number;
  feedback: number;
  distortion: DistortionParams;
  time: number;
}

export class WebGLRenderer {
  private gl: WebGLRenderingContext;
  private shaderManager: ShaderManager;
  private textureManager: TextureManager;
  private quadBuffer: WebGLBuffer | null = null;
  private initialized = false;
  private width = 0;
  private height = 0;

  constructor(canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      throw new Error('WebGL not supported');
    }

    this.gl = gl as WebGLRenderingContext;
    this.shaderManager = new ShaderManager(this.gl);
    this.textureManager = new TextureManager(this.gl);
  }

  // Initialize WebGL resources
  initialize(width: number, height: number): boolean {
    if (this.initialized) {
      this.resize(width, height);
      return true;
    }

    this.width = width;
    this.height = height;

    // Create fullscreen quad buffer
    this.createQuadBuffer();

    // Create shader programs
    if (!this.createShaderPrograms()) {
      console.error('Failed to create shader programs');
      return false;
    }

    // Create framebuffers for multi-pass rendering
    if (!this.createFramebuffers()) {
      console.error('Failed to create framebuffers');
      return false;
    }

    // Set initial WebGL state
    this.gl.disable(this.gl.DEPTH_TEST);
    this.gl.disable(this.gl.CULL_FACE);
    this.gl.enable(this.gl.BLEND);
    this.gl.blendFunc(this.gl.SRC_ALPHA, this.gl.ONE_MINUS_SRC_ALPHA);

    this.initialized = true;
    return true;
  }

  // Create fullscreen quad vertex buffer
  private createQuadBuffer(): void {
    const quadVertices = new Float32Array([
      // Position (x, y) and TexCoord (u, v)
      -1.0, -1.0,  0.0, 0.0,  // Bottom left
       1.0, -1.0,  1.0, 0.0,  // Bottom right
      -1.0,  1.0,  0.0, 1.0,  // Top left
       1.0,  1.0,  1.0, 1.0   // Top right
    ]);

    this.quadBuffer = this.gl.createBuffer();
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);
    this.gl.bufferData(this.gl.ARRAY_BUFFER, quadVertices, this.gl.STATIC_DRAW);
  }

  // Create all shader programs
  private createShaderPrograms(): boolean {
    // Circle mask shader
    const maskProgram = this.shaderManager.createProgram(
      'mask',
      FULLSCREEN_VERTEX_SHADER,
      CIRCLE_MASK_FRAGMENT_SHADER,
      ['u_resolution', 'u_center', 'u_radius', 'u_time'],
      ['a_position', 'a_texCoord']
    );

    // Distortion shader
    const distortionProgram = this.shaderManager.createProgram(
      'distortion',
      FULLSCREEN_VERTEX_SHADER,
      DISTORTION_FRAGMENT_SHADER,
      ['u_texture', 'u_resolution', 'u_time', 'u_distortionIntensity', 'u_distortionScale', 'u_distortionSpeed', 'u_distortionType', 'u_audioIntensity'],
      ['a_position', 'a_texCoord']
    );

    // Feedback shader
    const feedbackProgram = this.shaderManager.createProgram(
      'feedback',
      FULLSCREEN_VERTEX_SHADER,
      FEEDBACK_FRAGMENT_SHADER,
      ['u_currentTexture', 'u_previousTexture', 'u_feedbackAmount'],
      ['a_position', 'a_texCoord']
    );

    return maskProgram !== null && distortionProgram !== null && feedbackProgram !== null;
  }

  // Create framebuffers for multi-pass rendering
  private createFramebuffers(): boolean {
    // Create framebuffers for different render passes
    const maskFB = this.textureManager.createFramebuffer('mask', this.width, this.height);
    const distortionFB = this.textureManager.createFramebuffer('distortion', this.width, this.height);
    
    // Create ping-pong framebuffers for feedback effect
    const pingPong = this.textureManager.createPingPongFramebuffers('feedback', this.width, this.height);

    return maskFB !== null && distortionFB !== null && pingPong;
  }

  // Resize renderer
  resize(width: number, height: number): void {
    if (this.width === width && this.height === height) return;

    this.width = width;
    this.height = height;

    // Resize framebuffers
    this.textureManager.resizeFramebuffer('mask', width, height);
    this.textureManager.resizeFramebuffer('distortion', width, height);
    this.textureManager.resizeFramebuffer('feedback_1', width, height);
    this.textureManager.resizeFramebuffer('feedback_2', width, height);
  }

  // Setup vertex attributes for fullscreen quad
  private setupQuadAttributes(program: ShaderProgram): void {
    if (!this.quadBuffer) return;

    this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.quadBuffer);

    // Position attribute
    if (program.attributes.a_position >= 0) {
      this.gl.enableVertexAttribArray(program.attributes.a_position);
      this.gl.vertexAttribPointer(program.attributes.a_position, 2, this.gl.FLOAT, false, 16, 0);
    }

    // Texture coordinate attribute
    if (program.attributes.a_texCoord >= 0) {
      this.gl.enableVertexAttribArray(program.attributes.a_texCoord);
      this.gl.vertexAttribPointer(program.attributes.a_texCoord, 2, this.gl.FLOAT, false, 16, 8);
    }
  }

  // Draw fullscreen quad
  private drawQuad(): void {
    this.gl.drawArrays(this.gl.TRIANGLE_STRIP, 0, 4);
  }

  // Pass 1: Render circle mask
  private renderMask(params: RenderParams): void {
    const program = this.shaderManager.useProgram('mask');
    if (!program) return;

    // Bind mask framebuffer
    this.textureManager.bindFramebuffer('mask');
    
    // Clear framebuffer
    this.gl.clearColor(0, 0, 0, 1);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    // Setup attributes
    this.setupQuadAttributes(program);

    // Set uniforms
    if (program.uniforms.u_resolution) {
      this.gl.uniform2f(program.uniforms.u_resolution, this.width, this.height);
    }
    if (program.uniforms.u_center) {
      this.gl.uniform2f(program.uniforms.u_center, params.centerX, this.height - params.centerY);
    }
    if (program.uniforms.u_radius) {
      this.gl.uniform1f(program.uniforms.u_radius, params.radius);
    }
    if (program.uniforms.u_time) {
      this.gl.uniform1f(program.uniforms.u_time, params.time);
    }

    this.drawQuad();
  }

  // Pass 2: Apply distortion
  private renderDistortion(params: RenderParams): void {
    const program = this.shaderManager.useProgram('distortion');
    if (!program) return;

    // Bind distortion framebuffer
    this.textureManager.bindFramebuffer('distortion');
    
    // Clear framebuffer
    this.gl.clearColor(0, 0, 0, 1);
    this.gl.clear(this.gl.COLOR_BUFFER_BIT);

    // Bind mask texture
    this.textureManager.bindTexture('mask', 0);

    // Setup attributes
    this.setupQuadAttributes(program);

    // Set uniforms
    if (program.uniforms.u_texture) {
      this.gl.uniform1i(program.uniforms.u_texture, 0);
    }
    if (program.uniforms.u_resolution) {
      this.gl.uniform2f(program.uniforms.u_resolution, this.width, this.height);
    }
    if (program.uniforms.u_time) {
      this.gl.uniform1f(program.uniforms.u_time, params.time);
    }
    if (program.uniforms.u_distortionIntensity) {
      this.gl.uniform1f(program.uniforms.u_distortionIntensity, params.distortion.intensity);
    }
    if (program.uniforms.u_distortionScale) {
      this.gl.uniform1f(program.uniforms.u_distortionScale, params.distortion.scale);
    }
    if (program.uniforms.u_distortionSpeed) {
      this.gl.uniform1f(program.uniforms.u_distortionSpeed, params.distortion.speed);
    }
    if (program.uniforms.u_distortionType) {
      this.gl.uniform1i(program.uniforms.u_distortionType, params.distortion.type);
    }
    if (program.uniforms.u_audioIntensity) {
      this.gl.uniform1f(program.uniforms.u_audioIntensity, params.audioIntensity);
    }

    this.drawQuad();
  }

  // Pass 3: Apply feedback and render to screen
  private renderFeedback(params: RenderParams): void {
    const program = this.shaderManager.useProgram('feedback');
    if (!program) return;

    // Bind screen framebuffer
    this.textureManager.bindDefaultFramebuffer(this.width, this.height);

    // Bind textures
    this.textureManager.bindTexture('distortion', 0); // Current frame
    this.textureManager.bindTexture('feedback_1', 1); // Previous frame

    // Setup attributes
    this.setupQuadAttributes(program);

    // Set uniforms
    if (program.uniforms.u_currentTexture) {
      this.gl.uniform1i(program.uniforms.u_currentTexture, 0);
    }
    if (program.uniforms.u_previousTexture) {
      this.gl.uniform1i(program.uniforms.u_previousTexture, 1);
    }
    if (program.uniforms.u_feedbackAmount) {
      this.gl.uniform1f(program.uniforms.u_feedbackAmount, params.feedback);
    }

    this.drawQuad();

    // Copy current result to feedback buffer for next frame
    this.copyToFeedbackBuffer();
  }

  // Copy current screen to feedback buffer
  private copyToFeedbackBuffer(): void {
    // This would ideally copy the screen to feedback buffer
    // For now, we'll swap the ping-pong buffers
    this.textureManager.swapPingPongFramebuffers('feedback');
  }

  // Main render function
  render(params: RenderParams): void {
    if (!this.initialized) return;

    // Multi-pass rendering
    this.renderMask(params);      // Pass 1: Generate circle mask
    this.renderDistortion(params); // Pass 2: Apply distortion
    this.renderFeedback(params);   // Pass 3: Apply feedback and render to screen
  }

  // Check if WebGL is supported
  static isSupported(): boolean {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
      return gl !== null;
    } catch (e) {
      return false;
    }
  }

  // Clean up resources
  dispose(): void {
    this.shaderManager.dispose();
    this.textureManager.dispose();
    
    if (this.quadBuffer) {
      this.gl.deleteBuffer(this.quadBuffer);
      this.quadBuffer = null;
    }

    this.initialized = false;
  }
}