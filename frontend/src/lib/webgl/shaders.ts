// WebGL Shader utilities for audio visualization distortion effects

export interface ShaderProgram {
  program: WebGLProgram;
  uniforms: { [key: string]: WebGLUniformLocation | null };
  attributes: { [key: string]: number };
}

export class ShaderManager {
  private gl: WebGLRenderingContext;
  private programs: Map<string, ShaderProgram> = new Map();

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }

  // Compile a shader from source
  private compileShader(source: string, type: number): WebGLShader | null {
    const shader = this.gl.createShader(type);
    if (!shader) return null;

    this.gl.shaderSource(shader, source);
    this.gl.compileShader(shader);

    if (!this.gl.getShaderParameter(shader, this.gl.COMPILE_STATUS)) {
      const error = this.gl.getShaderInfoLog(shader);
      console.error('Shader compilation error:', error);
      console.error('Shader source:', source);
      this.gl.deleteShader(shader);
      return null;
    }

    return shader;
  }

  // Create a shader program from vertex and fragment shader sources
  createProgram(
    name: string,
    vertexSource: string,
    fragmentSource: string,
    uniforms: string[] = [],
    attributes: string[] = []
  ): ShaderProgram | null {
    const vertexShader = this.compileShader(vertexSource, this.gl.VERTEX_SHADER);
    const fragmentShader = this.compileShader(fragmentSource, this.gl.FRAGMENT_SHADER);

    if (!vertexShader || !fragmentShader) {
      return null;
    }

    const program = this.gl.createProgram();
    if (!program) return null;

    this.gl.attachShader(program, vertexShader);
    this.gl.attachShader(program, fragmentShader);
    this.gl.linkProgram(program);

    if (!this.gl.getProgramParameter(program, this.gl.LINK_STATUS)) {
      const error = this.gl.getProgramInfoLog(program);
      console.error('Shader program linking error:', error);
      this.gl.deleteProgram(program);
      return null;
    }

    // Get uniform and attribute locations
    const uniformLocations: { [key: string]: WebGLUniformLocation | null } = {};
    const attributeLocations: { [key: string]: number } = {};

    uniforms.forEach(uniformName => {
      uniformLocations[uniformName] = this.gl.getUniformLocation(program, uniformName);
    });

    attributes.forEach(attributeName => {
      attributeLocations[attributeName] = this.gl.getAttribLocation(program, attributeName);
    });

    const shaderProgram: ShaderProgram = {
      program,
      uniforms: uniformLocations,
      attributes: attributeLocations
    };

    this.programs.set(name, shaderProgram);
    return shaderProgram;
  }

  // Get a previously created program
  getProgram(name: string): ShaderProgram | null {
    return this.programs.get(name) || null;
  }

  // Use a shader program
  useProgram(name: string): ShaderProgram | null {
    const program = this.programs.get(name);
    if (program) {
      this.gl.useProgram(program.program);
      return program;
    }
    return null;
  }

  // Clean up all programs
  dispose(): void {
    this.programs.forEach(program => {
      this.gl.deleteProgram(program.program);
    });
    this.programs.clear();
  }
}

// Standard vertex shader for fullscreen quad
export const FULLSCREEN_VERTEX_SHADER = `
  attribute vec2 a_position;
  attribute vec2 a_texCoord;
  
  varying vec2 v_texCoord;
  
  void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texCoord = a_texCoord;
  }
`;

// Circle mask generation fragment shader
export const CIRCLE_MASK_FRAGMENT_SHADER = `
  precision mediump float;
  
  uniform vec2 u_resolution;
  uniform vec2 u_center;
  uniform float u_radius;
  uniform float u_time;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec2 coord = gl_FragCoord.xy;
    vec2 center = u_center;
    
    float distance = length(coord - center);
    float circle = smoothstep(u_radius + 1.0, u_radius - 1.0, distance);
    
    gl_FragColor = vec4(circle, circle, circle, 1.0);
  }
`;

// Distortion fragment shader with multiple effect types
export const DISTORTION_FRAGMENT_SHADER = `
  precision mediump float;
  
  uniform sampler2D u_texture;
  uniform vec2 u_resolution;
  uniform float u_time;
  uniform float u_distortionIntensity;
  uniform float u_distortionScale;
  uniform float u_distortionSpeed;
  uniform int u_distortionType; // 0: wave, 1: ripple, 2: noise, 3: bulge
  uniform float u_audioIntensity;
  
  varying vec2 v_texCoord;
  
  // Noise function for distortion
  float random(vec2 st) {
    return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453123);
  }
  
  float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);
    
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));
    
    vec2 u = f * f * (3.0 - 2.0 * f);
    
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
  }
  
  vec2 getDistortedCoord(vec2 coord) {
    vec2 center = u_resolution * 0.5;
    vec2 offset = coord - center;
    float intensity = u_distortionIntensity * (1.0 + u_audioIntensity * 2.0);
    float time = u_time * u_distortionSpeed;
    
    if (u_distortionType == 0) {
      // Wave distortion
      float wave = sin(coord.x * u_distortionScale + time) * 
                   cos(coord.y * u_distortionScale + time);
      return coord + vec2(wave * intensity, wave * intensity * 0.5);
      
    } else if (u_distortionType == 1) {
      // Ripple distortion
      float distance = length(offset);
      float ripple = sin(distance * u_distortionScale - time * 3.0) / distance;
      return coord + normalize(offset) * ripple * intensity;
      
    } else if (u_distortionType == 2) {
      // Noise distortion
      float noiseX = noise(coord * u_distortionScale + time);
      float noiseY = noise(coord * u_distortionScale + time + 100.0);
      return coord + vec2(noiseX - 0.5, noiseY - 0.5) * intensity;
      
    } else if (u_distortionType == 3) {
      // Bulge distortion
      float distance = length(offset);
      float bulge = 1.0 + intensity * exp(-distance * u_distortionScale);
      return center + offset * bulge;
    }
    
    return coord;
  }
  
  void main() {
    vec2 distortedCoord = getDistortedCoord(gl_FragCoord.xy);
    vec2 uv = distortedCoord / u_resolution;
    
    // Sample with bounds checking
    if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
      gl_FragColor = texture2D(u_texture, uv);
    } else {
      gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    }
  }
`;

// Feedback/trails fragment shader
export const FEEDBACK_FRAGMENT_SHADER = `
  precision mediump float;
  
  uniform sampler2D u_currentTexture;
  uniform sampler2D u_previousTexture;
  uniform float u_feedbackAmount;
  
  varying vec2 v_texCoord;
  
  void main() {
    vec4 current = texture2D(u_currentTexture, v_texCoord);
    vec4 previous = texture2D(u_previousTexture, v_texCoord);
    
    // Blend current frame with faded previous frame
    gl_FragColor = current + previous * u_feedbackAmount;
  }
`;