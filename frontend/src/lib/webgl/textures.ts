// WebGL Texture management for audio visualization

export interface FramebufferInfo {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
}

export class TextureManager {
  private gl: WebGLRenderingContext;
  private textures: Map<string, WebGLTexture> = new Map();
  private framebuffers: Map<string, FramebufferInfo> = new Map();

  constructor(gl: WebGLRenderingContext) {
    this.gl = gl;
  }

  // Create a texture from image data or dimensions
  createTexture(
    name: string,
    width: number,
    height: number,
    data?: Uint8Array | null
  ): WebGLTexture | null {
    const texture = this.gl.createTexture();
    if (!texture) return null;

    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);

    // Set texture parameters
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);

    // Upload texture data
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      width,
      height,
      0,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      data || null
    );

    this.gl.bindTexture(this.gl.TEXTURE_2D, null);

    this.textures.set(name, texture);
    return texture;
  }

  // Create a framebuffer with attached texture for render-to-texture
  createFramebuffer(name: string, width: number, height: number): FramebufferInfo | null {
    const framebuffer = this.gl.createFramebuffer();
    const texture = this.gl.createTexture();

    if (!framebuffer || !texture) return null;

    // Setup texture
    this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_S, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_WRAP_T, this.gl.CLAMP_TO_EDGE);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MIN_FILTER, this.gl.LINEAR);
    this.gl.texParameteri(this.gl.TEXTURE_2D, this.gl.TEXTURE_MAG_FILTER, this.gl.LINEAR);

    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      width,
      height,
      0,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      null
    );

    // Setup framebuffer
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebuffer);
    this.gl.framebufferTexture2D(
      this.gl.FRAMEBUFFER,
      this.gl.COLOR_ATTACHMENT0,
      this.gl.TEXTURE_2D,
      texture,
      0
    );

    // Check framebuffer completeness
    if (this.gl.checkFramebufferStatus(this.gl.FRAMEBUFFER) !== this.gl.FRAMEBUFFER_COMPLETE) {
      console.error('Framebuffer not complete');
      this.gl.deleteFramebuffer(framebuffer);
      this.gl.deleteTexture(texture);
      return null;
    }

    // Cleanup
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);

    const framebufferInfo: FramebufferInfo = {
      framebuffer,
      texture,
      width,
      height
    };

    this.framebuffers.set(name, framebufferInfo);
    this.textures.set(name, texture);

    return framebufferInfo;
  }

  // Get a texture by name
  getTexture(name: string): WebGLTexture | null {
    return this.textures.get(name) || null;
  }

  // Get a framebuffer by name
  getFramebuffer(name: string): FramebufferInfo | null {
    return this.framebuffers.get(name) || null;
  }

  // Bind a framebuffer for rendering
  bindFramebuffer(name: string): boolean {
    const framebufferInfo = this.framebuffers.get(name);
    if (framebufferInfo) {
      this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, framebufferInfo.framebuffer);
      this.gl.viewport(0, 0, framebufferInfo.width, framebufferInfo.height);
      return true;
    }
    return false;
  }

  // Bind the default framebuffer (screen)
  bindDefaultFramebuffer(width: number, height: number): void {
    this.gl.bindFramebuffer(this.gl.FRAMEBUFFER, null);
    this.gl.viewport(0, 0, width, height);
  }

  // Bind a texture to a specific texture unit
  bindTexture(name: string, unit: number = 0): boolean {
    const texture = this.textures.get(name);
    if (texture) {
      this.gl.activeTexture(this.gl.TEXTURE0 + unit);
      this.gl.bindTexture(this.gl.TEXTURE_2D, texture);
      return true;
    }
    return false;
  }

  // Resize a framebuffer and its texture
  resizeFramebuffer(name: string, width: number, height: number): boolean {
    const framebufferInfo = this.framebuffers.get(name);
    if (!framebufferInfo) return false;

    framebufferInfo.width = width;
    framebufferInfo.height = height;

    this.gl.bindTexture(this.gl.TEXTURE_2D, framebufferInfo.texture);
    this.gl.texImage2D(
      this.gl.TEXTURE_2D,
      0,
      this.gl.RGBA,
      width,
      height,
      0,
      this.gl.RGBA,
      this.gl.UNSIGNED_BYTE,
      null
    );
    this.gl.bindTexture(this.gl.TEXTURE_2D, null);

    return true;
  }

  // Create ping-pong framebuffers for feedback effects
  createPingPongFramebuffers(baseName: string, width: number, height: number): boolean {
    const fb1 = this.createFramebuffer(`${baseName}_1`, width, height);
    const fb2 = this.createFramebuffer(`${baseName}_2`, width, height);
    return fb1 !== null && fb2 !== null;
  }

  // Swap ping-pong framebuffers (for feedback effects)
  swapPingPongFramebuffers(baseName: string): void {
    const fb1 = this.framebuffers.get(`${baseName}_1`);
    const fb2 = this.framebuffers.get(`${baseName}_2`);

    if (fb1 && fb2) {
      // Swap framebuffers
      this.framebuffers.set(`${baseName}_1`, fb2);
      this.framebuffers.set(`${baseName}_2`, fb1);

      // Swap textures
      this.textures.set(`${baseName}_1`, fb2.texture);
      this.textures.set(`${baseName}_2`, fb1.texture);
    }
  }

  // Clear a framebuffer with specified color
  clearFramebuffer(name: string, r: number = 0, g: number = 0, b: number = 0, a: number = 1): boolean {
    if (this.bindFramebuffer(name)) {
      this.gl.clearColor(r, g, b, a);
      this.gl.clear(this.gl.COLOR_BUFFER_BIT);
      return true;
    }
    return false;
  }

  // Clean up all textures and framebuffers
  dispose(): void {
    // Delete framebuffers
    this.framebuffers.forEach(framebufferInfo => {
      this.gl.deleteFramebuffer(framebufferInfo.framebuffer);
    });

    // Delete textures
    this.textures.forEach(texture => {
      this.gl.deleteTexture(texture);
    });

    this.framebuffers.clear();
    this.textures.clear();
  }
}