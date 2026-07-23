/**
 * PCM Processor — AudioWorklet that captures mic audio and
 * downsamples to 16 kHz mono PCM16 for the MeetASR WebSocket.
 *
 * Contract:
 *   - Input:  float32, any sample rate, mono (channel 0)
 *   - Output: Int16Array (PCM16 LE), 16 000 Hz, posted via message
 *
 * The main thread sends { command: "init", sampleRate } once after
 * construction so the processor knows the native rate.
 */

class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._nativeSR = 48000; // default; overwritten by init message
    this._targetSR = 16000;
    this._resampleRatio = 1;
    this._accumulator = new Float32Array(0);

    // Buffer to accumulate enough samples before posting.
    // Target: ~100 ms = 1 600 samples @ 16 kHz = 3 200 bytes PCM16.
    this._chunkSamples = 1600;

    this.port.onmessage = (e) => {
      if (e.data.command === "init") {
        this._nativeSR = e.data.sampleRate;
        this._resampleRatio = this._nativeSR / this._targetSR;
      }
    };
  }

  /**
   * Called ~every 128 samples by the audio thread.
   */
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) return true;

    const mono = input[0]; // Float32Array, native sample rate

    // ---- Simple linear-interpolation downsampling ----
    const outLen = Math.floor(mono.length / this._resampleRatio);
    if (outLen === 0) return true;

    const resampled = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const srcIdx = i * this._resampleRatio;
      const lo = Math.floor(srcIdx);
      const hi = Math.min(lo + 1, mono.length - 1);
      const frac = srcIdx - lo;
      resampled[i] = mono[lo] * (1 - frac) + mono[hi] * frac;
    }

    // Accumulate
    const prev = this._accumulator;
    const merged = new Float32Array(prev.length + resampled.length);
    merged.set(prev, 0);
    merged.set(resampled, prev.length);
    this._accumulator = merged;

    // Emit chunks of _chunkSamples
    while (this._accumulator.length >= this._chunkSamples) {
      const chunk = this._accumulator.slice(0, this._chunkSamples);
      this._accumulator = this._accumulator.slice(this._chunkSamples);

      // Float32 → Int16 (PCM16 LE)
      const pcm16 = new Int16Array(chunk.length);
      for (let j = 0; j < chunk.length; j++) {
        const s = Math.max(-1, Math.min(1, chunk[j]));
        pcm16[j] = s < 0 ? s * 0x8000 : s * 0x7fff;
      }

      this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    }

    return true;
  }
}

registerProcessor("pcm-processor", PCMProcessor);
