class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(1024);
    this.offset = 0;
  }

  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel?.length) return true;

    let sourceOffset = 0;
    while (sourceOffset < channel.length) {
      const count = Math.min(
        channel.length - sourceOffset,
        this.buffer.length - this.offset,
      );
      this.buffer.set(
        channel.subarray(sourceOffset, sourceOffset + count),
        this.offset,
      );
      this.offset += count;
      sourceOffset += count;

      if (this.offset === this.buffer.length) {
        const samples = this.buffer;
        this.port.postMessage(samples, [samples.buffer]);
        this.buffer = new Float32Array(1024);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
