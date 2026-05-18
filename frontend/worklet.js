// AudioWorklet processor: forwards every input buffer to the main thread.
// Each buffer is a Float32Array of ~128 samples at the AudioContext sample rate.

class CaptureProcessor extends AudioWorkletProcessor {
    process(inputs) {
        const input = inputs[0];
        if (input && input[0] && input[0].length > 0) {
            // The underlying buffer is recycled, so copy before sending.
            this.port.postMessage(input[0].slice());
        }
        return true;
    }
}

registerProcessor("capture", CaptureProcessor);
