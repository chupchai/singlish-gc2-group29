/* Browser microphone capture. Record in a supported format, then export mono PCM WAV. */
function encodePCM16(audio, maxSeconds = 10) {
  const frames = Math.min(audio.length, Math.floor(audio.sampleRate * maxSeconds));
  if (!frames || !audio.numberOfChannels) throw new Error('No audio was recorded. Please try again.');
  const data = new ArrayBuffer(44 + frames * 2);
  const view = new DataView(data);
  const text = (offset, value) => [...value].forEach((character, i) => view.setUint8(offset + i, character.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + frames * 2, true);
  text(8, 'WAVE'); text(12, 'fmt '); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, audio.sampleRate, true); view.setUint32(28, audio.sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, frames * 2, true);
  const channels = Array.from({ length: audio.numberOfChannels }, (_, i) => audio.getChannelData(i));
  for (let i = 0; i < frames; i++) {
    let sample = 0;
    for (const channel of channels) sample += channel[i];
    sample /= channels.length;
    sample = Number.isFinite(sample) ? Math.max(-1, Math.min(1, sample)) : 0;
    view.setInt16(44 + i * 2, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
  }
  return data;
}

class MicrophoneRecorder {
  static supported() {
    return globalThis.isSecureContext !== false && Boolean(globalThis.navigator?.mediaDevices?.getUserMedia &&
      globalThis.MediaRecorder && (globalThis.AudioContext || globalThis.webkitAudioContext));
  }
  constructor({ onState, onTime, onRecording, onError }) {
    this.state = 'idle';
    this.session = null;
    Object.assign(this, { onState, onTime, onRecording, onError });
  }
  active(session) { return this.session === session; }
  setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.onState(state);
  }
  stopTracks(session) { session.stream?.getTracks().forEach(track => track.stop()); }
  clearTimers(session) { clearInterval(session.ticker); clearTimeout(session.limit); }
  dispose(session) {
    this.clearTimers(session);
    if (session.recorder) {
      session.recorder.onstop = null;
      session.recorder.onerror = null;
      session.recorder.ondataavailable = null;
      if (session.recorder.state !== 'inactive') {
        try { session.recorder.stop(); } catch { /* Already stopped by the browser. */ }
      }
    }
    this.stopTracks(session);
    if (session.context && session.context.state !== 'closed') session.context.close().catch(() => {});
    if (this.active(session)) this.session = null;
  }
  fail(session, error) {
    if (!this.active(session)) return;
    this.dispose(session);
    this.setState('idle');
    this.onError(error);
  }
  async start() {
    if (this.state !== 'idle') return;
    const session = { chunks: [] };
    this.session = session;
    this.setState('requesting');
    try {
      const AudioContext = globalThis.AudioContext || globalThis.webkitAudioContext;
      // Resume during the button interaction for browsers that require a user gesture.
      session.context = new AudioContext();
      await session.context.resume();
      if (!this.active(session)) return;
      session.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!this.active(session)) { this.stopTracks(session); return; }
      const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/ogg;codecs=opus']
        .find(type => MediaRecorder.isTypeSupported(type));
      session.recorder = new MediaRecorder(session.stream, mimeType ? { mimeType } : undefined);
      session.recorder.ondataavailable = event => { if (event.data.size) session.chunks.push(event.data); };
      session.recorder.onerror = event => this.fail(session, event.error || new Error('Recording failed. Please try again.'));
      session.recorder.onstop = () => this.finish(session);
      session.stream.getTracks().forEach(track => track.addEventListener('ended', () => {
        if (this.active(session) && this.state === 'recording') {
          this.fail(session, new Error('The microphone disconnected. Please try again.'));
        }
      }));
      session.recorder.start();
      session.started = performance.now();
      this.setState('recording');
      this.onTime(0);
      session.ticker = setInterval(() => this.onTime(Math.min(10, (performance.now() - session.started) / 1000)), 250);
      session.limit = setTimeout(() => this.stop(), 10_000);
    } catch (error) { this.fail(session, error); }
  }
  stop() {
    if (this.state !== 'recording') return;
    const session = this.session;
    this.clearTimers(session);
    this.setState('processing');
    try { session.recorder.stop(); } catch (error) { this.fail(session, error); }
    // Release the physical microphone immediately, before decoding or uploading.
    this.stopTracks(session);
  }
  async finish(session) {
    if (!this.active(session)) return;
    this.clearTimers(session);
    this.setState('processing');
    this.stopTracks(session);
    try {
      const encoded = new Blob(session.chunks, { type: session.recorder.mimeType });
      const audio = await session.context.decodeAudioData(await encoded.arrayBuffer());
      if (!this.active(session)) return;
      const wav = encodePCM16(audio);
      const file = new File([wav], 'microphone-recording.wav', { type: 'audio/wav' });
      const duration = Math.min(audio.length / audio.sampleRate, 10);
      this.dispose(session);
      this.setState('idle');
      this.onRecording(file, duration);
    } catch (error) {
      this.fail(session, new Error('The recording could not be prepared. Please try again or upload a WAV file.'));
    }
  }
  cancel() {
    if (!this.session) return;
    this.dispose(this.session);
    this.setState('idle');
  }
}
