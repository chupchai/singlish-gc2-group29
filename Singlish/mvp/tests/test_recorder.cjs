const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/recorder.js'), 'utf8');
const tick = () => new Promise(resolve => setImmediate(resolve));
const audio = (channels, rate = 16000) => ({sampleRate: rate, length: channels[0].length,
  numberOfChannels: channels.length, getChannelData: i => channels[i]});
function setup(options = {}) {
  const tracks = [{stopped: false, listeners: {}, stop() {this.stopped = true;},
    addEventListener(type, callback) {this.listeners[type] = callback;}}];
  const stream = {getTracks: () => tracks};
  const contexts = [], recorders = [], timers = [], states = [], errors = [], recordings = [];
  class FakeContext {
    constructor() {this.state = 'suspended'; contexts.push(this);}
    async resume() {this.state = 'running';}
    async close() {this.state = 'closed';}
    async decodeAudioData() {return options.decode ? options.decode() : audio([new Float32Array(48000).fill(.2)]);}
  }
  class FakeRecorder {
    static isTypeSupported(type) {return type === (options.mimeType || 'audio/webm;codecs=opus');}
    constructor(_, config) {this.mimeType = config?.mimeType || ''; this.state = 'inactive'; recorders.push(this);}
    start() {this.state = 'recording';}
    stop() {this.state = 'inactive'; queueMicrotask(() => {
      this.ondataavailable?.({data: new Blob(['encoded'])}); this.onstop?.();
    });}
  }
  const context = vm.createContext({Blob, File, DataView, ArrayBuffer, Uint8Array,
    navigator: {mediaDevices: {getUserMedia: () => options.getUserMedia ? options.getUserMedia(stream) : Promise.resolve(stream)}},
    AudioContext: FakeContext, MediaRecorder: FakeRecorder, performance: {now: () => 0}, isSecureContext: true,
    setInterval(callback) {return {callback};}, clearInterval() {},
    setTimeout(callback, delay) {const timer={callback,delay}; timers.push(timer); return timer;},
    clearTimeout(timer) {if (timer) timer.cleared=true;}});
  vm.runInContext(source, context);
  const {MicrophoneRecorder, encodePCM16} = vm.runInContext('({MicrophoneRecorder, encodePCM16})', context);
  const recorder = new MicrophoneRecorder({onState: state => states.push(state), onTime() {},
    onRecording: (file, duration) => recordings.push({file, duration}), onError: error => errors.push(error)});
  return {context, recorder, encodePCM16, tracks, contexts, recorders, timers, states, errors, recordings};
}

test('WAV export writes mono PCM16 headers, downmixes and clips samples', () => {
  const {encodePCM16}=setup();
  const wav=encodePCM16(audio([new Float32Array([1, -1, 2, -2, .5]), new Float32Array([1,-1,2,-2,-.5])]));
  const bytes=Buffer.from(wav), view=new DataView(wav);
  assert.equal(bytes.toString('ascii',0,4),'RIFF');
  assert.equal(bytes.toString('ascii',8,12),'WAVE');
  assert.equal(view.getUint16(20,true),1);
  assert.equal(view.getUint16(22,true),1);
  assert.equal(view.getUint32(24,true),16000);
  assert.equal(view.getUint16(34,true),16);
  assert.equal(view.getUint32(40,true),10);
  assert.deepEqual(Array.from({length:5},(_,i)=>view.getInt16(44+2*i,true)),[32767,-32768,32767,-32768,0]);
});

test('WAV export caps decoded audio at ten seconds', () => {
  const {encodePCM16}=setup();
  const wav=encodePCM16(audio([new Float32Array(11*48000)],48000));
  assert.equal(new DataView(wav).getUint32(40,true),10*48000*2);
});

test('stop releases microphone before decoding and returns a WAV for preview', async () => {
  let finishDecode;
  const s=setup({mimeType:'audio/mp4',decode:()=>new Promise(resolve=>finishDecode=resolve)});
  await s.recorder.start();
  assert.equal(s.recorders[0].mimeType,'audio/mp4');
  s.recorder.stop(); await tick();
  assert.equal(s.tracks[0].stopped,true);
  assert.equal(s.recorder.state,'processing');
  finishDecode(audio([new Float32Array(48000).fill(.2)])); await tick();
  assert.equal(s.recordings[0].file.name,'microphone-recording.wav');
  assert.equal(s.recordings[0].file.type,'audio/wav');
  assert.equal(s.recordings[0].duration,3);
  assert.equal(s.contexts[0].state,'closed');
  assert.deepEqual(s.states,['requesting','recording','processing','idle']);
});

test('ten-second timer stops capture and clears the recording resources', async () => {
  const s=setup(); await s.recorder.start();
  assert.equal(s.timers[0].delay,10000);
  s.timers[0].callback(); await tick();
  assert.equal(s.tracks[0].stopped,true);
  assert.equal(s.timers[0].cleared,true);
  assert.equal(s.recorder.state,'idle');
  assert.equal(s.recordings.length,1);
});

test('permission rejection restores idle and closes audio context', async () => {
  const s=setup({getUserMedia:async()=>{throw new DOMException('Denied','NotAllowedError');}});
  await s.recorder.start();
  assert.equal(s.recorder.state,'idle');
  assert.equal(s.errors[0].name,'NotAllowedError');
  assert.equal(s.contexts[0].state,'closed');
  assert.equal(s.recordings.length,0);
});

test('cancel during permission request releases a stream that arrives late', async () => {
  let grant;
  const s=setup({getUserMedia:stream=>new Promise(resolve=>grant=()=>resolve(stream))});
  const start=s.recorder.start(); await tick(); s.recorder.cancel(); grant(); await start;
  assert.equal(s.tracks[0].stopped,true);
  assert.equal(s.contexts[0].state,'closed');
  assert.equal(s.recorders.length,0);
  assert.equal(s.recorder.state,'idle');
});

test('cancel during recording discards audio and releases the microphone', async () => {
  const s=setup(); await s.recorder.start(); s.recorder.cancel(); await tick();
  assert.equal(s.tracks[0].stopped,true);
  assert.equal(s.contexts[0].state,'closed');
  assert.equal(s.recordings.length,0);
  assert.equal(s.recorder.state,'idle');
});

test('cancel during decoding prevents a late recording from replacing selected audio', async () => {
  let finishDecode;
  const s=setup({decode:()=>new Promise(resolve=>finishDecode=resolve)});
  await s.recorder.start(); s.recorder.stop(); await tick(); s.recorder.cancel();
  finishDecode(audio([new Float32Array(48000)])); await tick();
  assert.equal(s.recordings.length,0);
  assert.equal(s.recorder.state,'idle');
});

test('microphone disconnection restores idle and reports an error', async () => {
  const s=setup(); await s.recorder.start(); s.tracks[0].listeners.ended(); await tick();
  assert.equal(s.tracks[0].stopped,true);
  assert.equal(s.recorder.state,'idle');
  assert.match(s.errors[0].message,/disconnected/);
  assert.equal(s.recordings.length,0);
});

test('decode failure releases resources and allows retry', async () => {
  const s=setup({decode:async()=>{throw new Error('Invalid encoding');}});
  await s.recorder.start(); s.recorder.stop(); await tick();
  assert.equal(s.tracks[0].stopped,true);
  assert.equal(s.contexts[0].state,'closed');
  assert.equal(s.recorder.state,'idle');
  assert.equal(s.errors.length,1);
});
