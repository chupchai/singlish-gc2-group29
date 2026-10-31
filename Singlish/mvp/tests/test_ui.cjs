const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
function setup(fetch) {
  const elements = {};
  const html = fs.readFileSync(path.join(__dirname, '../static/index.html'), 'utf8');
  const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]));
  const document = { addEventListener() {}, getElementById(id) {
    if (!ids.has(id)) return null;
    if (!elements[id]) elements[id] = { hidden: false, disabled: false, textContent: '',
      listeners: {}, attributes: {}, classList: { add() {}, remove() {} },
      addEventListener(type, handler) { this.listeners[type] = handler; },
      scrollIntoView() { this.scrolledIntoView = true; },
      pause() {}, removeAttribute(name) { delete this.attributes[name]; },
      setAttribute(name, value) { this.attributes[name] = value; } };
    return elements[id];
  }};
  const context = vm.createContext({ document, fetch, addEventListener() {}, URL: {createObjectURL: ()=>'blob:preview',revokeObjectURL(){}},
    FormData: class { constructor() {this.items=[]} append(...args) {this.items.push(args)} } });
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/recorder.js'),'utf8'), context);
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../static/app.js'),'utf8'), context);
  const select = (file) => {context.file = file; vm.runInContext('selectFile(file)',context)};
  const submit = () => elements['upload-form'].listeners.submit({preventDefault(){}});
  return {elements,select,submit};
}
const file = {name:'test.wav',size:1000};
const ml = {display_label:'Singapore Accent',probability:0.8765,decision_score:0.8765};
const ensemble = {display_label:'Singapore Accent',probability:0.9234,decision_score:1.2345};
const success = {prediction:ensemble, models:{ml:{name:'ML',version:'svm_65_c_261024',prediction:ml},
  ensemble:{name:'Ensemble',version:'fusion_svm_c_261024',prediction:ensemble}},agreement:'agree',
  audio:{duration_seconds:3,classified_duration_seconds:3},model_version:'fusion_svm_c_261024',warnings:[]};

test('selection clears stale result and validates file', () => {
  const {elements,select} = setup();
  select(file);
  assert.equal(elements.submit.disabled,false);
  assert.equal(elements.filename.textContent,'test.wav');
  assert.equal(elements.result.hidden,true);
  select({name:'bad.mp3',size:100});
  assert.equal(elements.submit.disabled,true);
  assert.equal(elements.error.hidden,false);
  select(file);
  assert.equal(elements.error.hidden,true);
  select({name:'speech.FLAC',size:1000});
  assert.equal(elements.submit.disabled,false);
  assert.equal(elements.filename.textContent,'speech.FLAC');
});

test('loading prevents selection changes; success renders label and restores controls', async () => {
  let resolve;
  const {elements,select,submit} = setup(() => new Promise(r=>resolve=r));
  select(file);
  const request=submit();
  assert.equal(elements.submit.disabled,true);
  assert.equal(elements['audio-file'].disabled,true);
  assert.equal(elements.loading.hidden,false);
  select({name:'other.wav',size:100});
  assert.equal(elements.filename.textContent,'test.wav');
  resolve({ok:true,json:async()=>success});
  await request;
  assert.equal(elements['prediction-label'].textContent,'Singapore Accent');
  assert.equal(elements['ml-score'].textContent,'87.6%');
  assert.equal(elements['ensemble-label'].textContent,'Singapore Accent');
  assert.equal(elements['ensemble-score'].textContent,'92.3%');
  assert.equal(elements['model-agreement'].textContent,'Model agreement: Agree');
  assert.equal(elements.result.hidden,false);
  assert.equal(elements['result-card'].scrolledIntoView,true);
  assert.equal(elements.loading.hidden,true);
  assert.equal(elements.submit.disabled,false);
  assert.equal(elements['result-card'].attributes['aria-busy'],'false');
  assert.equal(elements.duration.textContent,'3.00 s');
  select({name:'other.wav',size:100});
  assert.equal(elements.duration.textContent,'—');
  assert.equal(elements.result.hidden,true);
});

test('API failure hides result and permits retry', async () => {
  let count=0;
  const {elements,select,submit}=setup(async()=> ++count === 1 ?
    {ok:false,json:async()=>({error:{code:'invalid_duration',message:'The clip is less than 3 seconds after silence trimming. Please retry with a longer recording.'}})} : {ok:true,json:async()=>success});
  select(file); await submit();
  assert.match(elements.error.textContent,/less than 3 seconds after silence trimming.*Please retry/);
  assert.equal(elements.result.hidden,true);
  assert.equal(elements.submit.disabled,false);
  await submit();
  assert.equal(elements.result.hidden,false);
  assert.equal(elements.error.hidden,true);
});

test('network failure produces readable error', async () => {
  const {elements,select,submit}=setup(async()=>{throw new Error('Failed to fetch')});
  select(file); await submit();
  assert.match(elements.error.textContent,/Cannot reach the server/);
  assert.equal(elements.loading.hidden,true);
  assert.equal(elements.submit.disabled,false);
});


test('disagreement shows Inconclusive and keeps both model predictions', async () => {
  const {elements,select,submit} = setup(async () => ({ok:true,json:async () => ({...success,
    prediction:ensemble,agreement:'disagree',
    models:{...success.models,ml:{...success.models.ml,prediction:{display_label:'Non-Singapore Accent',probability:0.876,decision_score:-0.876}},
      ensemble:success.models.ensemble}})}));
  select(file);
  await submit();
  assert.equal(elements['prediction-label'].textContent,'Inconclusive');
  assert.equal(elements['ml-label'].textContent,'Non-Singapore Accent');
  assert.equal(elements['ml-score'].textContent,'87.6%');
  assert.equal(elements['ensemble-label'].textContent,'Singapore Accent');
  assert.equal(elements['model-agreement'].textContent,'Model agreement: Disagree. Please record again.');
});

test('non-Singapore Ensemble result shows its predicted-class probability', async () => {
  const prediction = {display_label:'Non-Singapore Accent',probability:0.912,decision_score:-1.2};
  const {elements,select,submit} = setup(async () => ({ok:true,json:async () => ({...success,
    prediction,agreement:'disagree',models:{...success.models,
      ensemble:{...success.models.ensemble,prediction}}})}));
  select(file);
  await submit();
  assert.equal(elements['prediction-label'].textContent,'Inconclusive');
  assert.equal(elements['ensemble-score'].textContent,'91.2%');
});

const tick = () => new Promise(r => setImmediate(r));
const transcript = {transcript:' they tried to keep it a secret ',model_version:'whisper-small-mnsc-sg-15k-heldout',
  decoding:'temperature fallback',audio:{duration_seconds:3,transcribed_duration_seconds:3},warnings:[]};
function routes(handlers, urls) {
  return (url) => { urls.push(url); return handlers[url](); };
}

test('successful classification is followed by a rendered transcript', async () => {
  const urls = [];
  const {elements,select,submit} = setup(routes({'/api/classify': async () => ({ok:true,json:async()=>success}),
    '/api/transcribe': async () => ({ok:true,json:async()=>transcript})}, urls));
  select(file);
  await submit();
  await tick();
  assert.deepEqual(urls,['/api/classify','/api/transcribe']);
  assert.equal(elements['transcript-text'].textContent,'they tried to keep it a secret');
  assert.equal(elements['transcript-text'].hidden,false);
  assert.equal(elements['transcript-loading'].hidden,true);
  assert.equal(elements['transcript-error'].hidden,true);
  assert.equal(elements.transcript.attributes['aria-busy'],'false');
});

test('pending transcription shows loading without blocking the controls', async () => {
  const urls = [];
  const {elements,select,submit} = setup(routes({'/api/classify': async () => ({ok:true,json:async()=>success}),
    '/api/transcribe': () => new Promise(() => {})}, urls));
  select(file);
  await submit();
  assert.deepEqual(urls,['/api/classify','/api/transcribe']);
  assert.equal(elements['transcript-loading'].hidden,false);
  assert.equal(elements.transcript.attributes['aria-busy'],'true');
  assert.equal(elements.submit.disabled,false);
  assert.equal(elements.result.hidden,false);
});

test('transcription failure stays in the transcript region', async () => {
  const urls = [];
  const {elements,select,submit} = setup(routes({'/api/classify': async () => ({ok:true,json:async()=>success}),
    '/api/transcribe': async () => ({ok:false,json:async()=>({error:{code:'transcription_unavailable',
      message:'Transcription is unavailable on this server. Classification still works.'}})})}, urls));
  select(file);
  await submit();
  await tick();
  assert.equal(elements['transcript-error'].textContent,'Transcription is unavailable on this server. Classification still works.');
  assert.equal(elements['transcript-error'].hidden,false);
  assert.equal(elements['transcript-text'].hidden,true);
  assert.equal(elements['transcript-loading'].hidden,true);
  assert.equal(elements['prediction-label'].textContent,'Singapore Accent');
  assert.equal(elements.result.hidden,false);
  assert.equal(elements.error.hidden,true);
});

test('failed classification does not request a transcript', async () => {
  const urls = [];
  const {elements,select,submit} = setup(routes({'/api/classify': async () => ({ok:false,json:async()=>({error:{code:'invalid_duration',message:'Too short. Please retry.'}})})}, urls));
  select(file);
  await submit();
  await tick();
  assert.deepEqual(urls,['/api/classify']);
  assert.equal(elements.error.hidden,false);
});

test('new selection clears a pending transcript and ignores its late response', async () => {
  let resolve;
  const urls = [];
  const {elements,select,submit} = setup(routes({'/api/classify': async () => ({ok:true,json:async()=>success}),
    '/api/transcribe': () => new Promise(r => resolve = r)}, urls));
  select(file);
  await submit();
  assert.equal(elements['transcript-loading'].hidden,false);
  select({name:'other.wav',size:1000});
  assert.equal(elements['transcript-loading'].hidden,true);
  assert.equal(elements.transcript.attributes['aria-busy'],'false');
  resolve({ok:true,json:async()=>transcript});
  await tick();
  assert.equal(elements['transcript-text'].hidden,true);
  assert.equal(elements['transcript-text'].textContent,'');
});

test('empty transcript shows the server warning', async () => {
  const {elements,select,submit} = setup(routes({'/api/classify': async () => ({ok:true,json:async()=>success}),
    '/api/transcribe': async () => ({ok:true,json:async()=>({...transcript,transcript:'',warnings:['No speech was recognised in this recording.']})})}, []));
  select(file);
  await submit();
  await tick();
  assert.equal(elements['transcript-text'].textContent,'No speech was recognised in this recording.');
});

test('uncalibrated model overrides show no probability', async () => {
  const {elements,select,submit} = setup(async () => ({ok:true,json:async () => ({...success,
    models:{...success.models,ml:{...success.models.ml,prediction:{...ml,probability:null}}}})}));
  select(file);
  await submit();
  assert.equal(elements['ml-score'].textContent,'—');
  assert.equal(elements['ensemble-score'].textContent,'92.3%');
});
