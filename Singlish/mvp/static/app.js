const $ = (id) => document.getElementById(id);
function formatProbability(probability) {
  return probability == null ? '—' : `${(probability * 100).toFixed(1)}%`;
}
let selectedFile = null;
let previewURL = null;
let busy = false;
let validFile = false;
let transcriptRequest = 0;
const microphoneSupported = MicrophoneRecorder.supported();
const recorder = new MicrophoneRecorder({
  onState(state) {
    const labels = { idle: 'Record audio', requesting: 'Requesting access…', recording: 'Stop recording', processing: 'Preparing audio…' };
    $('record-label').textContent = labels[state];
    $('record-button').setAttribute('aria-pressed', String(state === 'recording'));
    $('recording-live').hidden = state !== 'recording';
    const messages = { requesting: 'Allow microphone access in your browser to start.',
      recording: 'Read “They tried to keep it a secret but he was able to figure out everything”, then select Stop recording.', processing: 'Preparing your recording…' };
    if (messages[state]) $('recording-status').textContent = messages[state];
    updateControls();
  },
  onTime(seconds) { $('recording-timer').textContent = `0:${String(Math.floor(seconds)).padStart(2, '0')} / 0:10`; },
  onRecording(file) {
    selectFile(file);
    $('recording-status').textContent = 'Recording ready. Listen to the preview, then classify.';
  },
  onError(error) {
    const messages = {
      NotAllowedError: 'Microphone access was denied. Allow microphone access in your browser settings, or upload a WAV or FLAC file.',
      SecurityError: 'Microphone access is blocked by the browser. Check the site permissions or upload a WAV or FLAC file.',
      NotFoundError: 'No microphone was found. Connect one or upload a WAV or FLAC file.',
      NotReadableError: 'The microphone could not be opened. Check whether another app is using it, or upload a WAV or FLAC file.'
    };
    $('error').textContent = messages[error.name] || error.message || 'Recording failed. Please try again or upload a WAV or FLAC file.';
    $('error').hidden = false;
    $('recording-status').textContent = 'Recording did not complete. You can retry or upload audio.';
  }
});
function updateControls() {
  const capturing = recorder.state !== 'idle';
  $('audio-file').disabled = busy || capturing;
  $('dropzone').setAttribute('aria-disabled', String(busy || capturing));
  $('submit').disabled = busy || capturing || !validFile;
  $('record-button').disabled = busy || !microphoneSupported || ['requesting', 'processing'].includes(recorder.state);
  $('cancel-recording').hidden = !capturing;
  $('file-info').hidden = capturing || !selectedFile;
  $('preview').hidden = capturing || !validFile;
}
function clearResult() {
  $('result').hidden = true;
  $('empty-result').hidden = false;
  $('error').hidden = true;
  $('duration').textContent = '—';
  clearTranscript();
}
function clearTranscript() {
  transcriptRequest += 1; // Responses for earlier requests are ignored.
  $('transcript-loading').hidden = true;
  $('transcript-text').hidden = true;
  $('transcript-text').textContent = '';
  $('transcript-raw').hidden = true;
  $('transcript-raw').textContent = '';
  $('transcript-error').hidden = true;
  $('transcript').setAttribute('aria-busy', 'false');
}
function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (c) =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
// Escape the corrected transcript, then highlight each replaced word with <mark>.
function renderCorrected(text, changes) {
  let html = escapeHtml(text);
  const seen = new Set();
  for (const change of changes) {
    const replacement = change && change.replacement;
    if (!replacement || seen.has(replacement)) continue;
    seen.add(replacement);
    const escaped = escapeHtml(replacement);
    html = html.split(escaped).join(`<mark class="lex">${escaped}</mark>`);
  }
  return html;
}
// Runs after a successful classification; its failures stay inside the transcript region.
async function transcribe(file, applyLexicon = false) {
  const id = ++transcriptRequest;
  $('transcript-loading').hidden = false;
  $('transcript').setAttribute('aria-busy', 'true');
  const form = new FormData(); form.append('audio', file);
  // The Singlish lexicon lookup runs server-side only when a Singapore accent was detected.
  const url = applyLexicon ? '/api/transcribe?correct=1' : '/api/transcribe';
  try {
    const response = await fetch(url, { method: 'POST', body: form });
    const data = await response.json();
    if (id !== transcriptRequest) return;
    if (!response.ok || typeof data.transcript !== 'string') throw new Error(data.error?.message || 'Transcription failed. Try again.');
    const text = data.transcript.trim();
    const raw = (data.transcript_raw || '').trim();
    const changes = Array.isArray(data.changes) ? data.changes : [];
    if (text && changes.length) {
      // After: corrected transcript with the replaced words highlighted.
      $('transcript-text').innerHTML = renderCorrected(text, changes);
      // Before: raw Whisper output, shown only when the lexicon changed something.
      $('transcript-raw').innerHTML = `<span class="transcript-raw-label">Before correction:</span> ${escapeHtml(raw)}`;
      $('transcript-raw').hidden = false;
    } else {
      $('transcript-text').textContent = text || data.warnings?.[0] || 'No speech was recognised in this recording.';
      $('transcript-raw').hidden = true;
    }
    $('transcript-text').hidden = false;
  } catch (error) {
    if (id !== transcriptRequest) return;
    $('transcript-error').textContent = error.message === 'Failed to fetch' ? 'Cannot reach the server. Please retry.' : error.message;
    $('transcript-error').hidden = false;
  } finally {
    if (id === transcriptRequest) {
      $('transcript-loading').hidden = true;
      $('transcript').setAttribute('aria-busy', 'false');
    }
  }
}
function selectFile(file) {
  if (busy || recorder.state !== 'idle') return;
  clearResult();
  selectedFile = file || null;
  validFile = false;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  $('preview').pause();
  $('preview').removeAttribute('src');
  if (microphoneSupported) $('recording-status').textContent = 'Recording stops automatically at 10 seconds.';
  updateControls();
  if (!selectedFile) return;
  $('filename').textContent = selectedFile.name;
  $('filesize').textContent = `${(selectedFile.size / 1024 / 1024).toFixed(2)} MiB`;
  if (selectedFile.size > 10 * 1024 * 1024 || !/\.(wav|flac)$/i.test(selectedFile.name)) {
    $('error').textContent = 'Choose a .wav or .flac file up to 10 MiB.';
    $('error').hidden = false;
    $('submit').disabled = true;
    $('preview').hidden = true;
    return;
  }
  previewURL = URL.createObjectURL(selectedFile);
  $('preview').src = previewURL;
  validFile = true;
  updateControls();
}
$('record-button').addEventListener('click', () => {
  if (busy || !microphoneSupported) return;
  if (recorder.state === 'recording') { recorder.stop(); return; }
  if (recorder.state !== 'idle') return;
  clearResult();
  $('preview').pause();
  recorder.start();
});
$('cancel-recording').addEventListener('click', () => {
  recorder.cancel();
  $('recording-status').textContent = validFile ? 'Recording cancelled. Your previous audio is still available.' :
    'Recording cancelled. Record again or upload a WAV or FLAC file.';
});
function cancelForPageLeave() {
  if (recorder.state !== 'idle') {
    recorder.cancel();
    $('recording-status').textContent = 'Recording cancelled because the page was left. Please try again.';
  }
}
globalThis.addEventListener('pagehide', cancelForPageLeave);
document.addEventListener('visibilitychange', () => {
  if (document.hidden) cancelForPageLeave();
});
updateControls();
if (!microphoneSupported) $('recording-status').textContent = 'Microphone recording requires a supported browser on HTTPS or localhost. You can still upload a WAV or FLAC file.';
$('audio-file').addEventListener('change', (e) => selectFile(e.target.files[0]));
for (const event of ['dragenter', 'dragover']) $('dropzone').addEventListener(event, (e) => {
  e.preventDefault(); if (!busy && recorder.state === 'idle') $('dropzone').classList.add('dragging');
});
for (const event of ['dragleave', 'drop']) $('dropzone').addEventListener(event, (e) => {
  e.preventDefault(); $('dropzone').classList.remove('dragging');
});
$('dropzone').addEventListener('drop', (e) => selectFile(e.dataTransfer.files[0]));
$('upload-form').addEventListener('submit', async (e) => {
  e.preventDefault(); if (!validFile || busy || recorder.state !== 'idle') return;
  busy = true; clearResult();
  updateControls();
  $('empty-result').hidden = true;
  $('loading').hidden = false;
  $('result-card').setAttribute('aria-busy', 'true');
  $('result-card').scrollIntoView({ block: 'start' });
  const form = new FormData(); form.append('audio', selectedFile);
  try {
    const response = await fetch('/api/classify', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error?.message || 'Classification failed. Try again.');
    const agreed = data.agreement === 'agree';
    const ml = data.models.ml;
    const ensemble = data.models.ensemble;
    // Run the Singlish lexicon lookup only when the ensemble predicts Singapore (class_id 1).
    const singaporeDetected = ensemble.prediction.class_id === 1;
    $('prediction-label').textContent = ensemble.prediction.display_label;
    $('ml-label').textContent = ml.prediction.display_label;
    $('ml-score').textContent = formatProbability(ml.prediction.probability);
    $('ensemble-label').textContent = ensemble.prediction.display_label;
    $('ensemble-score').textContent = formatProbability(ensemble.prediction.probability);
    $('ml-version').textContent = ml.version;
    $('model-agreement').textContent = agreed ? 'Model agreement: Agree' : 'Model agreement: Disagree';
    $('duration').textContent = `${data.audio.duration_seconds.toFixed(2)} s`;
    $('classified-duration').textContent = `${data.audio.classified_duration_seconds.toFixed(2)} s`;
    $('model-version').textContent = data.model_version;
    $('warnings').textContent = data.warnings.join(' ');
    $('warnings').hidden = data.warnings.length === 0;
    $('result').hidden = false;
    transcribe(selectedFile, singaporeDetected); // Not awaited: classification controls return immediately.
  } catch (error) {
    $('error').textContent = error.message === 'Failed to fetch' ? 'Cannot reach the server. Please retry.' : error.message;
    $('error').hidden = false;
    $('empty-result').hidden = false;
  } finally {
    busy = false;
    $('loading').hidden = true;
    updateControls();
    $('result-card').setAttribute('aria-busy', 'false');
    if (!$('result').hidden) $('result-card').scrollIntoView({ block: 'start' });
  }
});
