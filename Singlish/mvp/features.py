"""Exact 91-feature extraction for the upstream-preprocessed E1 study."""
import librosa
import numpy as np

SAMPLE_RATE = 16_000
MIN_SECONDS = 3
MAX_SECONDS = 6
DELTA_WIDTH = 9
N_MFCC = 13
N_FFT = 1024
HOP_LENGTH = 256
EXTRACTION_VERSION = "librosa91-parquet-upstream-v2"

FEATURE_NAMES = (
    [f'mfcc_{i + 1}_mean' for i in range(N_MFCC)]
    + [f'mfcc_{i + 1}_std' for i in range(N_MFCC)]
    + ['f0_mean', 'f0_std', 'voiced_ratio']
    + [
        f'{family}_{stat}'
        for family in ('spectral_centroid', 'spectral_bandwidth', 'spectral_rolloff', 'zcr', 'rms')
        for stat in ('mean', 'std')
    ]
)
FEATURE_NAMES += [
    f'{family}_{i}_{stat}'
    for family in ('mfcc_delta', 'mfcc_delta_delta')
    for stat in ('mean', 'std')
    for i in range(1, N_MFCC + 1)
]
assert len(FEATURE_NAMES) == 91


def trim_silence(y):
    """Match upstream 25 ms RMS frames, 10 ms hop, and 100 ms padding."""
    frame, hop = 400, 160
    if not len(y):
        return y
    frames = y[None, :] if len(y) < frame else np.lib.stride_tricks.sliding_window_view(y, frame)[::hop]
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    ref = rms.max()
    if ref <= 0:
        return y
    with np.errstate(divide='ignore'):
        db = 20.0 * np.log10(np.maximum(rms, 1e-20) / ref)
    active = np.nonzero(db > -40.0)[0]
    return y[max(0, active[0] * hop - 1600):min(len(y), active[-1] * hop + frame + 1600)]


def normalize_loudness(y):
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) if len(y) else 0.0
    if rms < 1e-5:
        raise ValueError('The recording is silent or too quiet. Record clear English speech.')
    y = y * (10.0 ** (-20.0 / 20.0) / rms)
    peak = float(np.max(np.abs(y)))
    if peak > 0.99:
        y = y * (0.99 / peak)
    return y.astype(np.float32)


def mean_and_std(feature):
    return np.concatenate([np.mean(feature, axis=1), np.std(feature, axis=1)])


def extract_features(y):
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)
    f0 = librosa.yin(
        y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'),
        sr=SAMPLE_RATE, frame_length=2048, hop_length=HOP_LENGTH,
    )
    energy = rms.ravel()[:len(f0)]
    voiced = np.isfinite(f0) & (energy > max(float(np.median(energy)) * 0.1, 1e-4))
    voiced_f0 = f0[voiced]
    pitch_stats = [
        float(np.mean(voiced_f0)) if voiced_f0.size else 0.0,
        float(np.std(voiced_f0)) if voiced_f0.size else 0.0,
        float(np.mean(voiced)),
    ]
    time_features = [
        librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH),
        librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH),
        rms,
    ]
    delta = librosa.feature.delta(mfcc, order=1, width=DELTA_WIDTH)
    delta_delta = librosa.feature.delta(mfcc, order=2, width=DELTA_WIDTH)
    vector = np.concatenate([mean_and_std(mfcc), pitch_stats]
                            + [mean_and_std(x) for x in time_features]
                            + [mean_and_std(delta), mean_and_std(delta_delta)])
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert vector.shape == (len(FEATURE_NAMES),)
    return vector
