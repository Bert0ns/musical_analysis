import csv
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import librosa
import numpy as np
from librosa.beat import beat_track as beat_track
from librosa.effects import trim as effects_trim
from librosa.feature import (
    zero_crossing_rate,
    rms as feature_rms,
    spectral_centroid as feature_spectral_centroid,
    spectral_bandwidth as feature_spectral_bandwidth,
    spectral_rolloff as feature_spectral_rolloff,
    spectral_contrast as feature_spectral_contrast,
    spectral_flatness as feature_spectral_flatness,
    poly_features as feature_poly_features,
    chroma_stft as feature_chroma_stft,
    chroma_cqt as feature_chroma_cqt,
    chroma_cens as feature_chroma_cens,
    melspectrogram as feature_melspectrogram,
    mfcc as feature_mfcc,
    tonnetz as feature_tonnetz,
)
from librosa.onset import onset_strength as onset_strength, onset_detect as onset_detect
from librosa import decompose as librosa_decompose


def extract_features(file_path):
    """
    Extract a broad and robust set of audio features from a file using librosa.

    Quick contract:
    - Input: audio file path (string)
    - Output: (feature_vector: np.ndarray[float64], feature_names: list[str])
    - Properties: no NaN/Inf, scalars only, stable name order

    Strategy:
    1) Load mono audio with a fixed sample rate for consistency; trim leading/trailing silence.
    2) Compute temporal features (ZCR, RMS, onset rate, tempo/beat).
    3) Compute spectral features (centroid, bandwidth, rolloff 85/95, contrast, flatness, poly features).
    4) Compute chroma features (chroma STFT/CQT/CENS).
    5) Compute mel/MFCC features.
    6) Compute tonnetz (tonal features) and harmonic/percussive energy ratio (HPSS) from the spectrogram.

    All outputs are reduced to scalars (mean/std), cleaned from NaN/Inf, and returned in arrays.
    """
    # Consistent parameters for analysis
    sr_target = 22050
    n_fft = 2048
    hop_length = 512

    # Helper to safely push scalar values to output
    feature_names: list[str] = []
    feature_values: list[float] = []

    def _to_float_scalar(x) -> float:
        try:
            if isinstance(x, (list, tuple, np.ndarray)):
                x = np.asarray(x)
                if x.shape != ():
                    x = np.nanmean(x)
            x = float(x)
        except Exception:
            x = 0.0
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return x

    def push(name: str, value) -> None:
        feature_names.append(name)
        feature_values.append(_to_float_scalar(value))

    # 1) Load audio with fixed SR and trim silence
    y, sr = librosa.load(file_path, sr=sr_target, mono=True, res_type="kaiser_fast")
    if y is None or len(y) == 0:
        raise ValueError(f"Empty or unreadable audio: {file_path}")

    # Remove DC offset and trim silence
    y = y - np.mean(y)
    y_trimmed, _ = effects_trim(y, top_db=30)
    if len(y_trimmed) > 0:
        y = y_trimmed
    duration = max(1e-9, len(y) / sr)

    # Precompute STFT once (magnitude and power) for reuse
    S_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, center=True)
    S_mag = np.abs(S_complex)
    S_power = S_mag**2

    # 2) Temporal features
    zcr = zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    push("zcr_mean", np.mean(zcr))
    push("zcr_std", np.std(zcr))
    push("zcr_max", np.max(zcr) if zcr.size else 0.0)

    # RMS from power spectrogram (avoid recomputation)
    rms = feature_rms(S=S_power)[0]
    push("rms_mean", np.mean(rms))
    push("rms_std", np.std(rms))
    push("rms_max", np.max(rms) if rms.size else 0.0)

    # Onset rate (attacks per second)
    onset_env = onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames"
    )
    push("onset_rate", (len(onsets) / duration))

    # Tempo (BPM) and beats per second - reuse onset_env
    tempo, beats = beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    push("tempo_bpm", tempo)
    push("beats_per_sec", (len(beats) / duration))

    # 3) Spectral features (reuse S_power)
    spectral_centroid = feature_spectral_centroid(S=S_power, sr=sr)[0]
    push("centroid_mean", np.mean(spectral_centroid))
    push("centroid_std", np.std(spectral_centroid))

    spectral_bandwidth = feature_spectral_bandwidth(S=S_power, sr=sr)[0]
    push("bandwidth_mean", np.mean(spectral_bandwidth))
    push("bandwidth_std", np.std(spectral_bandwidth))

    rolloff85 = feature_spectral_rolloff(S=S_power, sr=sr, roll_percent=0.85)[0]
    rolloff95 = feature_spectral_rolloff(S=S_power, sr=sr, roll_percent=0.95)[0]
    push("rolloff85_mean", np.mean(rolloff85))
    push("rolloff85_std", np.std(rolloff85))
    push("rolloff95_mean", np.mean(rolloff95))
    push("rolloff95_std", np.std(rolloff95))

    contrast = feature_spectral_contrast(S=S_power, sr=sr)
    for i in range(contrast.shape[0]):
        push(f"contrast_{i}_mean", np.mean(contrast[i]))
        push(f"contrast_{i}_std", np.std(contrast[i]))

    flatness = feature_spectral_flatness(S=S_power)[0]
    push("flatness_mean", np.mean(flatness))
    push("flatness_std", np.std(flatness))

    # Poly features (up to order 2)
    poly = feature_poly_features(S=S_power, sr=sr, order=2)
    for i in range(poly.shape[0]):
        push(f"poly_{i}_mean", np.mean(poly[i]))
        push(f"poly_{i}_std", np.std(poly[i]))

    # 4) Chroma features
    chroma_stft = feature_chroma_stft(S=S_power, sr=sr)
    for i in range(chroma_stft.shape[0]):
        push(f"chroma_stft_{i}_mean", np.mean(chroma_stft[i]))
        push(f"chroma_stft_{i}_std", np.std(chroma_stft[i]))

    chroma_cqt = feature_chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    for i in range(chroma_cqt.shape[0]):
        push(f"chroma_cqt_{i}_mean", np.mean(chroma_cqt[i]))
        push(f"chroma_cqt_{i}_std", np.std(chroma_cqt[i]))

    # Reuse chroma_cqt for CENS (avoid recomputation)
    chroma_cens = feature_chroma_cens(C=chroma_cqt, hop_length=hop_length)
    for i in range(chroma_cens.shape[0]):
        push(f"chroma_cens_{i}_mean", np.mean(chroma_cens[i]))
        push(f"chroma_cens_{i}_std", np.std(chroma_cens[i]))

    # 5) Mel/MFCC (reuse mel for MFCC)
    mel_spec = feature_melspectrogram(S=S_power, sr=sr)
    push("mel_spec_mean", np.mean(mel_spec))
    push("mel_spec_std", np.std(mel_spec))

    mel_db = librosa.power_to_db(mel_spec)
    mfcc = feature_mfcc(S=mel_db, sr=sr, n_mfcc=20)
    for i in range(mfcc.shape[0]):
        push(f"mfcc_{i}_mean", np.mean(mfcc[i]))
        push(f"mfcc_{i}_std", np.std(mfcc[i]))

    # 6) Tonnetz and HPSS
    # Tonnetz from chroma (avoid additional transforms)
    tnx = feature_tonnetz(chroma=chroma_cqt, sr=sr)
    for i in range(tnx.shape[0]):
        push(f"tonnetz_{i}_mean", np.mean(tnx[i]))
        push(f"tonnetz_{i}_std", np.std(tnx[i]))

    # HPSS on the spectrogram (avoid extra STFTs and iSTFT)
    M_harm, M_perc = librosa_decompose.hpss(S_mag)
    e_harm = float(np.sum(M_harm**2))
    e_perc = float(np.sum(M_perc**2))
    e_tot = e_harm + e_perc
    push("hpss_harm_ratio", (e_harm / e_tot) if e_tot > 0 else 0.0)
    push("hpss_perc_ratio", (e_perc / e_tot) if e_tot > 0 else 0.0)

    # Return vector and names (order consistent with insertion)
    return np.asarray(feature_values, dtype=float), feature_names


def _extract_single(args):
    idx, filename, file_path, subfolder = args
    try:
        features, names = extract_features(file_path)
        return idx, filename, subfolder, features, names, None
    except Exception as e:
        return idx, filename, subfolder, None, None, str(e)


essential_exts = {".mp3", ".wav", ".flac", ".ogg"}


def extract_features_from_directory(
    directory, extensions=None, n_jobs: Optional[int] = 1
):
    if extensions is None:
        extensions = essential_exts
    # Normalize n_jobs (at least 1)
    try:
        n_jobs = int(n_jobs) if n_jobs is not None else 1
    except (TypeError, ValueError):
        n_jobs = 1
    n_jobs = max(1, n_jobs)

    feature_list = []
    filenames = []
    dir_names = []
    feature_names = None

    # Collect files and their subfolder label
    candidates = []
    for dirpath, _, files in os.walk(directory):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(dirpath, filename)
                subfolder = os.path.basename(dirpath)
                candidates.append((filename, file_path, subfolder))

    if not candidates:
        raise ValueError("No audio files found in the directory or subfolders")

    # Sort for stability
    candidates.sort(key=lambda x: (x[2].lower(), x[0].lower()))

    if n_jobs == 1:
        # Serial
        for filename, file_path, subfolder in candidates:
            try:
                print(f"Processing {filename} (folder: {subfolder})...")
                features, names = extract_features(file_path)
                if not isinstance(features, np.ndarray) or len(features.shape) != 1:
                    print(f"Error: invalid features for {filename}, skipping")
                    continue
                numeric_feats = features.astype(float)
                if np.isnan(numeric_feats).any() or np.isinf(numeric_feats).any():
                    print(f"Error: features contain NaN/Inf in {filename}, skipping")
                    continue
                if feature_names is None:
                    feature_names = names
                feature_list.append(features)
                filenames.append(filename)
                dir_names.append(str(subfolder))
                print(f"{len(filenames)}-Processed: {filename}")
            except Exception as e:
                print(f"Error with {filename}: {e}")
    else:
        # Parallel with order preserved
        items = [(i, fn, fp, sf) for i, (fn, fp, sf) in enumerate(candidates)]
        print(f"Starting parallel extraction with {n_jobs} workers...")
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            for idx, filename, subfolder, features, names, err in ex.map(
                _extract_single, items, chunksize=1
            ):
                if err is not None or features is None:
                    print(f"Error with {filename}: {err}")
                    continue
                if not isinstance(features, np.ndarray) or len(features.shape) != 1:
                    print(f"Error: invalid features for {filename}, skipping")
                    continue
                numeric_feats = features.astype(float)
                if np.isnan(numeric_feats).any() or np.isinf(numeric_feats).any():
                    print(f"Error: features contain NaN/Inf in {filename}, skipping")
                    continue
                if feature_names is None:
                    feature_names = names
                feature_list.append(features)
                filenames.append(filename)
                dir_names.append(str(subfolder))
                print(f"{len(filenames)}-Processed: {filename}")

    if not feature_list:
        raise ValueError("No valid features extracted from audio files")

    feature_array = np.vstack(feature_list)
    return filenames, dir_names, feature_array, feature_names


def save_features_to_csv(
    filenames, file_dirs, feature_array, feature_names, csv_filename
):
    header = ["filename", "filedir"] + feature_names
    with open(csv_filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fname, fdir, feats in zip(filenames, file_dirs, feature_array):
            writer.writerow([fname] + [fdir] + list(feats))


def read_features_from_csv(csv_filename):
    filenames = []
    feature_list = []
    file_dirs = []
    with open(csv_filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            filenames.append(row[0])
            file_dirs.append(row[1])
            features = np.array([float(x) for x in row[2:]])
            feature_list.append(features)
    feature_array = np.vstack(feature_list)
    feature_names = header[2:]
    return filenames, file_dirs, feature_array, feature_names


def get_audio_features(audio_dir, csv_filename, n_jobs: int = 1):
    if os.path.exists(csv_filename):
        print(f"Loading features from {csv_filename}")
        filenames, file_dirs, features, feature_names = read_features_from_csv(
            csv_filename
        )
        return filenames, file_dirs, features, feature_names
    else:
        print("Extracting features from audio files...")
        filenames, file_dirs, features, feature_names = extract_features_from_directory(
            audio_dir, n_jobs=n_jobs
        )
        save_features_to_csv(
            filenames, file_dirs, features, feature_names, csv_filename
        )
        print(f"Features saved to {csv_filename}")
        return filenames, file_dirs, features, feature_names
