import csv
import os

import librosa
import numpy as np
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
    delta as feature_delta,
    tonnetz as feature_tonnetz,
)
from librosa.effects import trim as effects_trim, hpss as effects_hpss
from librosa.onset import onset_strength as onset_strength, onset_detect as onset_detect
from librosa.beat import beat_track as beat_track
from librosa import yin as yin, note_to_hz as note_to_hz


def extract_features(file_path):
    """
    Estrae un insieme ampio e robusto di caratteristiche audio da un file usando librosa.

    Contratto rapido:
    - Input: percorso del file audio (stringa)
    - Output: (feature_vector: np.ndarray[float64], feature_names: list[str])
    - Proprietà: nessun NaN/Inf, solo scalari, ordine stabile dei nomi

    Strategia:
    1) Carica audio in mono con sample rate fisso per coerenza tra i file; rimuove silenzio iniziale/finale.
    2) Calcola feature temporali (ZCR, RMS, onset rate, tempo/beat).
    3) Calcola feature spettrali (centroid, bandwidth, rolloff 85/95, contrasto, flatness, poly features).
    4) Calcola feature cromatiche (chroma STFT/CQT/CENS).
    5) Calcola feature su mel/MFCC (MFCC, delta, delta-delta).
    6) Calcola tonnetz (caratteristiche tonali) e rapporto energia armonico/percussivo (HPSS).
    7) Stima fondamentale (F0) con YIN e statistiche.

    Tutti i risultati sono ridotti a scalari (mean/std), depurati da NaN/Inf e restituiti in array.
    """
    # Parametri consistenti per l'analisi
    sr_target = 22050
    n_fft = 2048
    hop_length = 512

    # Helper per push sicuro di valori scalari in uscita
    feature_names: list[str] = []
    feature_values: list[float] = []

    def _to_float_scalar(x) -> float:
        # Riduce qualsiasi valore/array a float scalare, con fallback 0 per NaN/Inf
        try:
            if isinstance(x, (list, tuple, np.ndarray)):
                x = np.asarray(x)
                # Se non scalare, usa la media
                if x.shape != ():
                    x = np.nanmean(x)
            x = float(x)
        except Exception:
            x = 0.0
        # Sanifica NaN/Inf
        if np.isnan(x) or np.isinf(x):
            return 0.0
        return x

    def push(name: str, value) -> None:
        feature_names.append(name)
        feature_values.append(_to_float_scalar(value))

    # 1) Caricamento audio con SR fisso e trim del silenzio
    y, sr = librosa.load(file_path, sr=sr_target, mono=True, res_type="kaiser_fast")
    if y is None or len(y) == 0:
        raise ValueError(f"Audio vuoto o non leggibile: {file_path}")

    # Rimuove offset DC e trim del silenzio
    y = y - np.mean(y)
    y_trimmed, idx = effects_trim(y, top_db=30)
    if len(y_trimmed) > 0:
        y = y_trimmed
    duration = max(1e-9, len(y) / sr)  # evita divisioni per zero

    # 2) Feature temporali
    zcr = zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    push("zcr_mean", np.mean(zcr))
    push("zcr_std", np.std(zcr))
    push("zcr_max", np.max(zcr) if zcr.size else 0.0)

    rms = feature_rms(y=y, frame_length=n_fft, hop_length=hop_length, center=True)[0]
    push("rms_mean", np.mean(rms))
    push("rms_std", np.std(rms))
    push("rms_max", np.max(rms) if rms.size else 0.0)

    # Onset rate (attacchi al secondo)
    onset_env = onset_strength(y=y, sr=sr, hop_length=hop_length)
    onsets = onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, units="frames")
    push("onset_rate", (len(onsets) / duration))

    # Tempo (BPM) e battiti al secondo
    tempo, beats = beat_track(y=y, sr=sr, hop_length=hop_length)
    push("tempo_bpm", tempo)
    push("beats_per_sec", (len(beats) / duration))

    # 3) Feature spettrali
    spectral_centroid = feature_spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    push("centroid_mean", np.mean(spectral_centroid))
    push("centroid_std", np.std(spectral_centroid))

    spectral_bandwidth = feature_spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    push("bandwidth_mean", np.mean(spectral_bandwidth))
    push("bandwidth_std", np.std(spectral_bandwidth))

    rolloff85 = feature_spectral_rolloff(y=y, sr=sr, roll_percent=0.85, n_fft=n_fft, hop_length=hop_length)[0]
    rolloff95 = feature_spectral_rolloff(y=y, sr=sr, roll_percent=0.95, n_fft=n_fft, hop_length=hop_length)[0]
    push("rolloff85_mean", np.mean(rolloff85))
    push("rolloff85_std", np.std(rolloff85))
    push("rolloff95_mean", np.mean(rolloff95))
    push("rolloff95_std", np.std(rolloff95))

    contrast = feature_spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(contrast.shape[0]):
        push(f"contrast_{i}_mean", np.mean(contrast[i]))
        push(f"contrast_{i}_std", np.std(contrast[i]))

    flatness = feature_spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    push("flatness_mean", np.mean(flatness))
    push("flatness_std", np.std(flatness))

    # Poly features (fino a ordine 2)
    poly = feature_poly_features(y=y, sr=sr, order=2, n_fft=n_fft, hop_length=hop_length)
    for i in range(poly.shape[0]):
        push(f"poly_{i}_mean", np.mean(poly[i]))
        push(f"poly_{i}_std", np.std(poly[i]))

    # 4) Feature cromatiche
    chroma_stft = feature_chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    for i in range(chroma_stft.shape[0]):
        push(f"chroma_stft_{i}_mean", np.mean(chroma_stft[i]))
        push(f"chroma_stft_{i}_std", np.std(chroma_stft[i]))

    chroma_cqt = feature_chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    for i in range(chroma_cqt.shape[0]):
        push(f"chroma_cqt_{i}_mean", np.mean(chroma_cqt[i]))
        push(f"chroma_cqt_{i}_std", np.std(chroma_cqt[i]))

    chroma_cens = feature_chroma_cens(y=y, sr=sr, hop_length=hop_length)
    for i in range(chroma_cens.shape[0]):
        push(f"chroma_cens_{i}_mean", np.mean(chroma_cens[i]))
        push(f"chroma_cens_{i}_std", np.std(chroma_cens[i]))

    # 5) Mel/MFCC
    mel_spec = feature_melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, power=2.0)
    push("mel_spec_mean", np.mean(mel_spec))
    push("mel_spec_std", np.std(mel_spec))

    mfcc = feature_mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop_length)
    for i in range(mfcc.shape[0]):
        push(f"mfcc_{i}_mean", np.mean(mfcc[i]))
        push(f"mfcc_{i}_std", np.std(mfcc[i]))

    # Delta e Delta-Delta dei MFCC
    try:
        mfcc_delta = feature_delta(mfcc, order=1)
        mfcc_delta2 = feature_delta(mfcc, order=2)
        for i in range(mfcc.shape[0]):
            push(f"mfcc_delta_{i}_mean", np.mean(mfcc_delta[i]))
            push(f"mfcc_delta_{i}_std", np.std(mfcc_delta[i]))
            push(f"mfcc_delta2_{i}_mean", np.mean(mfcc_delta2[i]))
            push(f"mfcc_delta2_{i}_std", np.std(mfcc_delta2[i]))
    except Exception:
        # In caso di problemi, inserisci placeholder 0 per mantenere dimensione costante
        for i in range(mfcc.shape[0]):
            push(f"mfcc_delta_{i}_mean", 0.0)
            push(f"mfcc_delta_{i}_std", 0.0)
            push(f"mfcc_delta2_{i}_mean", 0.0)
            push(f"mfcc_delta2_{i}_std", 0.0)

    # 6) Tonnetz e HPSS
    try:
        tonnetz = feature_tonnetz(y=y, sr=sr)
        for i in range(tonnetz.shape[0]):
            push(f"tonnetz_{i}_mean", np.mean(tonnetz[i]))
            push(f"tonnetz_{i}_std", np.std(tonnetz[i]))
    except Exception:
        # Tonnetz ha tipicamente 6 dimensioni
        for i in range(6):
            push(f"tonnetz_{i}_mean", 0.0)
            push(f"tonnetz_{i}_std", 0.0)

    y_harm, y_perc = effects_hpss(y)
    e_harm = float(np.sum(y_harm ** 2))
    e_perc = float(np.sum(y_perc ** 2))
    e_tot = e_harm + e_perc
    push("hpss_harm_ratio", (e_harm / e_tot) if e_tot > 0 else 0.0)
    push("hpss_perc_ratio", (e_perc / e_tot) if e_tot > 0 else 0.0)

    # 7) Stima F0 (YIN)
    try:
        fmin = note_to_hz("C2")
        fmax = note_to_hz("C7")
        f0 = yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=n_fft, hop_length=hop_length)
        # f0 può contenere NaN per frame non intonati
        push("f0_mean", np.nanmean(f0))
        push("f0_std", np.nanstd(f0))
        voiced_frac = np.mean(~np.isnan(f0)) if f0.size else 0.0
        push("f0_voiced_fraction", voiced_frac)
    except Exception:
        push("f0_mean", 0.0)
        push("f0_std", 0.0)
        push("f0_voiced_fraction", 0.0)

    # Info di durata utili
    push("duration_sec", duration)

    # Ritorna vettore e nomi (ordine coerente con l'inserimento)
    return np.asarray(feature_values, dtype=float), feature_names


def extract_features_from_directory(directory, extensions=None):
    if extensions is None:
        extensions = {'.mp3', '.wav', '.flac', '.ogg'}
    feature_list = []
    filenames = []
    dir_names = []
    feature_names = None

    # 1) Raccogli i file (anche in sottocartelle) e la loro etichetta di sottocartella
    candidates = []
    for dirpath, _, files in os.walk(directory):
        for filename in files:
            if any(filename.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(dirpath, filename)
                # Nome della sottocartella che contiene il file (o __root__ per la radice)
                if os.path.abspath(dirpath) == os.path.abspath(directory):
                    subfolder = dirpath.split("/")[-1]
                else:
                    subfolder = os.path.basename(dirpath)
                candidates.append((filename, file_path, subfolder))

    if not candidates:
        raise ValueError("Nessun file audio trovato nella directory o sottocartelle")

    # Ordina per stabilità
    candidates.sort(key=lambda x: (x[2].lower(), x[0].lower()))

    # 3) Elabora i file e aggiungi la singola feature stringa della sottocartella
    for filename, file_path, subfolder in candidates:
        try:
            print(f"Elaborazione di {filename} (cartella: {subfolder})...")
            features, names = extract_features(file_path)

            # Verifica che features sia un array unidimensionale di float
            if not isinstance(features, np.ndarray) or len(features.shape) != 1:
                print(f"Errore: feature non valide per {filename}, skip")
                continue

            numeric_feats = features.astype(float)

            # Verifica che non ci siano NaN o infiniti nelle sole feature numeriche
            if np.isnan(numeric_feats).any() or np.isinf(numeric_feats).any():
                print(f"Errore: feature con NaN/Inf in {filename}, skip")
                continue


            if feature_names is None:
                feature_names = names

            feature_list.append(features)
            filenames.append(filename)
            dir_names.append(str(subfolder))
            print(f"{len(filenames)}-Elaborato: {filename}")

        except Exception as e:
            print(f"Errore con {filename}: {e}")

    if not feature_list:
        raise ValueError("Nessuna feature valida estratta dai file audio")

    feature_array = np.vstack(feature_list)
    return filenames, dir_names, feature_array, feature_names


def save_features_to_csv(filenames, file_dirs, feature_array, feature_names, csv_filename):
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
    with open(csv_filename, "r", encoding='utf-8', newline="") as f:
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


def get_audio_features(audio_dir, csv_filename):
    if os.path.exists(csv_filename):
        print(f"Caricamento feature da {csv_filename}")
        filenames, file_dirs, features, feature_names = read_features_from_csv(csv_filename)
        return filenames, file_dirs, features, feature_names
    else:
        print("Estrazione feature dai file audio...")
        filenames, file_dirs, features, feature_names = extract_features_from_directory(audio_dir)
        save_features_to_csv(filenames, file_dirs, features, feature_names, csv_filename)
        print(f"Feature salvate in {csv_filename}")
        return filenames, file_dirs, features, feature_names
