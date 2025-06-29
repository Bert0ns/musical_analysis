import csv
import os

import librosa
import numpy as np


def extract_features(file_path):
    """
        Estrae un insieme completo di caratteristiche audio da un file.
        Restituisce un vettore di feature con etichette per identificarle.
        """
    # Caricamento audio
    y, sr = librosa.load(file_path, mono=True, duration=None)

    # Durata del brano in secondi
    duration = librosa.get_duration(y=y, sr=sr)

    # Dizionario per memorizzare tutte le feature
    features = {}
    feature_names = []

    # === CARATTERISTICHE TEMPORALI ===

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    features['zcr_max'] = np.max(zcr)
    feature_names.extend(['zcr_mean', 'zcr_std', 'zcr_max'])

    # RMS (energia)
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    features['rms_max'] = np.max(rms)
    feature_names.extend(['rms_mean', 'rms_std', 'rms_max'])

    # Tempo e battiti
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    features['tempo'] = tempo
    features['beats_count'] = len(beats) / duration  # Battiti per secondo
    feature_names.extend(['tempo', 'beats_count'])

    # === CARATTERISTICHE SPETTRALI ===

    # Centroide spettrale
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['centroid_mean'] = np.mean(spectral_centroid)
    features['centroid_std'] = np.std(spectral_centroid)
    feature_names.extend(['centroid_mean', 'centroid_std'])

    # Larghezza di banda spettrale
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features['bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['bandwidth_std'] = np.std(spectral_bandwidth)
    feature_names.extend(['bandwidth_mean', 'bandwidth_std'])

    # Rolloff spettrale
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['rolloff_mean'] = np.mean(rolloff)
    features['rolloff_std'] = np.std(rolloff)
    feature_names.extend(['rolloff_mean', 'rolloff_std'])

    # Contrasto spettrale (7 bande)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    for i in range(contrast.shape[0]):
        features[f'contrast_{i}_mean'] = np.mean(contrast[i])
        features[f'contrast_{i}_std'] = np.std(contrast[i])
        feature_names.extend([f'contrast_{i}_mean', f'contrast_{i}_std'])

    # Piattezza spettrale
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    features['flatness_mean'] = np.mean(flatness)
    features['flatness_std'] = np.std(flatness)
    feature_names.extend(['flatness_mean', 'flatness_std'])

    # Polyfeatures (coefficienti polinomiali dello spettro)
    poly_features = librosa.feature.poly_features(y=y, sr=sr, order=2)
    for i in range(poly_features.shape[0]):
        features[f'poly_{i}_mean'] = np.mean(poly_features[i])
        features[f'poly_{i}_std'] = np.std(poly_features[i])
        feature_names.extend([f'poly_{i}_mean', f'poly_{i}_std'])

    # Tonnetz (caratteristiche tonali)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    for i in range(tonnetz.shape[0]):
        features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])
        features[f'tonnetz_{i}_std'] = np.std(tonnetz[i])
        feature_names.extend([f'tonnetz_{i}_mean', f'tonnetz_{i}_std'])

    # === CARATTERISTICHE CROMATICHE ===

    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(chroma_stft.shape[0]):
        features[f'chroma_stft_{i}_mean'] = np.mean(chroma_stft[i])
        features[f'chroma_stft_{i}_std'] = np.std(chroma_stft[i])
        feature_names.extend([f'chroma_stft_{i}_mean', f'chroma_stft_{i}_std'])

    # Chroma CQT (corretto)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    for i in range(chroma_cqt.shape[0]):
        features[f'chroma_cqt_{i}_mean'] = np.mean(chroma_cqt[i])
        features[f'chroma_cqt_{i}_std'] = np.std(chroma_cqt[i])
        feature_names.extend([f'chroma_cqt_{i}_mean', f'chroma_cqt_{i}_std'])

    # Chroma CENS
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    for i in range(chroma_cens.shape[0]):
        features[f'chroma_cens_{i}_mean'] = np.mean(chroma_cens[i])
        features[f'chroma_cens_{i}_std'] = np.std(chroma_cens[i])
        feature_names.extend([f'chroma_cens_{i}_mean', f'chroma_cens_{i}_std'])

    # === CARATTERISTICHE MEL ===

    # MFCCs (20 coefficienti)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for i in range(mfccs.shape[0]):
        features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
        features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        feature_names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_std'])

    # Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    features['mel_spec_mean'] = np.mean(mel_spec)
    features['mel_spec_std'] = np.std(mel_spec)
    feature_names.extend(['mel_spec_mean', 'mel_spec_std'])

    # Crea vettore ordinato di feature assicurandosi che siano tutti valori scalari
    feature_vector = []
    for name in feature_names:
        value = features[name]
        # Verifica che il valore sia scalare
        if hasattr(value, 'shape') and value.shape != ():
            print(f"Attenzione: {name} non è scalare, uso la media")
            value = float(np.mean(value))
        elif not np.isscalar(value):
            print(f"Attenzione: {name} non è scalare, conversione a float")
            value = float(value)

        # Gestione di valori NaN o infiniti
        if np.isnan(value) or np.isinf(value):
            print(f"Attenzione: {name} contiene NaN/Inf, sostituito con 0")
            value = 0.0

        feature_vector.append(value)

    return np.array(feature_vector, dtype=float), feature_names


def extract_features_from_directory(directory, extensions=None):
    if extensions is None:
        extensions = {'.mp3', '.wav', '.flac', '.ogg'}
    feature_list = []
    filenames = []
    feature_names = None
    expected_length = None

    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(directory, filename)
            try:
                print(f"Elaborazione di {filename}...")
                features, names = extract_features(file_path)

                # Verifica che features sia un array unidimensionale di float
                if not isinstance(features, np.ndarray) or len(features.shape) != 1:
                    print(f"Errore: feature non valide per {filename}, skip")
                    continue

                # Verifica che non ci siano NaN o infiniti
                if np.isnan(features).any() or np.isinf(features).any():
                    print(f"Errore: feature con NaN/Inf in {filename}, skip")
                    continue

                if feature_names is None:
                    feature_names = names

                feature_list.append(features)
                filenames.append(filename)
                print(f"{len(filenames)}-Elaborato: {filename}")

            except Exception as e:
                print(f"Errore con {filename}: {e}")

    if not feature_list:
        raise ValueError("Nessuna feature valida estratta dai file audio")

    # Debug prima di vstack
    print(f"Creazione matrice di {len(feature_list)} vettori di feature...")

    feature_array = np.vstack(feature_list)
    return filenames, feature_array, feature_names


def save_features_to_csv(filenames, feature_array, feature_names, csv_filename):
    header = ["filename"] + feature_names
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for fname, feats in zip(filenames, feature_array):
            writer.writerow([fname] + list(feats))


def read_features_from_csv(csv_filename):
    filenames = []
    feature_list = []
    with open(csv_filename, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            filenames.append(row[0])
            features = np.array([float(x) for x in row[1:]])
            feature_list.append(features)
    feature_array = np.vstack(feature_list)
    return filenames, feature_array


def get_audio_features(audio_dir, csv_filename):
    if os.path.exists(csv_filename):
        print(f"Caricamento feature da {csv_filename}")
        filenames, features = read_features_from_csv(csv_filename)
        return filenames, features
    else:
        print("Estrazione feature dai file audio...")
        filenames, features, feature_names = extract_features_from_directory(audio_dir)
        save_features_to_csv(filenames, features, feature_names, csv_filename)
        print(f"Feature salvate in {csv_filename}")
        return filenames, features
