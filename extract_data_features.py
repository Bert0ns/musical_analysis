import os
import numpy as np
import librosa
import csv


def extract_features(file_path):
    y, sr = librosa.load(file_path, mono=True, duration=120)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_mean = np.mean(spectral_contrast, axis=1)

    zero_crossings = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossings)

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)

    rmse = librosa.feature.rms(y=y)
    rmse_mean = np.mean(rmse)

    feature_vector = np.concatenate([
        mfccs_mean,                # 13
        chroma_mean,               # 12
        spectral_contrast_mean,    # 7
        [zero_crossing_rate_mean], # 1
        [spectral_centroid_mean],  # 1
        [spectral_bandwidth_mean], # 1
        [rolloff_mean],            # 1
        [rmse_mean]                # 1
    ])
    return feature_vector

def extract_features_from_directory(directory, extensions=None):
    if extensions is None:
        extensions = {'.mp3', '.wav', '.flac', '.ogg'}
    feature_list = []
    filenames = []
    for filename in os.listdir(directory):
        if any(filename.lower().endswith(ext) for ext in extensions):
            file_path = os.path.join(directory, filename)
            try:
                features = extract_features(file_path)
                feature_list.append(features)
                filenames.append(filename)
                print(f"{len(filenames)}-Elaborato: {filename}")
            except Exception as e:
                print(f"{len(filenames)}-Errore con {filename}: {e}")
    feature_array = np.vstack(feature_list)
    return filenames, feature_array

def save_features_to_csv(filenames, feature_array, csv_filename):
    header = ["filename"] + [f"f{i+1}" for i in range(feature_array.shape[1])]
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
    else:
        print("Estrazione feature dai file audio...")
        filenames, features = extract_features_from_directory(audio_dir)
        save_features_to_csv(filenames, features, csv_filename)
        print(f"Feature salvate in {csv_filename}")
    return filenames, features
