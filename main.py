import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from lib.extract_data_features import get_audio_features
from lib.dbscan_clustering import run_dbscan_clustering_pipeline
from lib.k_means_clustering import run_kmeans_clustering_pipeline
from lib.spectral_clustering import run_spectral_clustering_pipeline

CSV_FEATURE_FILENAME = "dataset/songs_features/songs_features_all.csv"
SONGS_DIR = "dataset/songs/trap"  # Cambia con il percorso della tua cartella di canzoni

RESULTS_SC = "clustering_results/spectral_clustering"
RESULTS_KM = "clustering_results/kmeans"
RESULTS_DBSCAN = "clustering_results/dbscan"

N_CLUSTERS = 5  # Numero di cluster da creare
PCA_COMPONENTS = 0.98  # Percentuale di varianza da mantenere con PCA

SPECTRAL_CLUSTERING_GAMMA = 0.2  # Parametro gamma per lo spectral clustering

# Parametri DBSCAN (valori di default, puoi regolarli dopo aver guardato i grafici)
DBSCAN_EPS = 0.7
DBSCAN_MIN_SAMPLES = 8
DBSCAN_METRIC = 'euclidean'


if __name__ == "__main__":
    print("Caricamento delle feature audio...")
    filenames, music_genres, features, features_names = get_audio_features(SONGS_DIR, CSV_FEATURE_FILENAME)
    print("Shape feature array:", features.shape)

    # Rimozione dei duplicati
    features, unique_indices = np.unique(features, axis=0, return_index=True)
    filenames = [filenames[i] for i in unique_indices]
    music_genres = [music_genres[i] for i in unique_indices]
    print("Shape feature array dopo rimozione duplicati:", features.shape)

    # Normalizzazione delle feature (MinMax: range 0-1)
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)

    # Copia per report dettagliato (prima della PCA)
    features_norm_original = features_norm.copy()

    # Riduzione della dimensionalità con PCA
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='full')
    features_reduced = pca.fit_transform(features_norm)
    print("Shape feature array dopo PCA:", features_reduced.shape)

    # =========================== SPECTRAL CLUSTERING ===========================
    spectral_labels = run_spectral_clustering_pipeline(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        results_dir=RESULTS_SC,
        n_clusters=N_CLUSTERS,
        gamma=SPECTRAL_CLUSTERING_GAMMA,
    )

    # ============================= K-MEANS CLUSTERING =============================
    kmeans_labels, kmeans_centers = run_kmeans_clustering_pipeline(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        results_dir=RESULTS_KM,
        n_clusters=N_CLUSTERS,
    )

    # ============================= DBSCAN CLUSTERING =============================
    dbscan_labels, dbscan_model = run_dbscan_clustering_pipeline(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        results_dir=RESULTS_DBSCAN,
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES,
        metric=DBSCAN_METRIC,
    )

    print("\nPipeline completata (Spectral + K-Means + DBSCAN)")

