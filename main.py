import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lib.extract_data_features import get_audio_features
from lib.dbscan_clustering import run_dbscan_clustering_pipeline
from lib.k_means_clustering import run_kmeans_clustering_pipeline
from lib.spectral_clustering import run_spectral_clustering_pipeline
from lib.extract_msd_h5_features import get_msd_h5_features

CSV_FEATURE_FILENAME = "dataset/songs_features/songs_features_all.csv"
SONGS_DIR = "dataset/songs"  # Cambia con il percorso della tua cartella di canzoni

RESULTS_SC = "clustering_results/spectral_clustering"
RESULTS_KM = "clustering_results/kmeans"
RESULTS_DBSCAN = "clustering_results/dbscan"

N_CLUSTERS = 5  # Numero di cluster da creare
SPECTRAL_CLUSTERING_GAMMA = 0.2  # Parametro gamma per lo spectral clustering

# Parametri DBSCAN (valori di default, puoi regolarli dopo aver guardato i grafici)
DBSCAN_EPS = 0.7
DBSCAN_MIN_SAMPLES = 8
DBSCAN_METRIC = 'euclidean'

# ===================== PARAM GRID (valori esempio, modifica liberamente) =====================
import os
import csv
from lib.utils import computer_clustering_scores, _param_product, _fmt_float

PCA_COMPONENTS = 0.98  # Percentuale di varianza da mantenere con PCA

SPECTRAL_PARAM_GRID = {
    'n_clusters': [3, 5, 7],
    'gamma': [0.1, 0.2, 0.3, 0.5],
}

KMEANS_PARAM_GRID = {
    'n_clusters': [3, 4, 5, 6, 7],
}

DBSCAN_PARAM_GRID = {
    'eps': [0.031, 0.034, 0.038],
    'min_samples': [5, 8, 12, 18],
    'metric': ['euclidean'], # 'cosine'
}


# --- Helper: scegli lo spazio per DBSCAN
def _choose_dbscan_space(features_reduced: np.ndarray, features_norm: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'normalized':
        # Usa le feature normalizzate pre-PCA
        return features_norm
    elif mode == 'reduced_minmax':
        # Riapplica MinMax alle componenti PCA
        return MinMaxScaler().fit_transform(features_reduced)
    else:  # 'reduced'
        # Usa direttamente le componenti PCA
        return features_reduced


def grid_search_spectral(
    filenames,
    features_reduced,
    features_norm_original,
    features_names,
    music_genres,
    base_results_dir: str = RESULTS_SC,
    param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = SPECTRAL_PARAM_GRID
    os.makedirs(base_results_dir, exist_ok=True)

    summary_path = os.path.join(base_results_dir, "grid_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["n_clusters", "gamma", "n_clusters_found", "silhouette", "davies_bouldin", "results_dir"])
        for params in _param_product(param_grid):
            n_clusters = params['n_clusters']
            gamma = params['gamma']
            run_dir = os.path.join(base_results_dir, f"grid_k{n_clusters}_g{_fmt_float(gamma)}")
            labels = run_spectral_clustering_pipeline(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                results_dir=run_dir,
                n_clusters=n_clusters,
                gamma=gamma,
            )
            unique_cls = len(np.unique(labels))
            sil, dbi = computer_clustering_scores(features_reduced, labels)
            writer.writerow([n_clusters, gamma, unique_cls, f"{sil:.6f}", f"{dbi:.6f}", run_dir])
    print(f"Riepilogo grid Spectral scritto in: {summary_path}")


def grid_search_kmeans(
    filenames,
    features_reduced,
    features_norm_original,
    features_names,
    music_genres,
    base_results_dir: str = RESULTS_KM,
    param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = KMEANS_PARAM_GRID
    os.makedirs(base_results_dir, exist_ok=True)
    summary_path = os.path.join(base_results_dir, "grid_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["n_clusters", "n_clusters_found", "silhouette", "davies_bouldin", "results_dir"])
        for params in _param_product(param_grid):
            n_clusters = params['n_clusters']
            run_dir = os.path.join(base_results_dir, f"grid_k{n_clusters}")
            labels, _ = run_kmeans_clustering_pipeline(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                results_dir=run_dir,
                n_clusters=n_clusters,
            )
            unique_cls = len(np.unique(labels))
            sil, dbi = computer_clustering_scores(features_reduced, labels)
            writer.writerow([n_clusters, unique_cls, f"{sil:.6f}", f"{dbi:.6f}", run_dir])
    print(f"Riepilogo grid K-Means scritto in: {summary_path}")


def grid_search_dbscan(
    filenames,
    features_reduced,
    features_norm_original,
    features_names,
    music_genres,
    base_results_dir: str = RESULTS_DBSCAN,
    param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = DBSCAN_PARAM_GRID
    os.makedirs(base_results_dir, exist_ok=True)
    summary_path = os.path.join(base_results_dir, "grid_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["eps", "min_samples", "metric", "n_clusters_found", "noise_ratio", "silhouette_non_noise", "davies_bouldin_non_noise", "results_dir"])
        for params in _param_product(param_grid):
            eps = params['eps']
            min_samples = params['min_samples']
            metric = params['metric']
            run_dir = os.path.join(base_results_dir, f"grid_eps{_fmt_float(eps)}_min{min_samples}_{metric}")
            labels, _ = run_dbscan_clustering_pipeline(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                results_dir=run_dir,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
            )
            unique_valid = [c for c in np.unique(labels) if c != -1]
            noise_ratio = float(np.sum(labels == -1)) / float(len(labels)) if len(labels) else 0.0
            if len(unique_valid) >= 2:
                mask = labels != -1
                sil, dbi = computer_clustering_scores(features_reduced[mask], labels[mask])
                sil_s = f"{sil:.6f}"; dbi_s = f"{dbi:.6f}"
            else:
                sil_s = ""
                dbi_s = ""
            writer.writerow([eps, min_samples, metric, len(unique_valid), f"{noise_ratio:.6f}", sil_s, dbi_s, run_dir])
    print(f"Riepilogo grid DBSCAN scritto in: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clustering musicale - esecuzione singola o grid search parametri")
    parser.add_argument('--mode', choices=['single', 'grid'], default='single', help='single: esegue una volta con i parametri di default; grid: testa combinazioni di parametri')
    parser.add_argument('--which', nargs='*', choices=['spectral', 'kmeans', 'dbscan'], help='Se in modalità grid, limita agli algoritmi indicati')
    parser.add_argument(
        '--dbscan-space',
        choices=['reduced', 'reduced_minmax', 'normalized'],
        default='reduced_minmax',
        help='Spazio feature usato da DBSCAN'
    )
    # Scelta dello scaler per la normalizzazione delle feature
    parser.add_argument('--scaler', choices=['minmax', 'standard'], default='minmax', help='Seleziona lo scaler per la normalizzazione: minmax (default) o standard')
    # Nuovo: numero di processi per l'estrazione feature audio
    parser.add_argument('--workers', type=int, default=1, help='Numero di processi per estrazione feature audio (default: 1)')
    # NUOVI ARGOMENTI per usare le feature del Million Song Dataset
    parser.add_argument('--feature-source', choices=['audio', 'msd'], default='audio', help='Origine delle feature: audio locale (librosa) oppure msd (file .h5)')
    parser.add_argument('--msd-root', type=str, default=None, help='Cartella radice dei file .h5 MSD')
    parser.add_argument('--msd-csv', type=str, default='dataset/songs_features/msd_h5_features.csv', help='CSV cache per feature MSD')
    parser.add_argument('--msd-titles-file', type=str, default=None, help='File mapping track_id<SEP>song_id<SEP>artist_name<SEP>song_title')
    parser.add_argument('--msd-max-files', type=int, default=None, help='Limita numero di file .h5 (debug)')
    args = parser.parse_args()

    # ================= CARICAMENTO FEATURE =================
    if args.feature_source == 'audio':
        print("Caricamento delle feature audio (estrazione locale/librosa)...")
        filenames, music_genres, features, features_names = get_audio_features(SONGS_DIR, CSV_FEATURE_FILENAME, n_jobs=args.workers)
        source_label = 'audio'
    else:
        if args.msd_root is None and args.msd_csv is None:
            raise ValueError("Per usare le feature MSD, specificare --msd-root e/o --msd-csv")

        print("Caricamento / estrazione feature MSD (.h5)...")
        filenames, artist_names, features, features_names = get_msd_h5_features(
            args.msd_root,
            args.msd_csv,
            max_files=args.msd_max_files,
            verbose=False,
            titles_file=args.msd_titles_file,
        )
        # Riutilizziamo lo slot music_genres per compatibilità pipeline (report) usando l'artista
        music_genres = [a if a else 'UNKNOWN' for a in artist_names]
        source_label = 'msd'
    print(f"Origine feature: {source_label}")
    print("Shape feature array:", features.shape)

    # Rimozione dei duplicati
    features, unique_indices = np.unique(features, axis=0, return_index=True)
    filenames = [filenames[i] for i in unique_indices]
    music_genres = [music_genres[i] for i in unique_indices]
    print("Shape feature array dopo rimozione duplicati:", features.shape)

    # Normalizzazione delle feature con scaler selezionabile
    if args.scaler == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    print(f"Scaler selezionato: {scaler.__class__.__name__}")
    features_norm = scaler.fit_transform(features)

    # Copia per report dettagliato (prima della PCA)
    features_norm_original = features_norm.copy()

    # Riduzione della dimensionalità con PCA
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='full')
    features_reduced = pca.fit_transform(features_norm)
    print("Shape feature array dopo PCA:", features_reduced.shape)

    if args.mode == 'single':
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
            _choose_dbscan_space(features_reduced, features_norm_original, args.dbscan_space),
            features_norm_original,
            features_names,
            music_genres,
            results_dir=RESULTS_DBSCAN,
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric=DBSCAN_METRIC,
        )

        print("\nPipeline completata (Spectral + K-Means + DBSCAN)")
    else:
        which = set(args.which) if args.which else {'spectral', 'kmeans', 'dbscan'}
        if 'spectral' in which:
            print("\n[GRID] Avvio grid search per Spectral Clustering...")
            grid_search_spectral(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                base_results_dir=RESULTS_SC,
                param_grid=SPECTRAL_PARAM_GRID,
            )
        if 'kmeans' in which:
            print("\n[GRID] Avvio grid search per K-Means...")
            grid_search_kmeans(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                base_results_dir=RESULTS_KM,
                param_grid=KMEANS_PARAM_GRID,
            )
        if 'dbscan' in which:
            print("\n[GRID] Avvio grid search per DBSCAN...")
            grid_search_dbscan(
                filenames,
                _choose_dbscan_space(features_reduced, features_norm_original, args.dbscan_space),
                features_norm_original,
                features_names,
                music_genres,
                base_results_dir=RESULTS_DBSCAN,
                param_grid=DBSCAN_PARAM_GRID,
            )
        print("\nGrid search completata.")
