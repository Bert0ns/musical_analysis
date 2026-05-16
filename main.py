import csv
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from lib.dbscan_clustering import run_dbscan_clustering_pipeline, choose_dbscan_space
from lib.extract_data_features import get_audio_features
from lib.extract_msd_h5_features import get_msd_h5_features
from lib.k_means_clustering import run_kmeans_clustering_pipeline
from lib.spectral_clustering import run_spectral_clustering_pipeline
from lib.utils import param_product, fmt_float

#CSV_FEATURE_FILENAME = "dataset/GTZAN/GTZAN_features.csv"
CSV_FEATURE_FILENAME = "dataset/songs_features/songs_features_all.csv"
SONGS_DIR = "dataset/songs"  # Change to the path of your songs folder

RESULTS_SC = "clustering_results/spectral_clustering"
RESULTS_KM = "clustering_results/kmeans"
RESULTS_DBSCAN = "clustering_results/dbscan"

N_CLUSTERS = 5  # Number of clusters to create
SPECTRAL_CLUSTERING_GAMMA = 0.2  # Gamma parameter for spectral clustering

# DBSCAN parameters (defaults, tune after reviewing plots)
DBSCAN_EPS = 0.7
DBSCAN_MIN_SAMPLES = 8
DBSCAN_METRIC = 'euclidean'

# ===================== PARAM GRID (example values, adjust freely) =====================
PCA_COMPONENTS = 0.98  # Variance ratio to keep with PCA

SPECTRAL_PARAM_GRID = {
    'n_clusters': [9, 11],
    'gamma': [0.001, 0.003],
}

KMEANS_PARAM_GRID = {
    'n_clusters': [9, 11],
}

DBSCAN_PARAM_GRID = {
    'eps': [7.0, 7.5, 9.0],
    'min_samples': [3, 5],
    'metric': ['euclidean'],  # 'cosine' 'euclidean', 'manhattan', 'minkowski'
}


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
        for params in param_product(param_grid):
            n_clusters = params['n_clusters']
            gamma = params['gamma']
            run_dir = os.path.join(base_results_dir, f"grid_k{n_clusters}_g{fmt_float(gamma)}")

            labels, sil, dbi = run_spectral_clustering_pipeline(
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

            writer.writerow([n_clusters, gamma, unique_cls, f"{sil:.6f}", f"{dbi:.6f}", run_dir])
    print(f"Spectral grid summary written to: {summary_path}")


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
        for params in param_product(param_grid):
            n_clusters = params['n_clusters']
            run_dir = os.path.join(base_results_dir, f"grid_k{n_clusters}")
            labels, _, sil, dbi = run_kmeans_clustering_pipeline(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                results_dir=run_dir,
                n_clusters=n_clusters,
            )
            unique_cls = len(np.unique(labels))
            writer.writerow([n_clusters, unique_cls, f"{sil:.6f}", f"{dbi:.6f}", run_dir])
    print(f"K-Means grid summary written to: {summary_path}")


def grid_search_dbscan(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        eps_values_to_compute_sil_and_dbi,
        base_results_dir: str = RESULTS_DBSCAN,
        param_grid: dict | None = None,
):
    if param_grid is None:
        param_grid = DBSCAN_PARAM_GRID
    os.makedirs(base_results_dir, exist_ok=True)
    summary_path = os.path.join(base_results_dir, "grid_summary.csv")
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["eps", "min_samples", "metric", "n_clusters_found", "noise_ratio", "silhouette_non_noise",
                         "davies_bouldin_non_noise", "results_dir"])
        for params in param_product(param_grid):
            eps = params['eps']
            min_samples = params['min_samples']
            metric = params['metric']
            run_dir = os.path.join(base_results_dir, f"grid_eps{fmt_float(eps)}_min{min_samples}_{metric}")
            labels, _, sil, dbi = run_dbscan_clustering_pipeline(
                filenames,
                features_reduced,
                features_norm_original,
                features_names,
                music_genres,
                results_dir=run_dir,
                eps=eps,
                min_samples=min_samples,
                metric=metric,
                eps_values_to_compute_sil_and_dbi=eps_values_to_compute_sil_and_dbi,
            )
            unique_valid = [c for c in np.unique(labels) if c != -1]
            noise_ratio = float(np.sum(labels == -1)) / float(len(labels)) if len(labels) else 0.0

            if sil is not None and dbi is not None:
                sil_s = f"{sil:.6f}"
                dbi_s = f"{dbi:.6f}"
            else:
                sil_s = "N/A"
                dbi_s = "N/A"

            writer.writerow([eps, min_samples, metric, len(unique_valid), f"{noise_ratio:.6f}", sil_s, dbi_s, run_dir])
    print(f"DBSCAN grid summary written to: {summary_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Music clustering - single run or parameter grid search")
    parser.add_argument('--mode', choices=['single', 'grid'], default='single',
                        help='single: run once with default parameters; grid: test parameter combinations')
    parser.add_argument('--which', nargs='*', choices=['spectral', 'kmeans', 'dbscan'],
                        help='When in grid mode, limit to the selected algorithms')
    parser.add_argument(
        '--dbscan-space',
        choices=['reduced', 'reduced_minmax', 'normalized'],
        default='reduced',
        help='Feature space used by DBSCAN'
    )
    # Feature scaling selection
    parser.add_argument('--scaler', choices=['minmax', 'standard'], default='minmax',
                        help='Select feature scaling: minmax (default) or standard')
    # Number of processes for audio feature extraction
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of processes for audio feature extraction (default: 1)')
    # Arguments for Million Song Dataset features
    parser.add_argument('--feature-source', choices=['audio', 'msd'], default='audio',
                        help='Feature source: local audio (librosa) or msd (.h5 files)')
    parser.add_argument('--msd-root', type=str, default=None, help='Root folder containing MSD .h5 files')
    parser.add_argument('--msd-csv', type=str, default='dataset/songs_features/msd_h5_features.csv',
                        help='CSV cache for MSD features')
    parser.add_argument('--msd-titles-file', type=str, default=None,
                        help='Mapping file track_id<SEP>song_id<SEP>artist_name<SEP>song_title')
    parser.add_argument('--msd-max-files', type=int, default=None, help='Limit number of .h5 files (debug)')
    args = parser.parse_args()

    # ================= FEATURE LOADING =================
    if args.feature_source == 'audio':
        print("Loading audio features (local extraction/librosa)...")
        filenames, music_genres, features, features_names = get_audio_features(SONGS_DIR, CSV_FEATURE_FILENAME,
                                                                               n_jobs=args.workers)
        source_label = 'audio'
    else:
        if args.msd_root is None and args.msd_csv is None:
            raise ValueError("To use MSD features, specify --msd-root and/or --msd-csv")

        print("Loading / extracting MSD features (.h5)...")
        filenames, artist_names, features, features_names = get_msd_h5_features(
            args.msd_root,
            args.msd_csv,
            max_files=args.msd_max_files,
            verbose=False,
            titles_file=args.msd_titles_file,
        )
        # Reuse the music_genres slot for pipeline compatibility (reports) using artist name
        music_genres = [a if a else 'UNKNOWN' for a in artist_names]
        source_label = 'msd'
    print(f"Feature source: {source_label}")
    print("Shape feature array:", features.shape)

    # Remove duplicate feature vectors
    features, unique_indices = np.unique(features, axis=0, return_index=True)
    filenames = [filenames[i] for i in unique_indices]
    music_genres = [music_genres[i] for i in unique_indices]
    print("Shape feature array after duplicate removal:", features.shape)

    # Feature scaling with user-selected scaler
    if args.scaler == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    print(f"Selected scaler: {scaler.__class__.__name__}")
    features_norm = scaler.fit_transform(features)

    # Copy for detailed report (before PCA)
    features_norm_original = features_norm.copy()

    # Dimensionality reduction with PCA
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='full', random_state=42)
    features_reduced = pca.fit_transform(features_norm)
    print("Shape feature array after PCA:", features_reduced.shape)

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
            choose_dbscan_space(features_reduced, features_norm_original, args.dbscan_space),
            features_norm_original,
            features_names,
            music_genres,
            results_dir=RESULTS_DBSCAN,
            eps=DBSCAN_EPS,
            min_samples=DBSCAN_MIN_SAMPLES,
            metric=DBSCAN_METRIC,
        )

        print("\nPipeline completed (Spectral + K-Means + DBSCAN)")
    else:
        which = set(args.which) if args.which else {'spectral', 'kmeans', 'dbscan'}
        if 'spectral' in which:
            print("\n[GRID] Starting grid search for Spectral Clustering...")
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
            print("\n[GRID] Starting grid search for K-Means...")
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
            print("\n[GRID] Starting grid search for DBSCAN...")

            # get first value in this list DBSCAN_PARAM_GRID.get('eps') as a starting point, and stopping point take the last value
            eps = DBSCAN_PARAM_GRID.get('eps', [])
            start, stop = (eps[0], eps[-1]) if eps else (0.0, 0.0)
            eps_values = np.linspace(min(start, stop), max(start, stop), num=20)

            grid_search_dbscan(
                filenames,
                choose_dbscan_space(features_reduced, features_norm_original, args.dbscan_space),
                features_norm_original,
                features_names,
                music_genres,
                eps_values_to_compute_sil_and_dbi=eps_values,
                base_results_dir=RESULTS_DBSCAN,
                param_grid=DBSCAN_PARAM_GRID,
            )
        print("\nGrid search completed.")
