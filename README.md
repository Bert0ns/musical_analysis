# Musical Analysis – Unsupervised Clustering of Songs

This project extracts audio features from a local songs folder, reduces dimensionality with PCA, and clusters the tracks using multiple unsupervised algorithms (Spectral Clustering, K‑Means, and DBSCAN). It also performs parameter grid searches and saves visualizations and reports for each run.

## Key Features
- End‑to‑end pipeline: feature loading, normalization, PCA, clustering, evaluation, and reporting.
- Multiple algorithms: Spectral Clustering, K‑Means, DBSCAN.
- Grid search for hyperparameters with per‑run folders and a summary CSV.
- Figures: cluster scatter plots, t‑SNE projections, DBSCAN k‑distance plots.
- Metrics: Silhouette and Davies–Bouldin indices (overall and non‑noise for DBSCAN).

## Project Structure
- `main.py` – CLI entry point to run a single experiment or a grid search.
- `lib/` – Algorithm pipelines and utilities:
  - `extract_data_features.py` – load/compute audio features and metadata.
  - `spectral_clustering.py`, `k_means_clustering.py`, `dbscan_clustering.py` – run and report clustering.
  - `utils.py` – helpers (scores, formatting, parameter product, plotting, etc.).
- `dataset/` – Songs and precomputed features CSVs.
  - `songs/` – put your audio files here, e.g., `dataset/songs/trap/`.
  - `songs_features/` – feature CSVs (e.g., `songs_features_all.csv`).
  - `download_songs.md` – optional notes for preparing/downloading songs.
- `clustering_results/` – Outputs per run/grid: plots, markdown reports, and `grid_summary.csv`.
- `generate_figures/` – Optional scripts to aggregate/compare results across runs.

## Requirements
- Python 3.9+ (tested with CPython 3.12) on Windows/Linux/macOS
- Dependencies (see `requirements.txt`):
  - numpy, scipy, scikit‑learn, librosa, resampy, matplotlib, seaborn
  - spotdl (optional, for downloading tracks if you choose to use it)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

On Windows, if you plan to use `spotdl`, you may also need a working FFmpeg installation available on PATH.

## Data Preparation
- Place your audio files under `dataset/songs/<genre>/` (e.g., `dataset/songs/trap/`).
- Ensure `CSV_FEATURE_FILENAME` in `main.py` points to the features CSV you want to use (e.g., `dataset/songs_features/songs_features_all.csv`).
- The script will load the features and metadata using `get_audio_features(...)`. You can provide or regenerate CSVs using the utilities under `lib/` (see their docstrings/comments).

## How It Works (Pipeline)
1. Load features and metadata: filenames, genres, features array, and feature names.
2. Deduplicate identical feature vectors (keeps the first occurrence).
3. Normalize features with Min‑Max scaling to [0, 1].
4. Reduce dimensionality via PCA, keeping 98% variance by default (`PCA_COMPONENTS = 0.98`).
5. Run clustering algorithms and compute Silhouette and Davies–Bouldin scores.
6. Save plots and markdown reports in `clustering_results/<algo>/...`.

## Usage
Run from the project root.

### Single run (default parameters)
This runs Spectral, K‑Means, and DBSCAN once using the constants defined at the top of `main.py`:

```bash
python main.py --mode single [--dbscan-space reduced|reduced_minmax|normalized]
```

- `--dbscan-space` controls which feature space DBSCAN uses:
  - `reduced` – the PCA‑reduced features as‑is (default if you use `reduced_minmax` in code).
  - `reduced_minmax` – applies Min‑Max scaling after PCA (often helpful for distance‑based clustering).
  - `normalized` – the pre‑PCA normalized features.

Key defaults in `main.py` you may want to tweak:
- `SONGS_DIR` – folder with your audio files (default: `dataset/songs/trap`).
- `N_CLUSTERS` – number of clusters for Spectral and K‑Means (default: 5).
- `PCA_COMPONENTS` – PCA variance to keep (default: 0.98).
- `SPECTRAL_CLUSTERING_GAMMA` – RBF gamma for Spectral (default: 0.2).
- `DBSCAN_EPS`, `DBSCAN_MIN_SAMPLES`, `DBSCAN_METRIC` – DBSCAN parameters.

Outputs are written under:
- `clustering_results/spectral_clustering/`
- `clustering_results/kmeans/`
- `clustering_results/dbscan/`

Each run produces:
- Plots: cluster scatter plots and t‑SNE (`tsne_clusters_plot_*.png`).
- DBSCAN also saves `k_distance_dbscan.png`.
- Markdown reports: overall summary and feature breakdown on original feature space.

### Grid search
Test multiple hyperparameter combinations and get a summary CSV.

```bash
python main.py --mode grid --which spectral kmeans dbscan [--dbscan-space ...]
```

- `--which` filters which algorithms to grid search: choose any of `spectral`, `kmeans`, `dbscan`.
- Parameter grids are configured in `main.py` via:
  - `SPECTRAL_PARAM_GRID = {"n_clusters": [...], "gamma": [...]}`
  - `KMEANS_PARAM_GRID = {"n_clusters": [...]}`
  - `DBSCAN_PARAM_GRID = {"eps": [...], "min_samples": [...], "metric": ["euclidean", "cosine"]}`

Each algorithm writes a `grid_summary.csv` with per‑combination scores and a link to the detailed run folder.

## Reproducibility Notes
- PCA and K‑Means can be sensitive to scaling; ensure consistent preprocessing when changing inputs.
- DBSCAN behavior depends strongly on the feature space (`--dbscan-space`), `eps`, and `min_samples`. Use the saved k‑distance plot to pick `eps`.
- Deduplication removes exact duplicate feature vectors; confirm this matches your intent if you aggregate datasets.

## Troubleshooting
- "No module named <X>": ensure you installed `requirements.txt` into the active environment.
- Empty or few clusters: adjust `N_CLUSTERS` (for Spectral/K‑Means) or tune DBSCAN (`eps`, `min_samples`, and feature space).
- Plots not generated: verify the `clustering_results/` directory is writable and that `matplotlib` backend works in your environment.
- Audio loading errors: check file formats supported by `librosa` and codec availability.
