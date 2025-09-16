# Musical Analysis – Unsupervised clustering of music tracks

This project extracts audio features from local files (via librosa) or from HDF5 files in the Million Song Dataset (MSD), reduces dimensionality with PCA, and applies multiple clustering algorithms (Spectral Clustering, K‑Means, DBSCAN). It supports parameter grid search, feature caching, and produces reports/plots.

## Highlights
- End‑to‑end pipeline: feature loading, normalization, PCA, clustering, metrics, reports, and figures.
- Multiple feature sources:
  - audio: direct extraction with librosa from local audio files.
  - msd: read/extract from MSD .h5 files (subset or full) with CSV caching.
- Algorithms: Spectral Clustering, K‑Means, DBSCAN.
- Grid search with CSV summaries (grid_summary.csv).
- Figures: PCA/t‑SNE scatter plots, DBSCAN k‑distance, etc.
- Metrics: Silhouette, Davies–Bouldin (DBSCAN also reports non‑noise metrics).
- Optional mapping track_id → (song_title, artist_name) to improve readability in outputs.

## Project structure
- main.py – CLI entry point (single run or grid search, audio or MSD).
- lib/ – Pipelines and utilities:
  - extract_data_features.py – feature extraction from local audio.
  - extract_msd_h5_features.py – feature extraction from MSD .h5 + titles mapping.
  - spectral_clustering.py, k_means_clustering.py, dbscan_clustering.py – algorithms and reports.
  - utils.py – helper functions (metrics, plotting, parameter combinations, etc.).
- dataset/ – Data and cached feature CSVs (audio and MSD).
- clustering_results/ – Outputs per run (plots + markdown + grid summaries).
- generate_figures/ – Optional scripts for comparative figures.

## Requirements
- Python 3.9+ (tested with CPython 3.12)
- Dependencies listed in requirements.txt
- Optional: Million Song Dataset (subset) http://millionsongdataset.com/pages/getting-dataset/
- Optional: Tracks mapping file http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt

Install:
```bash
python -m pip install -r requirements.txt
```
On Windows, if you use spotdl, also install FFmpeg in PATH.

## Feature sources
### 1) Local audio (--feature-source audio)
- Place files under dataset/songs/<genre>/ (e.g., dataset/songs/trap/).
- Update CSV_FEATURE_FILENAME in main.py if you want to reuse a pre‑computed CSV.

### 2) Million Song Dataset (--feature-source msd)
- Download/extract the MSD subset (root folder with nested .h5 files).
- The system generates (or reuses) a CSV cache via --msd-csv. If the file exists, it is reused to avoid re‑extraction.
- Optional: mapping file to replace track_id with song_title and add artist_name.

## Optional mapping file format
Each line:
```
TRACK_ID<SEP>SONG_ID<SEP>Artist_name<SEP>Song_title
```
Example:
```
TRMMMYQ128F932D901<SEP>SOQMMHC12AB0180CB8<SEP>Faster Pussy cat<SEP>Silent Night
```
Behavior:
- If provided: the first column of the final features CSV becomes song_title and the second artist_name.
- If not provided: the CSV keeps track_id as first column and does not include artist_name.

## Pipeline overview
1. Load or extract features (audio or MSD) + feature names.
2. Deduplicate identical feature vectors.
3. Normalize features with the selected scaler (MinMax [0,1] by default or StandardScaler via --scaler).
4. PCA (default keeps 0.98 explained variance).
5. Clustering (Spectral, K‑Means, DBSCAN) with Silhouette & Davies–Bouldin metrics.
6. Save markdown reports and figures under clustering_results/<algo>/...

## CLI arguments
- --mode {single,grid}: single run or grid search.
- --which [spectral kmeans dbscan]: restrict algorithms in grid mode.
- --dbscan-space {reduced,reduced_minmax,normalized}: feature space used by DBSCAN.
- --scaler {minmax,standard}: feature scaler (default: minmax).
- --workers INT: number of parallel processes for audio feature extraction (default: 1).
- --feature-source {audio,msd}: feature origin.
- --msd-root PATH: MSD .h5 root folder.
- --msd-csv PATH: CSV cache for MSD features (reused if exists).
- --msd-titles-file PATH: mapping track_id→title/artist.
- --msd-max-files INT: limit number of processed .h5 files (debug).

Base clustering parameters can be edited at the top of main.py: N_CLUSTERS, PCA_COMPONENTS, SPECTRAL_CLUSTERING_GAMMA, DBSCAN_EPS, DBSCAN_MIN_SAMPLES, DBSCAN_METRIC and the corresponding *_PARAM_GRID.

## Usage – Local audio
Single run (recommended DBSCAN space):
```bash
python main.py --feature-source audio --mode single --dbscan-space reduced_minmax
```
Use StandardScaler and 4 workers:
```bash
python main.py --feature-source audio --mode single --scaler standard --workers 4
```
Grid search (all algorithms):
```bash
python main.py --feature-source audio --mode grid --which spectral kmeans dbscan --workers 4
```

## Usage – Million Song Dataset (.h5)
Single run with titles mapping:
```bash
python main.py --feature-source msd --msd-root "C:\\path\\to\\millionsongsubset" --msd-csv dataset/songs_features/msd_h5_features.csv --msd-titles-file "C:\\path\\to\\unique_tracks.txt" --mode single
```
Grid search:
```bash
python main.py --mode grid --which spectral kmeans dbscan --feature-source msd --msd-root C:\\path\\to\\millionsongsubset --msd-csv .\dataset\TheMillionSongDataset_subset\songs_features_msd.csv
```
Limit number of files (quick debug):
```bash
python main.py --feature-source msd --msd-root "C:\\path\\to\\millionsongsubset" --msd-csv dataset/songs_features/msd_test50.csv --msd-max-files 50 --mode single
```
Minimal example without mapping:
```bash
python main.py --feature-source msd --msd-root "C:\\msd_subset" --mode single
```

## DBSCAN feature space
--dbscan-space:
- reduced: raw PCA components.
- reduced_minmax: re‑apply MinMax after PCA (default recommendation for Euclidean distances).
- normalized: pre‑PCA normalized space.

## Performance
- Parallelize audio feature extraction: use --workers N to run extraction with N processes (e.g., 4) while preserving output order.
- Reuse CSV caches: point CSV_FEATURE_FILENAME (audio) or --msd-csv (MSD) to an existing CSV to skip re‑extraction.

## Outputs
Per algorithm:
- Cluster plots (PCA + t‑SNE) and k‑distance for DBSCAN.
- Markdown report with metrics and breakdown of original features.
- In grid search: grid_summary.csv with results and run folders.

## MSD feature caching
- If the file passed to --msd-csv exists, it is reused (no re‑extraction).
- To regenerate: delete the CSV and rerun.

## Notes & tips
- DBSCAN eps and min_samples require inspection of the generated k‑distance plot.
- Too many duplicates can reduce variety: the pipeline removes identical feature vectors.
- You can adjust PCA variance (PCA_COMPONENTS) to balance noise vs. information.

## Troubleshooting
- MSD CSV does not change: delete the cache and rerun.
- No .h5 files found: check --msd-root.
- MemoryError: use --msd-max-files or lower PCA, or process in batches (future extension).
- Missing titles: ensure the track_id is present in the mapping file.
- Few clusters: tune N_CLUSTERS or DBSCAN parameters.
- Empty DBSCAN metrics: happens if fewer than 2 "valid" clusters (excluding noise).
