# Musical Analysis – Clustering non supervisionato di brani musicali

Questo progetto estrae feature audio da file locali (librosa) oppure da file HDF5 del Million Song Dataset (MSD), riduce la dimensionalità con PCA e applica più algoritmi di clustering (Spectral Clustering, K‑Means, DBSCAN). Supporta grid search degli iper‑parametri, caching delle feature e generazione di report/plot.

## Caratteristiche Principali
- Pipeline end‑to‑end: caricamento feature, normalizzazione, PCA, clustering, metriche, report e figure.
- Sorgenti feature multiple:
  - `audio`: estrazione diretta via librosa da file audio locali.
  - `msd`: lettura/estrazione da file `.h5` del Million Song Dataset (subset o intero), con caching CSV.
- Algoritmi: Spectral Clustering, K‑Means, DBSCAN.
- Grid search con salvataggio riassunti (`grid_summary.csv`).
- Figure: scatter plot PCA/t‑SNE, k‑distance DBSCAN, ecc.
- Metriche: Silhouette, Davies–Bouldin (e variante non‑noise per DBSCAN).
- Mapping opzionale `track_id → (song_title, artist_name)` per sostituire l’id tecnico nei CSV.

## Struttura del Progetto
- `main.py` – Entry point CLI (singola esecuzione o grid search, audio o MSD).
- `lib/` – Pipeline e utilità:
  - `extract_data_features.py` – feature da audio locale.
  - `extract_msd_h5_features.py` – feature da file `.h5` MSD + mapping titoli.
  - `spectral_clustering.py`, `k_means_clustering.py`, `dbscan_clustering.py` – algoritmi e report.
  - `utils.py` – funzioni di supporto (metriche, plotting, combinazioni parametri, ecc.).
- `dataset/` – Dati e CSV di feature (sia audio sia MSD).
- `clustering_results/` – Output per run (plot + markdown + summary grid).
- `generate_figures/` – Script opzionali per generare figure comparative.

## Requisiti
- Python 3.9+ (testato con CPython 3.12)
- Dipendenze in `requirements.txt`
- (Optional) The million song dataset (subset): http://millionsongdataset.com/pages/getting-dataset/
- (Optional) File mapping titoli: http://millionsongdataset.com/sites/default/files/AdditionalFiles/unique_tracks.txt

Installazione:
```bash
python -m pip install -r requirements.txt
```
Su Windows, per eventuale uso di `spotdl`, installare anche FFmpeg nel PATH.

## Origini delle Feature
### 1. Audio locale (`--feature-source audio`)
- Metti i file sotto `dataset/songs/<genere>/` (es: `dataset/songs/trap/`).
- Specifica/aggiorna `CSV_FEATURE_FILENAME` in `main.py` se vuoi riusare un CSV pre‑calcolato.

### 2. Million Song Dataset (`--feature-source msd`)
- Scarica/estrai il subset MSD (cartella radice con albero di sottocartelle `.h5`).
- Il sistema genera (o riusa) un CSV cache con le feature (argomento `--msd-csv`). Se il file esiste, viene riutilizzato per evitare ricalcolo.
- Opzionale: file mapping per sostituire `track_id` con titolo e aggiungere l’artista.

## Formato File di Mapping (Opzionale)
Ogni riga:
```
TRACK_ID<SEP>SONG_ID<SEP>Artist_name<SEP>Song_title
```
Esempio:
```
TRMMMYQ128F932D901<SEP>SOQMMHC12AB0180CB8<SEP>Faster Pussy cat<SEP>Silent Night
```
Comportamento:
- Se fornito: la prima colonna del CSV finale delle feature diventa `song_title` e la seconda `artist_name`.
- Se non fornito: il CSV mantiene `track_id` come prima colonna e non include `artist_name`.

## Pipeline (Sintesi)
1. Caricamento o estrazione feature (audio o MSD) + nomi feature.
2. Deduplicazione di vettori identici.
3. Normalizzazione Min‑Max [0,1].
4. PCA (default: varianza conservata `0.98`).
5. Clustering (Spectral, K‑Means, DBSCAN) con metriche Silhouette & Davies–Bouldin.
6. Report markdown + figure salvate in `clustering_results/<algo>/...`.

## Argomenti CLI Principali
| Argomento | Valori / Tipo | Descrizione |
|-----------|---------------|-------------|
| `--mode` | `single` / `grid` | Esecuzione singola o grid search |
| `--which` | elenco (`spectral kmeans dbscan`) | Limita algoritmi in modalità grid |
| `--dbscan-space` | `reduced` / `reduced_minmax` / `normalized` | Spazio feature per DBSCAN |
| `--feature-source` | `audio` / `msd` | Origine feature |
| `--msd-root` | path | Radice albero file `.h5` MSD |
| `--msd-csv` | path CSV | Cache feature MSD (riusata se esiste) |
| `--msd-titles-file` | path TXT | Mapping `track_id`→titolo/artista |
| `--msd-max-files` | int | Limita numero file (debug) |

(Parametri di clustering base si editano in cima a `main.py`: `N_CLUSTERS`, `PCA_COMPONENTS`, `SPECTRAL_CLUSTERING_GAMMA`, `DBSCAN_EPS`, `DBSCAN_MIN_SAMPLES`, `DBSCAN_METRIC` e le rispettive grid `*_PARAM_GRID`).

## Utilizzo – Audio Locale
Esecuzione singola:
```bash
python main.py --feature-source audio --mode single --dbscan-space reduced_minmax
```
Grid search (tutti gli algoritmi):
```bash
python main.py --feature-source audio --mode grid --which spectral kmeans dbscan
```

## Utilizzo – Million Song Dataset (.h5)
Esecuzione singola con mapping titoli:
```bash
python main.py --feature-source msd --msd-root "C:\\Users\\me\\Downloads\\millionsongsubset" --msd-csv dataset/songs_features/msd_h5_features.csv --msd-titles-file "C:\\Users\\me\\Downloads\\track_title_mapping.txt" --mode single
```
Grid search:
```bash
python .\main.py --mode grid --which spectral kmeans dbscan --feature-source msd --msd-root C:\Users\david\Downloads\millionsongsubset --msd-csv .\dataset\TheMillionSongDataset_subset\songs_features_msd.csv --msd-titles-file "C:\Users\david\Downloads\Nuova cartella\unique_tracks.txt"
```
Limita numero file (debug veloce):
```bash
python main.py --feature-source msd --msd-root "C:\\Users\\me\\Downloads\\millionsongsubset" --msd-csv dataset/songs_features/msd_test50.csv --msd-max-files 50 --mode single
```
Esempio minimale senza mapping:
```bash
python main.py --feature-source msd --msd-root "C:\\msd_subset" --mode single
```

## Spazio Feature per DBSCAN
`--dbscan-space`:
- `reduced`: componenti PCA così come sono.
- `reduced_minmax`: ri‑applica Min‑Max dopo PCA (default consigliato per distanze euclidee).
- `normalized`: spazio normalizzato pre‑PCA.

## Output
Per ogni algoritmo:
- Plot cluster (PCA + t‑SNE) e per DBSCAN anche k‑distance.
- Report markdown con metriche e breakdown feature originali.
- In grid search: `grid_summary.csv` con risultati e link cartelle run.

## Caching delle Feature MSD
- Se il file passato con `--msd-csv` esiste viene riusato (nessuna ri‑estrazione).
- Per rigenerare: cancellare il CSV e rilanciare il comando.

## Note di Qualità & Consigli
- La scelta di `eps` e `min_samples` in DBSCAN richiede ispezione del grafico k‑distance generato.
- Troppi duplicati possono ridurre varietà: la pipeline rimuove feature vector identici.
- Puoi ridurre/espandere la varianza PCA (`PCA_COMPONENTS`) per bilanciare rumore vs informazione.

## Troubleshooting
| Problema | Possibile Soluzione |
|----------|--------------------|
| CSV MSD non cambia | Elimina il file cache e rilancia |
| Nessun file .h5 trovato | Controlla il path in `--msd-root` |
| MemoryError | Usa `--msd-max-files` o abbassa PCA / processa a batch (estensione futura) |
| Titoli mancanti | Verifica presenza `track_id` nel file mapping |
| Pochi cluster | Regola `N_CLUSTERS` o parametri DBSCAN |
| Metriche vuote DBSCAN | Accade se meno di 2 cluster “validi” (escludendo noise) |
