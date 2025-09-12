import os
import csv
import math
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from dataset.TheMillionSongDataset_subset import hdf5_getters as GET

# Campi scalari diretti (se esistono) -> verranno letti con fallback a NaN
SCALAR_GETTERS = [
    'danceability','duration','end_of_fade_in','energy','key','key_confidence',
    'loudness','mode','mode_confidence','start_of_fade_out','tempo',
    'time_signature','time_signature_confidence','artist_familiarity',
    'artist_hotttnesss','song_hotttnesss'
]

# Array per cui calcoliamo statistiche. Nome getter -> tipo (matrix/pitch/timbre/array)
ARRAY_GETTERS = {
    'segments_pitches': 'matrix_pitch',   # (N,12)
    'segments_timbre': 'matrix_timbre',   # (N,12)
    'segments_loudness_max': 'array',
    'segments_loudness_max_time': 'array',
    'segments_loudness_start': 'array',
    'segments_confidence': 'array',
    'sections_start': 'array',
    'sections_confidence': 'array',
    'beats_start': 'array',
    'beats_confidence': 'array',
    'bars_start': 'array',
    'bars_confidence': 'array',
    'tatums_start': 'array',
    'tatums_confidence': 'array'
}


def _safe_call(funcname: str, h5, songidx: int):
    """Chiama getter se esiste, altrimenti restituisce np.nan."""
    fn = getattr(GET, f'get_{funcname}', None)
    if fn is None:
        return math.nan
    try:
        return fn(h5, songidx)
    except Exception:
        return math.nan


def _stat_array(name: str, arr: np.ndarray, feats: Dict[str, float]):
    if arr is None:
        feats[f'{name}_count'] = 0.0
        feats[f'{name}_mean'] = 0.0
        feats[f'{name}_std'] = 0.0
        feats[f'{name}_min'] = 0.0
        feats[f'{name}_max'] = 0.0
        return
    a = np.asarray(arr)
    if a.size == 0:
        feats[f'{name}_count'] = 0.0
        feats[f'{name}_mean'] = 0.0
        feats[f'{name}_std'] = 0.0
        feats[f'{name}_min'] = 0.0
        feats[f'{name}_max'] = 0.0
        return
    feats[f'{name}_count'] = float(a.shape[0])
    feats[f'{name}_mean'] = float(np.nanmean(a))
    feats[f'{name}_std'] = float(np.nanstd(a))
    feats[f'{name}_min'] = float(np.nanmin(a))
    feats[f'{name}_max'] = float(np.nanmax(a))


def _delta_stats(name: str, arr: np.ndarray, feats: Dict[str, float]):
    """Statistiche sugli intervalli (differenze successive)."""
    a = np.asarray(arr)
    if a.size < 2:
        feats[f'{name}_delta_mean'] = 0.0
        feats[f'{name}_delta_std'] = 0.0
        feats[f'{name}_delta_min'] = 0.0
        feats[f'{name}_delta_max'] = 0.0
        return
    d = np.diff(a)
    feats[f'{name}_delta_mean'] = float(np.nanmean(d))
    feats[f'{name}_delta_std'] = float(np.nanstd(d))
    feats[f'{name}_delta_min'] = float(np.nanmin(d))
    feats[f'{name}_delta_max'] = float(np.nanmax(d))


def _matrix_stats(base: str, mat: np.ndarray, feats: Dict[str, float]):
    m = np.asarray(mat) if mat is not None else np.asarray([])
    # Gestione casi: None, vuoto, 1D, oppure 2D classico
    if m.size == 0:
        for i in range(12):  # assumiamo 12 dimensioni desiderate (pitch/timbre)
            feats[f'{base}_{i}_mean'] = 0.0
            feats[f'{base}_{i}_std'] = 0.0
        return
    if m.ndim == 1:
        # Se 1D (es. lunghezza 12) trattiamo come una singola "riga"
        length = m.shape[0]
        for i in range(length):
            val = float(m[i]) if not (math.isnan(m[i]) or math.isinf(m[i])) else 0.0
            feats[f'{base}_{i}_mean'] = val
            feats[f'{base}_{i}_std'] = 0.0
        # Se meno di 12 colonne, completa a 12 per consistenza
        for i in range(length, 12):
            feats[f'{base}_{i}_mean'] = 0.0
            feats[f'{base}_{i}_std'] = 0.0
        return
    if m.ndim != 2:
        # Forma inaspettata: fallback a zeri
        for i in range(12):
            feats[f'{base}_{i}_mean'] = 0.0
            feats[f'{base}_{i}_std'] = 0.0
        return
    cols = m.shape[1]
    # Limita a 12 se più grande, oppure completa se più piccolo
    for i in range(min(cols, 12)):
        col = m[:, i]
        if col.size == 0:
            feats[f'{base}_{i}_mean'] = 0.0
            feats[f'{base}_{i}_std'] = 0.0
        else:
            feats[f'{base}_{i}_mean'] = float(np.nanmean(col))
            feats[f'{base}_{i}_std'] = float(np.nanstd(col))
    for i in range(cols, 12):
        feats[f'{base}_{i}_mean'] = 0.0
        feats[f'{base}_{i}_std'] = 0.0


def extract_song_features(h5, songidx: int) -> Tuple[Dict[str, Any], List[str]]:
    feats: Dict[str, float] = {}

    # Campi scalari
    for name in SCALAR_GETTERS:
        val = _safe_call(name, h5, songidx)
        try:
            if isinstance(val, (bytes, str)):
                # Ignora stringhe (non usate per clustering); salva come lunghezza
                feats[f'{name}_strlen'] = float(len(val))
            else:
                fval = float(val)
                if math.isnan(fval) or math.isinf(fval):
                    fval = 0.0
                feats[name] = fval
        except Exception:
            feats[name] = 0.0

    # Array e matrici
    for getter, kind in ARRAY_GETTERS.items():
        fn = getattr(GET, f'get_{getter}', None)
        if fn is None:
            continue
        try:
            arr = fn(h5, songidx)
        except Exception:
            arr = None
        if kind.startswith('matrix'):
            _matrix_stats(getter, arr, feats)
        else:
            _stat_array(getter, arr, feats)
            # delta stats per start arrays
            if getter.endswith('_start'):
                _delta_stats(getter, arr, feats)

    # Derivate su segments_timbre (energia media timbrica)
    if 'segments_timbre_0_mean' in feats and 'segments_timbre_0_std' in feats:
        # Esempio di feature aggregata personalizzata
        timbre_means = [v for k, v in feats.items() if k.startswith('segments_timbre_') and k.endswith('_mean')]
        if timbre_means:
            feats['segments_timbre_global_mean'] = float(np.mean(timbre_means))

    # Normalizzazione semplice di key/mode/time_signature (già numerici) -> nessuna

    # Ordine nomi stabile
    feature_names = sorted(feats.keys())
    return feats, feature_names


def extract_features_from_h5_file(h5_path: str, verbose: bool=False) -> Tuple[List[str], np.ndarray, List[str]]:
    import tables
    rows_features = []
    track_ids = []
    feature_names_master: List[str] | None = None
    try:
        h5 = GET.open_h5_file_read(h5_path)
    except Exception as e:
        print(f'Errore apertura {h5_path}: {e}')
        return [], np.empty((0,)), []
    try:
        n = GET.get_num_songs(h5)
        if verbose:
            print(f"  -> {os.path.basename(h5_path)} contiene {n} canzoni", flush=True)
        for idx in range(n):
            feats, names = extract_song_features(h5, idx)
            if feature_names_master is None:
                feature_names_master = names
            # Allineamento (riempi campi mancanti con 0)
            row = [feats.get(name, 0.0) for name in feature_names_master]
            rows_features.append(row)
            # track id
            try:
                tid = GET.get_track_id(h5, idx)
            except Exception:
                tid = f'{os.path.basename(h5_path)}::{idx}'
            if isinstance(tid, bytes):
                tid = tid.decode('utf-8', errors='ignore')
            track_ids.append(str(tid))
    finally:
        try:
            h5.close()
        except Exception:
            pass
    if not rows_features:
        return [], np.empty((0,)), []
    return track_ids, np.asarray(rows_features, dtype=float), feature_names_master or []


def walk_msd_h5(root_dir: str) -> List[str]:
    paths = []
    for dirpath, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith('.h5'):
                paths.append(os.path.join(dirpath, f))
    return paths


def extract_msd_features(root_dir: str, max_files: int | None = None, verbose: bool=False, log_every: int=100) -> Tuple[List[str], np.ndarray, List[str]]:
    all_rows = []
    all_ids = []
    names_master: List[str] | None = None
    h5_files = walk_msd_h5(root_dir)
    print(f"Trovati {len(h5_files)} file .h5", flush=True)
    if not h5_files:
        raise ValueError('Nessun file .h5 trovato')
    if max_files is not None:
        h5_files = h5_files[:max_files]
        print(f"Limite max_files attivo: analizzerò i primi {len(h5_files)} file", flush=True)
    for i, h5_path in enumerate(tqdm(h5_files, desc='HDF5 files')):
        tids, feats, names = extract_features_from_h5_file(h5_path, verbose=verbose)
        if feats.size == 0:
            continue
        if names_master is None:
            names_master = names
        for row in feats:
            all_rows.append(row.tolist())
        all_ids.extend(tids)
        if verbose or (i+1) % log_every == 0:
            print(f"Processati {i+1}/{len(h5_files)} file (tot rows: {len(all_rows)})", flush=True)
    if not all_rows:
        raise ValueError('Nessuna feature estratta')
    return all_ids, np.asarray(all_rows, dtype=float), names_master or []


def load_track_metadata_map(titles_file: str) -> Dict[str, tuple[str,str]]:
    """Carica file testo lines: track_id<SEP>song_id<SEP>artist_name<SEP>song_title.
    Ritorna dict {track_id: (song_title, artist_name)}.
    Mantiene il primo valore incontrato per duplicati.
    """
    mapping: Dict[str, tuple[str,str]] = {}
    if not titles_file:
        return mapping
    if not os.path.isfile(titles_file):
        print(f"ATTENZIONE: file titoli non trovato: {titles_file}")
        return mapping
    with open(titles_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line or '<SEP>' not in line:
                continue
            parts = line.split('<SEP>')
            if len(parts) < 4:
                continue
            track_id = parts[0].strip()
            artist_name = parts[2].strip()
            song_title = parts[3].strip()
            if track_id and track_id not in mapping:
                mapping[track_id] = (song_title, artist_name)
    print(f"Caricati {len(mapping)} titoli+artisti da mapping")
    return mapping


def save_msd_features_csv(id_or_titles: List[str], feature_array: np.ndarray, feature_names: List[str], csv_path: str, header_label: str, artist_values: List[str] | None = None):
    if artist_values is not None:
        header = [header_label, 'artist_name'] + feature_names
    else:
        header = [header_label] + feature_names
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        if artist_values is not None:
            for ident, artist, row in zip(id_or_titles, artist_values, feature_array):
                w.writerow([ident, artist] + [f'{v:.10g}' for v in row])
        else:
            for ident, row in zip(id_or_titles, feature_array):
                w.writerow([ident] + [f'{v:.10g}' for v in row])


def get_msd_h5_features(root_dir: str, csv_output: str, max_files: int | None = None, verbose: bool=False, titles_file: str | None = None):
    # Se esiste già il CSV lo ricarichiamo (accetta header track_id o song_title / song_title+artist_name)
    if os.path.exists(csv_output):
        with open(csv_output, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            header = next(r)
            # Determina offset feature
            if len(header) >= 2 and header[0] == 'song_title' and header[1] == 'artist_name':
                feature_names = header[2:]
                id_or_titles = []
                artist_dummy = []  # ignorato in ritorno
                rows = []
                for line in r:
                    if not line:
                        continue
                    id_or_titles.append(line[0])
                    artist_dummy.append(line[1])
                    rows.append([float(x) for x in line[2:]])
                return id_or_titles, np.asarray(rows, dtype=float), feature_names
            else:
                feature_names = header[1:]
                id_or_titles = []
                rows = []
                for line in r:
                    if not line:
                        continue
                    id_or_titles.append(line[0])
                    rows.append([float(x) for x in line[1:]])
                return id_or_titles, np.asarray(rows, dtype=float), feature_names

    # Estrazione
    track_ids, feats, names = extract_msd_features(root_dir, max_files=max_files, verbose=verbose)

    # Mapping metadati se fornito
    metadata_map = load_track_metadata_map(titles_file) if titles_file else {}
    if metadata_map:
        song_titles: List[str] = []
        artist_names: List[str] = []
        for tid in track_ids:
            song_title, artist_name = metadata_map.get(tid, (tid, ''))
            song_titles.append(song_title)
            artist_names.append(artist_name)
        save_msd_features_csv(song_titles, feats, names, csv_output, header_label='song_title', artist_values=artist_names)
        return song_titles, feats, names
    else:
        save_msd_features_csv(track_ids, feats, names, csv_output, header_label='track_id')
        return track_ids, feats, names


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Estrazione feature da file .h5 (Million Song Dataset)')
    p.add_argument('root_dir', help='Cartella radice con i file .h5')
    p.add_argument('output_csv', help='Percorso CSV di output')
    p.add_argument('--max-files', type=int, default=None, help='Limita il numero di file .h5 da processare (debug)')
    p.add_argument('--verbose', action='store_true', help='Log dettagliato per ogni file')
    p.add_argument('--log-every', type=int, default=100, help='Frequenza logging progresso (default 100)')
    p.add_argument('--titles-file', type=str, default=None, help='File testo mapping track_id<SEP>song_id<SEP>artist_name<SEP>song_title')
    args = p.parse_args()
    print('Avvio estrazione feature MSD...', flush=True)
    ids_or_titles, feats, names = get_msd_h5_features(args.root_dir, args.output_csv, max_files=args.max_files, verbose=args.verbose, titles_file=args.titles_file)
    print(f'Creato {args.output_csv} con {len(ids_or_titles)} tracce e {len(names)} feature.')
