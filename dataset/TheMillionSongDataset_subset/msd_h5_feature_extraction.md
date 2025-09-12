# Estrazione feature dai file .h5 del Million Song Dataset

Questo documento descrive come usare lo script `extract_msd_h5_features.py` per estrarre feature numeriche dai file HDF5 del Million Song Dataset (subset) e salvarle in un CSV.

## Percorso script
Lo script si trova in:
```
lib/extract_msd_h5_features.py
```
È eseguibile come modulo Python.

## Requisiti
Scarica il The Million Song Dataset (subset) da:
http://millionsongdataset.com/pages/getting-dataset/
Estrarre il contenuto in una cartella locale.
Scarica il file: List of all track Echo Nest ID, presente sulla stessa pagina

## Formato file mapping titoli
File di testo con una riga per traccia, formato (separatori `<SEP>`):
```
TRACK_ID<SEP>SONG_ID<SEP>Artist_name<SEP>Song_title
```
Esempio:
```
TRMMMYQ128F932D901<SEP>SOQMMHC12AB0180CB8<SEP>Faster Pussy cat<SEP>Silent Night
```
Lo script usa:
- 1° campo: TRACK_ID (chiave)
- 3° campo: Artist_name
- 4° campo: Song_title

Se un TRACK_ID non è presente nel mapping:
- `song_title` diventa il track_id
- `artist_name` resta vuoto

## Output CSV
Se fornito il mapping:
```
 song_title,artist_name,<feature_1>,<feature_2>,... 
```
Altrimenti:
```
 track_id,<feature_1>,<feature_2>,... 
```

## Esecuzione base
```
python -m lib.extract_msd_h5_features \
  "C:\\path\\al\\millionsongsubset" \
  dataset/songs_features/msd_h5_features.csv
```

## Con mapping titoli+artisti
```
python -m lib.extract_msd_h5_features \
  "C:\\path\\al\\millionsongsubset" \
  dataset/songs_features/msd_h5_features_titles.csv \
  --titles-file C:\\path\\a\\track_title_mapping.txt
```

## Parametri disponibili
| Parametro | Obblig. | Descrizione |
|----------|---------|-------------|
| `root_dir` | Sì | Cartella radice che contiene l'albero dei file `.h5` |
| `output_csv` | Sì | Percorso file CSV di output |
| `--titles-file` | No | File mapping per sostituire track_id con (song_title, artist_name) |
| `--max-files N` | No | Limita il numero massimo di file `.h5` (debug / test veloce) |
| `--verbose` | No | Log dettagliato per ogni file aperto |
| `--log-every K` | No | Frequenza (in file) dei messaggi di progresso (default 100) |

## Esempi aggiuntivi
Esecuzione di prova su 50 file per validazione rapida:
```
python -m lib.extract_msd_h5_features \
  "C:\\path\\al\\millionsongsubset" \
  dataset/songs_features/test50.csv \
  --max-files 50 --verbose
```

Rigenerare forzando un nuovo CSV (eliminare il precedente):
```
del dataset\songs_features\msd_h5_features.csv   # Windows
# oppure
rm dataset/songs_features/msd_h5_features.csv    # Linux/Mac
```
Poi rilanciare il comando.

## Feature estratte (sintesi)
Per ogni brano vengono calcolate statistiche su:
- Campi scalari: tempo, energy, loudness, key, mode, signature, danceability, ecc.
- Serie temporali: segments, beats, bars, tatums (count, mean, std, min, max)
- Differenze (delta) per vettori *_start
- Matrici: `segments_pitches`, `segments_timbre` (media e std per ciascuna delle 12 dimensioni)
- Feature aggregata: `segments_timbre_global_mean`

Eventuali valori NaN/inf vengono convertiti in 0.0.

## Strategia di ri-esecuzione
Se il CSV esiste lo script lo rilegge invece di ricalcolare. Per aggiornare le feature:
1. Eliminare il CSV esistente
2. Eseguire il comando nuovamente

## Troubleshooting
| Problema | Causa probabile | Soluzione |
|----------|-----------------|-----------|
| Nessun output / termina subito | Il CSV esiste già | Eliminare il CSV e rilanciare |
| "Nessun file .h5 trovato" | Percorso errato | Verificare `root_dir` |
| MemoryError | Troppi file per la RAM | Usare `--max-files`, eseguire batch, unire in seguito |
| Caratteri strani in titoli | Encoding mapping | Assicurarsi UTF-8 o ripulire input |



