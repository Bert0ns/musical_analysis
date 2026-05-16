# Feature extraction from Million Song Dataset .h5 files

This document describes how to use `extract_msd_h5_features.py` to extract numeric features from HDF5 files in the Million Song Dataset (subset) and save them to a CSV.

## Script path
The script lives at:
```
lib/extract_msd_h5_features.py
```
It can be run as a Python module.

## Requirements
Download the Million Song Dataset (subset) from:
http://millionsongdataset.com/pages/getting-dataset/
Extract the contents into a local folder.
Also download the file "List of all track Echo Nest ID" from the same page.

## Title mapping file format
Text file with one line per track, format (separator `<SEP>`):
```
TRACK_ID<SEP>SONG_ID<SEP>Artist_name<SEP>Song_title
```
Example:
```
TRMMMYQ128F932D901<SEP>SOQMMHC12AB0180CB8<SEP>Faster Pussy cat<SEP>Silent Night
```
The script uses:
- field 1: TRACK_ID (key)
- field 3: Artist_name
- field 4: Song_title

If a TRACK_ID is not present in the mapping:
- `song_title` becomes the track_id
- `artist_name` remains empty

## Output CSV
If mapping is provided:
```
song_title,artist_name,<feature_1>,<feature_2>,...
```
Otherwise:
```
track_id,<feature_1>,<feature_2>,...
```

## Basic run
```
python -m lib.extract_msd_h5_features \
  "C:\path\to\millionsongsubset" \
  dataset/songs_features/msd_h5_features.csv
```

## With title+artist mapping
```
python -m lib.extract_msd_h5_features \
  "C:\path\to\millionsongsubset" \
  dataset/songs_features/msd_h5_features_titles.csv \
  --titles-file C:\path\to\track_title_mapping.txt
```

## Available parameters
| Parameter | Required | Description |
|----------|----------|-------------|
| `root_dir` | Yes | Root folder containing the `.h5` file tree |
| `output_csv` | Yes | Output CSV path |
| `--titles-file` | No | Mapping file to replace track_id with (song_title, artist_name) |
| `--max-files N` | No | Limit the max number of `.h5` files (debug / quick test) |
| `--verbose` | No | Verbose logging per file |
| `--log-every K` | No | Progress message frequency (default 100) |

## Additional examples
Test run on 50 files for quick validation:
```
python -m lib.extract_msd_h5_features \
  "C:\path\to\millionsongsubset" \
  dataset/songs_features/test50.csv \
  --max-files 50 --verbose
```

Force regeneration by deleting the previous CSV:
```
del dataset\songs_features\msd_h5_features.csv   # Windows
# or
rm dataset/songs_features/msd_h5_features.csv    # Linux/Mac
```
Then rerun the command.

## Extracted features (summary)
For each track, stats are computed on:
- Scalar fields: tempo, energy, loudness, key, mode, signature, danceability, etc.
- Time series: segments, beats, bars, tatums (count, mean, std, min, max)
- Delta differences for *_start vectors
- Matrices: `segments_pitches`, `segments_timbre` (mean and std for each of 12 dimensions)
- Aggregated feature: `segments_timbre_global_mean`

Any NaN/inf values are converted to 0.0.

## Re-run strategy
If the CSV exists, the script reads it instead of recomputing. To refresh features:
1. Delete the existing CSV
2. Run the command again

## Troubleshooting
| Problem | Likely cause | Solution |
|--------|--------------|----------|
| No output / exits immediately | CSV already exists | Delete the CSV and rerun |
| "No .h5 files found" | Wrong path | Verify `root_dir` |
| MemoryError | Too many files for RAM | Use `--max-files`, run in batches, merge later |
| Strange characters in titles | Mapping encoding | Ensure UTF-8 or clean input |



