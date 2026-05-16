from pathlib import Path
import re
import argparse
import pandas as pd
from itertools import chain


def main():
    def _last_dir(path_str: str) -> str | None:
        if not isinstance(path_str, str) or not path_str:
            return None
        parts = [p for p in sep_regex.split(path_str) if p]
        return parts[-1] if parts else None

    def infer_genre(df: pd.DataFrame) -> str | None:
        # If a label column already exists, use it
        if label_col is not None:
            return label_col
        # Try from 'filedir'
        if "filedir" in df.columns:
            genres = df["filedir"].map(_last_dir).astype("string").str.lower()
            if genres.notna().any():
                df["genre_inferred"] = genres
                return "genre_inferred"
        # Try from 'filename' (e.g., genre.00001.wav)
        if "filename" in df.columns:
            genres = df["filename"].astype("string").str.split(".").str[0].str.lower()
            if genres.notna().any():
                df["genre_inferred"] = genres
                return "genre_inferred"
        return None

    # Argument parsing
    parser = argparse.ArgumentParser(
        description="Generate a Markdown summary for an audio feature dataset."
    )
    parser.add_argument(
        "csv", help="Path to the CSV file with features (expected schema)."
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists() or not csv_path.is_file():
        raise SystemExit(f"CSV file not found: {csv_path}")

    # Load dataset
    df = pd.read_csv(csv_path)

    # Candidate label columns
    label_candidates = [
        "genre",
        "label",
        "class",
        "target",
        "y",
        "Genre",
        "GENRE",
        "Category",
    ]
    label_col = next((c for c in label_candidates if c in df.columns), None)

    # Utility helpers
    sep_regex = re.compile(r"[\\/]")

    used_label_col = infer_genre(df)

    # Common meta/non-feature columns
    meta_guess = {
        "id",
        "ID",
        "filename",
        "file",
        "filedir",
        "path",
        "track",
        "track_id",
        "title",
        "artist",
        "duration",
        "length",
        "sr",
        "sample_rate",
        "fold",
        "split",
    }
    if used_label_col:
        meta_guess.add(used_label_col)

    # Feature family definitions (regex patterns)
    families = {
        "ZCR": [r"^zcr_"],
        "RMS": [r"^rms_"],
        "Onset/Tempo": [r"^onset_rate$", r"^tempo_bpm$", r"^beats_per_sec$"],
        "Spectral: centroid/bandwidth": [r"^centroid_", r"^bandwidth_"],
        "Rolloff (85/95)": [r"^rolloff85_", r"^rolloff95_"],
        "Spectral contrast (0-6)": [r"^contrast_\d+_"],
        "Flatness": [r"^flatness_"],
        "Polynomial (poly)": [r"^poly_\d+_"],
        "Chroma STFT (12x mean/std)": [r"^chroma_stft_\d+_"],
        "Chroma CQT (12x mean/std)": [r"^chroma_cqt_\d+_"],
        "Chroma CENS (12x mean/std)": [r"^chroma_cens_\d+_"],
        "Mel-spectrogram": [r"^mel_spec_"],
        "MFCC (0-19)": [r"^mfcc_\d+_"],
        "Tonnetz (0-5)": [r"^tonnetz_\d+_"],
        "HPSS ratios": [r"^hpss_harm_ratio$", r"^hpss_perc_ratio$"],
    }

    # Column matching per family
    all_cols = list(df.columns)
    family_matches = {
        fam: [c for c in all_cols if any(re.search(pat, c) for pat in patterns)]
        for fam, patterns in families.items()
    }

    # Merge known features across families
    features_in_families = sorted(set(chain.from_iterable(family_matches.values())))

    # Select numeric features and identify unclassified numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)]
    unknown_features = [
        c for c in numeric_cols if c not in features_in_families and c not in meta_guess
    ]

    # Final list of features to analyze
    feature_cols = features_in_families + unknown_features

    # Actual meta columns
    meta_cols = [c for c in df.columns if c not in feature_cols]

    # Primary stats
    n_samples = len(df)
    n_features = len(feature_cols)
    n_meta = len(meta_cols)
    missing_total = int(df[feature_cols].isna().sum().sum()) if n_features else 0
    missing_pct = (
        (missing_total / (n_samples * n_features) * 100)
        if (n_samples and n_features)
        else 0.0
    )
    dup_rows = int(df.duplicated(subset=feature_cols).sum()) if n_features else 0

    # Class stats (if label available)
    if used_label_col and used_label_col in df.columns:
        counts = df[used_label_col].value_counts(dropna=False)
        n_classes = counts.shape[0]
        per_class_min = int(counts.min())
        per_class_med = float(counts.median())
        per_class_max = int(counts.max())
    else:
        counts = pd.Series(dtype=int)
        n_classes = None
        per_class_min = per_class_med = per_class_max = None

    # Overall summary table
    rows = [
        ("Source file", csv_path.name),
        ("Sample count (rows)", f"{n_samples}"),
        ("Total feature count", f"{n_features}"),
        (
            "Feature families present",
            f"{sum(1 for k, v in family_matches.items() if len(v) > 0) + (1 if unknown_features else 0)}",
        ),
        ("Unclassified features", f"{len(unknown_features)}"),
        (
            "Meta/non-feature columns",
            f"{n_meta} ({', '.join(meta_cols[:6])}{'...' if n_meta > 6 else ''})",
        ),
        ("Label column used", f"{used_label_col if used_label_col else '-'}"),
        ("Unique genres/classes", f"{n_classes if n_classes is not None else '-'}"),
        (
            "Tracks per class (min/median/max)",
            (
                f"{per_class_min} / {per_class_med:.1f} / {per_class_max}"
                if n_classes is not None
                else "-"
            ),
        ),
        ("Missing values in features", f"{missing_total} ({missing_pct:.2f}%)"),
        ("Duplicate rows (features)", f"{dup_rows}"),
    ]

    summary_md_lines = ["| Field | Value |", "| --- | --- |"]
    summary_md_lines += [f"| {k} | {v} |" for k, v in rows]
    summary_md = "\n".join(summary_md_lines)

    # Table of feature families
    family_rows = []
    for fam, cols in family_matches.items():
        if cols:
            family_rows.append((fam, len(cols)))
    if unknown_features:
        family_rows.append(("Other features (unclassified)", len(unknown_features)))

    fam_md = None
    if family_rows:
        fam_md_lines = ["| Family | Column count |", "| --- | ---: |"]
        fam_md_lines += [f"| {fam} | {cnt} |" for fam, cnt in family_rows]
        fam_md = "\n".join(fam_md_lines)

    # Class distribution (if label available)
    dist_md = None
    if n_classes is not None:
        dist = pd.DataFrame(
            {
                "Genre/Class": counts.index.astype(str),
                "Count": counts.values,
                "Percentage": (counts.values / n_samples * 100.0).round(2),
            }
        ).sort_values("Count", ascending=False)

        dist_lines = ["| Genre/Class | Count | Percentage |", "| --- | ---: | ---: |"]
        dist_lines += [
            f"| {r['Genre/Class']} | {int(r['Count'])} | {r['Percentage']:.2f}% |"
            for _, r in dist.iterrows()
        ]
        dist_md = "\n".join(dist_lines)

    # Compose Markdown file
    out_parts = [
        f"# Dataset summary: {csv_path.stem}",
        "",
        f"Source: `{csv_path.name}`",
        "",
        "## Summary table",
        summary_md,
    ]
    if fam_md:
        out_parts += ["", "## Features by family", fam_md]
    if dist_md:
        out_parts += ["", "## Class distribution", dist_md]

    out_text = "\n".join(out_parts) + "\n"

    # Save Markdown file in the same folder as the CSV
    out_path = csv_path.with_name(f"{csv_path.stem}_summary.md")
    out_path.write_text(out_text, encoding="utf-8")

    # Minimal final message
    print(f"Markdown report saved to: {out_path}")


if __name__ == "__main__":
    main()
