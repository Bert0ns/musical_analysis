import itertools
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score


def plot_clusters_results(
    filenames,
    features,
    labels,
    fig_name="clustering_results/clusters.png",
    show_fig=False,
):
    """
    Visualize clustering results.
    """
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Plot clusters
    fig = plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="viridis", alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("PCA 2D projection of clustered tracks")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    # Explicitly close the figure to avoid memory buildup
    plt.close(fig)


def plot_tsne_clustering(
    features,
    labels,
    filenames,
    fig_name="clustering_results/clusters_tsne.png",
    show_fig=False,
):
    # Dimensionality reduction with t-SNE (better for visualizing clusters)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Create the visualization
    fig = plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        features_2d[:, 0], features_2d[:, 1], c=labels, cmap="tab10", alpha=0.7
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title("Track clustering (t-SNE)")

    # Add labels for a subset of points
    for i in range(0, len(filenames), max(1, len(filenames) // 20)):
        plt.annotate(
            filenames[i].split("/")[-1], (features_2d[i, 0], features_2d[i, 1])
        )

    plt.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    # Explicitly close the figure to avoid memory buildup
    plt.close(fig)


def salva_risultati_markdown(
    filenames,
    features,
    labels,
    feature_names=None,
    path="clustering_results/report.md",
    n_repr=5,
    generi=None,
    sil=None,
    dbi=None,
    show_all_samples_in_clusters=False,
):
    """Save a Markdown report with clustering results.

    Args:
        filenames: list of file names (tracks)
        features: numpy array (features used for clustering or transformed)
        labels: array of cluster labels
        feature_names: optional list of feature names (if length matches features.shape[1])
        path: output markdown file path
        n_repr: number of representative tracks (closest to centroid) to show per cluster
        generi: list of genres (subfolders) corresponding to tracks
    """
    if len(filenames) != len(labels):
        raise ValueError("filenames and labels must have the same length")
    if generi is not None and len(generi) != len(labels):
        raise ValueError("generi must have the same length as labels")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_clusters = len(np.unique(labels))
    counts = {cid: int(np.sum(labels == cid)) for cid in np.unique(labels)}
    totale = len(labels)

    # Genre distribution per cluster (if available)
    distrib_gen = None
    conteggio_genere_tot = None
    if generi is not None:
        distrib_gen, conteggio_genere_tot = distribuzione_generi_per_cluster(
            labels, generi
        )

    lines = []
    lines.append(f"# Music Clustering Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Number of tracks: {totale}")
    lines.append("")
    lines.append(f"Number of clusters: {n_clusters}")
    lines.append("")
    if sil is not None:
        lines.append(f"Silhouette Score: {sil:.3f} (higher is better, max 1)")
        lines.append("")
    if dbi is not None:
        lines.append(f"Davies-Bouldin Index: {dbi:.3f} (lower is better)")
        lines.append("")

    # Cluster size summary table
    lines.append("## Cluster Summary")
    lines.append("")
    lines.append("| Cluster | Tracks | Percentage |")
    lines.append("|---------|--------------|-------------|")
    for cid in sorted(counts.keys()):
        perc = counts[cid] / totale * 100
        lines.append(f"| {cid} | {counts[cid]} | {perc:.1f}% |")
    lines.append("")

    # Overall genre summary
    if distrib_gen is not None:
        lines.append("## Genre Summary (full dataset)")
        lines.append("")
        lines.append("| Genre | Tracks | Percentage |")
        lines.append("|--------|-------|-------------|")
        for g, c in sorted(
            conteggio_genere_tot.items(), key=lambda x: (-x[1], x[0].lower())
        ):
            lines.append(f"| {g} | {c} | {c / totale * 100:.1f}% |")
        lines.append("")

    # Per-cluster detail
    lines.append("## Cluster Details")
    for cid in sorted(counts.keys()):
        cluster_idx = np.where(labels == cid)[0]
        cluster_features = features[cluster_idx]
        lines.append("")
        lines.append(f"### Cluster {cid}")
        lines.append(
            f"Tracks in cluster: {counts[cid]} ({counts[cid]/totale*100:.1f}% of total)"
        )

        # Genre distribution per cluster
        if distrib_gen is not None:
            lines.append("")
            lines.append("**Genre distribution within cluster:**")
            lines.append("")
            lines.append(
                "| Genre | Tracks in cluster | % of cluster | % of genre tracks in this cluster |"
            )
            lines.append(
                "|-------|-------------------|--------------|-------------------------------------------|"
            )
            for g, info in distrib_gen[cid].items():
                lines.append(
                    f"| {g} | {info['count']} | {info['perc_cluster']:.1f}% | {info['perc_genere_in_tot']:.1f}% |"
                )

        # Compute centroid and representatives
        centro = np.mean(cluster_features, axis=0)
        distanze = np.linalg.norm(cluster_features - centro, axis=1)
        ord_idx = np.argsort(distanze)
        repr_global_idx = cluster_idx[ord_idx[: min(n_repr, len(ord_idx))]]

        lines.append("")
        lines.append("**Representative tracks (closest to centroid):**")
        for gi in repr_global_idx:
            lines.append(f"- {filenames[gi]}")

        # Full track list
        if show_all_samples_in_clusters:
            lines.append("")
            lines.append("**All tracks in the cluster:**")
            for gi in cluster_idx:
                if generi is not None:
                    lines.append(f"- {filenames[gi]} (genre: {generi[gi]})")
                else:
                    lines.append(f"- {filenames[gi]}")

        # Feature statistics (only if consistent with feature_names)
        if feature_names is not None and len(feature_names) == features.shape[1]:
            media = np.mean(cluster_features, axis=0)
            std = np.std(cluster_features, axis=0)
            lines.append("")
            lines.append("**Feature Statistics (mean +/- std):**")
            lines.append("")
            lines.append("| Feature | Mean | Std. Dev. |")
            lines.append("|---------|-------|---------|")
            for fname, m, s in list(zip(feature_names, media, std)):
                lines.append(f"| {fname} | {m:.3f} | {s:.3f} |")
        else:
            lines.append("")
            lines.append(
                "(Detailed feature stats not shown: names unavailable or dimension mismatch.)"
            )

    # Final notes
    lines.append("")
    lines.append("## Interpretation Notes")
    lines.append("- Representative tracks are those closest to the cluster centroid.")
    lines.append(
        "- Genre percentages help assess whether a cluster is homogeneous or mixed."
    )
    lines.append(
        "- If cluster sizes are highly imbalanced, revisit parameters (gamma, number of clusters, PCA, etc.)."
    )
    lines.append("- Use the generated plots to visually compare cluster separation.")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path


def computer_clustering_scores(features, labels):
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    print(f"Silhouette Score: {silhouette:.3f} (higher is better, max 1)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")
    return silhouette, davies_bouldin


def distribuzione_generi_per_cluster(labels, generi):
    """Compute genre (subfolder) distribution within each cluster.

    Returns:
        dict: {cluster_id: {genre: {'count': n, 'perc_cluster': p_cluster, 'perc_genere_in_tot': p_genre_in_total}}}
        where:
          - 'perc_cluster' is the genre percentage within the cluster
          - 'perc_genere_in_tot' is the percentage of that genre assigned to this cluster out of all tracks of that genre
    """
    if len(labels) != len(generi):
        raise ValueError("labels and generi must have the same length")

    labels = np.asarray(labels)
    generi = np.asarray(generi)

    distribuzione = {}
    # Global count per genre
    conteggio_genere_tot = {}
    for g in generi:
        conteggio_genere_tot[g] = conteggio_genere_tot.get(g, 0) + 1

    for cid in np.unique(labels):
        idx_cluster = np.where(labels == cid)[0]
        generi_cluster = generi[idx_cluster]
        totale_cluster = len(idx_cluster)
        distribuzione[cid] = {}
        # Counts within cluster
        conteggio_locale = {}
        for g in generi_cluster:
            conteggio_locale[g] = conteggio_locale.get(g, 0) + 1
        for g, c in sorted(
            conteggio_locale.items(), key=lambda x: (-x[1], x[0].lower())
        ):
            perc_cluster = c / totale_cluster * 100 if totale_cluster else 0.0
            perc_genere_in_tot = (
                c / conteggio_genere_tot[g] * 100 if conteggio_genere_tot[g] else 0.0
            )
            distribuzione[cid][g] = {
                "count": c,
                "perc_cluster": perc_cluster,
                "perc_genere_in_tot": perc_genere_in_tot,
            }
    return distribuzione, conteggio_genere_tot


def fmt_float(v: float) -> str:
    s = f"{v}"
    return s.replace(".", "p")


def param_product(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))
