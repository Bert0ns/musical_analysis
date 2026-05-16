import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import List

from lib.utils import plot_clusters_results, plot_tsne_clustering, computer_clustering_scores, salva_risultati_markdown


def kmeans_clustering_classifier(features, n_clusters=5, n_init='auto', random_state=42, max_iter=300):
    """Run K-Means on the feature set and return labels and centroids.

    Args:
        features (np.ndarray): matrix (n_samples, n_features)
        n_clusters (int): number of clusters
        n_init: number of initializations; if 'auto' is unsupported, fall back to 10.
        random_state (int): seed for reproducibility
        max_iter (int): max number of iterations

    Returns:
        labels (np.ndarray): cluster labels
        centers (np.ndarray): centroids (n_clusters, n_features)
    """
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state, max_iter=max_iter)
    labels = model.fit_predict(features)
    return labels, model.cluster_centers_


def trova_brani_rappresentativi_kmeans(features, labels, filenames, n=3, centers=None):
    """Find the most representative tracks (closest to centroid) for each K-Means cluster.

    If centers is None, compute them from the mean of assigned points.
    """
    labels = np.asarray(labels)
    for cluster_id in np.unique(labels):
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_idx]
        if centers is not None:
            centro = centers[cluster_id]
        else:
            centro = np.mean(cluster_features, axis=0)
        distanze = np.linalg.norm(cluster_features - centro, axis=1)
        idx_piu_vicini = np.argsort(distanze)[:min(n, len(distanze))]
        print(f"\nRepresentative tracks (K-Means) for Cluster {cluster_id}:")
        for i in idx_piu_vicini:
            print(f"  - {filenames[cluster_idx[i]]}")


def sil_dbi_score_analysis_kmeans(features, range_k=(2, 20),
                                  fig_name='clustering_results/sil_dbi_analysis_kmeans.png', show_fig=False):
    """Compute Silhouette Score and Davies-Bouldin Index over a range of k with a dual-axis plot.

    Returns: list of tuples (k, silhouette, dbi)
    """
    risultati = []  # (k, sil, dbi)
    for k in range(range_k[0], range_k[1]):
        try:
            labels, _ = kmeans_clustering_classifier(features, n_clusters=k)
            sil = silhouette_score(features, labels)
            dbi = davies_bouldin_score(features, labels)
            risultati.append((k, sil, dbi))
        except Exception:
            pass
    if not risultati:
        return []

    k_values = [r[0] for r in risultati]
    sil_values = [r[1] for r in risultati]
    dbi_values = [r[2] for r in risultati]

    best_sil_idx = int(np.argmax(sil_values))
    best_sil_k = k_values[best_sil_idx]
    best_sil_val = sil_values[best_sil_idx]

    best_dbi_idx = int(np.argmin(dbi_values))
    best_dbi_k = k_values[best_dbi_idx]
    best_dbi_val = dbi_values[best_dbi_idx]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Silhouette (left axis)
    line1 = ax1.plot(k_values, sil_values, 'o-', color='tab:blue', label='Silhouette Score')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Silhouette Score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Davies-Bouldin (right axis)
    ax2 = ax1.twinx()
    line2 = ax2.plot(k_values, dbi_values, 's--', color='tab:red', label='Davies-Bouldin Index')
    ax2.set_ylabel('Davies-Bouldin Index', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Combined legend
    lines = line1 + line2
    legend_labels: List[str] = [str(ln.get_label()) for ln in lines]
    ax1.legend(lines, legend_labels, loc='best')

    plt.title(
        f'Silhouette & Davies-Bouldin Analysis (K-Means) (best Sil k={best_sil_k}, {best_sil_val:.3f}; '
        f'best DBI k={best_dbi_k}, {best_dbi_val:.3f})'
    )

    fig.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    plt.close(fig)
    return risultati


def elbow_method_kmeans(features, range_k=(2, 20), fig_name='clustering_results/elbow_analysis_kmeans.png',
                        show_fig=False):
    """Compute inertia (within-cluster sum of squares) for k values and plot the elbow curve."""
    risultati = []  # (k, inertia)
    for k in range(range_k[0], range_k[1]):
        try:
            try:
                model = KMeans(n_clusters=k, n_init='auto', random_state=42)
            except TypeError:
                model = KMeans(n_clusters=k, n_init=10, random_state=42)
            model.fit(features)
            risultati.append((k, model.inertia_))
        except Exception:
            pass
    if not risultati:
        return []
    k_values = [r[0] for r in risultati]
    inertia_values = [r[1] for r in risultati]
    fig = plt.figure(figsize=(10, 6))
    plt.plot(k_values, inertia_values, 'o-')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (within-cluster sum of squares)')
    plt.title('Elbow Method K-Means')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    plt.close(fig)
    return risultati


def run_kmeans_clustering_pipeline(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        results_dir: str,
        n_clusters: int,
        report_detailed: bool = False,
):
    """Run the full K-Means pipeline and save plots/reports.

    Returns: (labels, centers)
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nRunning K-Means clustering with {n_clusters} clusters...")
    kmeans_labels, kmeans_centers = kmeans_clustering_classifier(features_reduced, n_clusters=n_clusters)
    print("K-Means clustering completed!")

    plot_clusters_results(filenames, features_reduced, kmeans_labels, results_dir + "/clusters_plot_kmeans.png")
    plot_tsne_clustering(features_reduced, kmeans_labels, filenames, results_dir + "/tsne_clusters_plot_kmeans.png")

    print("Cluster analysis (K-Means)...")
    trova_brani_rappresentativi_kmeans(features_reduced, kmeans_labels, filenames, n=5, centers=kmeans_centers)
    sil, dbi = computer_clustering_scores(features_reduced, kmeans_labels)

    print("Generating K-Means Markdown report...")
    report_km = salva_risultati_markdown(
        filenames,
        features_reduced,
        kmeans_labels,
        feature_names=None,
        path=results_dir + "/report_KM.md",
        n_repr=5,
        generi=music_genres,
        sil=sil,
        dbi=dbi,
    )
    print(f"K-Means report generated: {report_km}")
    if report_detailed:
        report_km_detailed = salva_risultati_markdown(filenames, features_norm_original, kmeans_labels,
                                                      feature_names=features_names,
                                                      path=results_dir + "/report_detailed_original_features_KM.md",
                                                      n_repr=5, generi=music_genres, sil=sil, dbi=dbi)
        print(f"K-Means detailed report generated: {report_km_detailed}")

    # K-Means silhouette analysis
    sil_dbi_score_analysis_kmeans(
        features_reduced,
        range_k=(2, 20),
        fig_name=results_dir + "/sil_dbi_analysis_kmeans.png",
    )
    # K-Means elbow method
    elbow_method_kmeans(
        features_reduced,
        range_k=(2, 20),
        fig_name=results_dir + "/elbow_analysis_kmeans.png",
    )

    return kmeans_labels, kmeans_centers, sil, dbi


__all__ = [
    'kmeans_clustering_classifier',
    'trova_brani_rappresentativi_kmeans',
    'sil_dbi_score_analysis_kmeans',
    'elbow_method_kmeans',
    'run_kmeans_clustering_pipeline'
]
