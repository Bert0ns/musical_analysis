import os

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from typing import List

from lib.utils import plot_clusters_results, plot_tsne_clustering, computer_clustering_scores, salva_risultati_markdown


def spectral_clustering_classifier(features, n_clusters=5, gamma=1.0, random_state=42):
    """
    Create a classifier using spectral clustering.
    :param gamma: RBF kernel parameter
    :param n_clusters: number of clusters to create
    :param features: feature array
    :param random_state: seed for reproducibility
    :return: labels: cluster labels assigned to each sample
    """
    # Compute the affinity matrix with an RBF kernel
    affinity_matrix = rbf_kernel(features, gamma=gamma)

    # Apply spectral clustering
    model = SpectralClustering(n_clusters=n_clusters,
                               affinity='precomputed',
                               random_state=random_state,
                               )

    # Fit the model and obtain labels
    labels = model.fit_predict(affinity_matrix)
    return labels


def trova_brani_rappresentativi(features, labels, filenames, n=3):
    for cluster_id in np.unique(labels):
        # Compute cluster centroid
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_idx]
        centro = np.mean(cluster_features, axis=0)

        # Compute distance of each track from centroid
        distanze = np.linalg.norm(cluster_features - centro, axis=1)

        # Find tracks closest to centroid
        idx_piu_vicini = np.argsort(distanze)[:n]

        print(f"\nRepresentative tracks for Cluster {cluster_id}:")
        for i in idx_piu_vicini:
            print(f"  - {filenames[cluster_idx[i]]}")



def sil_dbi_score_analysis_spectral_clustering(features, gamma=0.1, range_k=(2, 20), fig_name='clustering_results/silhouette_analysis_spectral_clustering.png', show_fig=False):
    """Run Silhouette Score and Davies-Bouldin Index analysis over k values using spectral clustering.

    Returns results as a list of tuples (k, silhouette_score, dbi).
    """
    risultati = []  # (k, sil, dbi)
    for k in range(range_k[0], range_k[1]):
        try:
            labels_k = spectral_clustering_classifier(features, n_clusters=k, gamma=gamma)
            sil = silhouette_score(features, labels_k)
            dbi = davies_bouldin_score(features, labels_k)
            risultati.append((k, sil, dbi))
        except Exception:
            # Some k/gamma combinations can fail; skip and continue
            pass

    # If there are no valid results, exit without generating figures
    if not risultati:
        print("[silhouette/dbi] No valid results: skipping figure generation.")
        return risultati

    # Plot results with dual Y axes
    k_values = [r[0] for r in risultati]
    sil_values = [r[1] for r in risultati]
    dbi_values = [r[2] for r in risultati]

    best_sil_idx = int(np.argmax(sil_values))
    best_sil_k = k_values[best_sil_idx]
    best_sil = sil_values[best_sil_idx]

    best_dbi_idx = int(np.argmin(dbi_values))
    best_dbi_k = k_values[best_dbi_idx]
    best_dbi = dbi_values[best_dbi_idx]

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
        f'Silhouette & Davies-Bouldin Analysis (Spectral Clustering) (best Sil k={best_sil_k}, {best_sil:.3f}; '
        f'best DBI k={best_dbi_k}, {best_dbi:.3f})'
    )

    fig.tight_layout()
    fig.savefig(fig_name)
    if show_fig:
        plt.show()
    # Explicitly close the figure to avoid memory buildup
    plt.close(fig)

    return risultati


def run_spectral_clustering_pipeline(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        results_dir: str,
        n_clusters: int,
        gamma: float,
        report_detailed: bool = False,
):
    """Run the full spectral clustering pipeline and save plots/reports.

    Returns: labels produced by spectral clustering.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"Running spectral clustering with {n_clusters} clusters...")
    spectral_clustering_labels = spectral_clustering_classifier(features=features_reduced, n_clusters=n_clusters, gamma=gamma)

    print("Spectral clustering classification completed!")
    plot_clusters_results(filenames, features_reduced, spectral_clustering_labels, results_dir + "/clusters_plot.png")

    print("Cluster analysis (Spectral)...")
    #trova_brani_rappresentativi(features_reduced, spectral_clustering_labels, filenames)
    plot_tsne_clustering(features_reduced, spectral_clustering_labels, filenames, results_dir + "/tsne_clusters_plot.png")
    sil, dbi = computer_clustering_scores(features_reduced, spectral_clustering_labels)

    print("Generating spectral clustering Markdown report...")
    report_path = salva_risultati_markdown(
        filenames,
        features_reduced,
        spectral_clustering_labels,
        feature_names=None,
        path=results_dir + "/report_SC.md",
        n_repr=5,
        generi=music_genres,
    )
    print(f"Report generated: {report_path}")
    if report_detailed:
        report_detailed_path = salva_risultati_markdown(
            filenames,
            features_norm_original,
            spectral_clustering_labels,
            feature_names=features_names,
            path=results_dir + "/report_detailed_original_features_SC.md",
            n_repr=5,
            generi=music_genres,
        )
        print(f"Detailed report generated: {report_detailed_path}")

    # Silhouette score analysis across cluster counts
    sil_dbi_score_analysis_spectral_clustering(
        features_reduced,
        gamma=gamma,
        range_k=(2, 20),
        fig_name=results_dir + "/sil_dbi_analysis_spectral_clustering.png",
    )

    return spectral_clustering_labels, sil, dbi
