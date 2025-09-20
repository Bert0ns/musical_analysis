import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from lib.utils import plot_clusters_results, plot_tsne_clustering, computer_clustering_scores, salva_risultati_markdown


def kmeans_clustering_classifier(features, n_clusters=5, n_init='auto', random_state=42, max_iter=300):
    """Esegue K-Means sul set di feature e restituisce le etichette e i centroidi.

    Parametri:
        features (np.ndarray): matrice (n_samples, n_features)
        n_clusters (int): numero di cluster
        n_init: numero di inizializzazioni; se 'auto' e non supportato dalla versione di scikit-learn si fa fallback a 10.
        random_state (int): seme random per riproducibilità
        max_iter (int): numero massimo iterazioni

    Ritorna:
        labels (np.ndarray): etichette di cluster
        centers (np.ndarray): centroidi (n_clusters, n_features)
    """
    model = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=random_state, max_iter=max_iter)
    labels = model.fit_predict(features)
    return labels, model.cluster_centers_


def trova_brani_rappresentativi_kmeans(features, labels, filenames, n=3, centers=None):
    """Trova i brani più rappresentativi (più vicini al centro) per ogni cluster K-Means.

    Se centers è None, li calcola dalla media dei punti assegnati.
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
        print(f"\nBrani rappresentativi (K-Means) del Cluster {cluster_id}:")
        for i in idx_piu_vicini:
            print(f"  - {filenames[cluster_idx[i]]}")


def silhouette_score_analysis_kmeans(features, range_k=(2, 20),
                                     fig_name='clustering_results/silhouette_analysis_kmeans.png', show_fig=False):
    """Calcola il silhouette score per un range di k e genera il grafico.

    Ritorna: lista di tuple (k, silhouette)
    """
    risultati = []
    for k in range(range_k[0], range_k[1]):
        try:
            labels, _ = kmeans_clustering_classifier(features, n_clusters=k)
            sil = silhouette_score(features, labels)
            risultati.append((k, sil))
        except Exception:
            pass
    if not risultati:
        return []
    k_values = [r[0] for r in risultati]
    sil_values = [r[1] for r in risultati]
    best_idx = int(np.argmax(sil_values))
    best_k = k_values[best_idx]
    best_sil = sil_values[best_idx]
    fig = plt.figure(figsize=(10, 6))
    plt.plot(k_values, sil_values, 'o-')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Silhouette Score')
    plt.title(f'Analisi Silhouette K-Means (k={best_k}, Silhouette={best_sil:.3f})')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    plt.close(fig)
    return risultati


def elbow_method_kmeans(features, range_k=(2, 20), fig_name='clustering_results/elbow_analysis_kmeans.png',
                        show_fig=False):
    """Calcola l'inertia (somma quadrati entro-cluster) per diversi k e genera grafico elbow."""
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
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Inertia (Somma quadrati entro-cluster)')
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
    """Esegue l'intera pipeline di K-Means e salva grafici/report.

    Ritorna: (labels, centri)
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"\nEsecuzione del K-Means clustering con {n_clusters} cluster...")
    kmeans_labels, kmeans_centers = kmeans_clustering_classifier(features_reduced, n_clusters=n_clusters)
    print("K-Means clustering completed!")

    plot_clusters_results(filenames, features_reduced, kmeans_labels, results_dir + "/clusters_plot_kmeans.png")
    plot_tsne_clustering(features_reduced, kmeans_labels, filenames, results_dir + "/tsne_clusters_plot_kmeans.png")

    print("Analisi dei cluster (K-Means)...")
    trova_brani_rappresentativi_kmeans(features_reduced, kmeans_labels, filenames, n=5, centers=kmeans_centers)
    sil, dbi = computer_clustering_scores(features_reduced, kmeans_labels)

    print("Generazione report Markdown K-Means...")
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
    print(f"Report K-Means generato: {report_km}")
    if report_detailed:
        report_km_detailed = salva_risultati_markdown(filenames, features_norm_original, kmeans_labels,
                                                      feature_names=features_names,
                                                      path=results_dir + "/report_dettagliato_feature_originali_KM.md",
                                                      n_repr=5, generi=music_genres, sil=sil, dbi=dbi)
        print(f"Report dettagliato K-Means generato: {report_km_detailed}")

    # Analisi silhouette K-Means
    silhouette_score_analysis_kmeans(
        features_reduced,
        range_k=(2, 20),
        fig_name=results_dir + "/silhouette_analysis_kmeans.png",
    )
    # Elbow method K-Means
    elbow_method_kmeans(
        features_reduced,
        range_k=(2, 20),
        fig_name=results_dir + "/elbow_analysis_kmeans.png",
    )

    return kmeans_labels, kmeans_centers, sil, dbi


__all__ = [
    'kmeans_clustering_classifier',
    'trova_brani_rappresentativi_kmeans',
    'silhouette_score_analysis_kmeans',
    'elbow_method_kmeans',
    'run_kmeans_clustering_pipeline'
]
