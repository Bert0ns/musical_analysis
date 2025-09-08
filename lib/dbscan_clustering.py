import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def dbscan_clustering_classifier(features, eps=0.5, min_samples=5, metric='euclidean'):
    """Esegue DBSCAN e restituisce le etichette e il modello.

    Parametri:
        features (np.ndarray): dati (n_samples, n_features)
        eps (float): raggio di vicinanza
        min_samples (int): punti minimi per formare un core point
        metric (str): metrica di distanza

    Ritorna:
        labels (np.ndarray): etichette cluster (-1 = noise)
        model (DBSCAN): modello addestrato
    """
    model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
    labels = model.fit_predict(features)
    return labels, model


def trova_brani_rappresentativi_dbscan(features, labels, filenames, n=3):
    """Trova brani rappresentativi per ogni cluster DBSCAN (esclude noise -1)."""
    labels = np.asarray(labels)
    cluster_ids = [cid for cid in np.unique(labels) if cid != -1]
    for cluster_id in cluster_ids:
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_idx]
        centro = np.mean(cluster_features, axis=0)
        distanze = np.linalg.norm(cluster_features - centro, axis=1)
        idx_piu_vicini = np.argsort(distanze)[:min(n, len(distanze))]
        print(f"\nBrani rappresentativi (DBSCAN) del Cluster {cluster_id}:")
        for i in idx_piu_vicini:
            print(f"  - {filenames[cluster_idx[i]]}")
    if -1 in np.unique(labels):
        noise_count = int(np.sum(labels == -1))
        print(f"\nPunti di rumore (noise, label -1): {noise_count}")


def silhouette_analysis_dbscan(features, eps_values, min_samples=5, metric='euclidean', fig_name='clustering_results/silhouette_analysis_dbscan.png', show_fig=False):
    """Valuta silhouette score al variare di eps.

    Vengono considerati solo i risultati con almeno 2 cluster validi e meno del 90% di noise.
    Ritorna lista di tuple (eps, n_clusters, noise_ratio, silhouette or None).
    """
    risultati = []
    for eps in eps_values:
        labels, _ = dbscan_clustering_classifier(features, eps=eps, min_samples=min_samples, metric=metric)
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        noise_ratio = np.sum(labels == -1) / len(labels)
        sil = None
        if len(unique_clusters) >= 2 and noise_ratio < 0.9:
            try:
                sil = silhouette_score(features, labels[labels != -1])  # silhouette sui soli punti assegnati
            except Exception:
                sil = None
        risultati.append((eps, len(unique_clusters), noise_ratio, sil))

    # Plot
    valid = [(e, s) for (e, nc, nr, s) in risultati if s is not None]
    if valid:
        x_plot = [v[0] for v in valid]
        y_plot = [v[1] for v in valid]
        best_idx = int(np.argmax(y_plot))
        best_eps = x_plot[best_idx]
        best_sil = y_plot[best_idx]
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, 'o-')
        plt.xlabel('eps')
        plt.ylabel('Silhouette Score (solo punti non-noise)')
        plt.title(f'Analisi Silhouette DBSCAN (best eps={best_eps}, sil={best_sil:.3f})')
        plt.grid(True)
        plt.savefig(fig_name)
        if show_fig:
            plt.show()
    return risultati


def k_distance_plot_dbscan(features, k=5, fig_name='clustering_results/k_distance_dbscan.png', show_fig=False):
    """Genera il k-distance plot (ordinando la distanza al k-esimo vicino) per aiutare a stimare eps.

    Suggerimento: il "gomito" della curva può indicare un buon valore di eps.
    """
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(features)
    distances, _ = neigh.kneighbors(features)
    # la distanza al k-esimo vicino è la colonna k-1 (ordinata successivamente)
    k_dist = np.sort(distances[:, k-1])
    plt.figure(figsize=(10, 6))
    plt.plot(k_dist)
    plt.xlabel('Punti ordinati')
    plt.ylabel(f'Distanza al {k}-esimo vicino')
    plt.title(f'K-distance Plot (k={k}) per stima eps DBSCAN')
    plt.grid(True)
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    return k_dist


__all__ = [
    'dbscan_clustering_classifier',
    'trova_brani_rappresentativi_dbscan',
    'silhouette_analysis_dbscan',
    'k_distance_plot_dbscan'
]

