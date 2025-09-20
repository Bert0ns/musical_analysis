import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from lib.utils import salva_risultati_markdown, computer_clustering_scores, plot_tsne_clustering, plot_clusters_results


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


def run_dbscan_clustering_pipeline(
        filenames,
        features_reduced,
        features_norm_original,
        features_names,
        music_genres,
        results_dir: str,
        eps: float,
        min_samples: int,
        metric: str,
        report_detailed: bool = False,
):
    """Esegue l'intera pipeline di DBSCAN e salva grafici/report.

    Ritorna: (labels, modello)
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\nAnalisi esplorativa per DBSCAN (k-distance plot)...")
    try:
        k_distance_plot_dbscan(features_reduced, k=min_samples, fig_name=results_dir + "/k_distance_dbscan.png")
    except Exception as e:
        print(f"Errore k-distance plot DBSCAN: {e}")

    print("Analisi silhouette su diversi eps per DBSCAN...")
    eps_values = np.linspace(0.1, 2.0, 25)
    try:
        silhouette_analysis_dbscan(
            features_reduced,
            eps_values,
            min_samples=min_samples,
            metric=metric,
            fig_name=results_dir + "/silhouette_analysis_dbscan.png",
        )
    except Exception as e:
        print(f"Errore silhouette analysis DBSCAN: {e}")

    print(f"Esecuzione DBSCAN finale (eps={eps}, min_samples={min_samples})...")
    dbscan_labels, dbscan_model = dbscan_clustering_classifier(features_reduced, eps=eps, min_samples=min_samples, metric=metric)
    unique_dbscan_clusters = [c for c in np.unique(dbscan_labels) if c != -1]
    print(f"Cluster trovati (senza noise): {len(unique_dbscan_clusters)} - con noise label -1 totale classi: {len(np.unique(dbscan_labels))}")
    noise_ratio = np.sum(dbscan_labels == -1) / len(dbscan_labels)
    print(f"Percentuale noise: {noise_ratio * 100:.1f}%")

    plot_clusters_results(filenames, features_reduced, dbscan_labels, results_dir + "/clusters_plot_dbscan.png")
    plot_tsne_clustering(features_reduced, dbscan_labels, filenames, results_dir + "/tsne_clusters_plot_dbscan.png")

    trova_brani_rappresentativi_dbscan(features_reduced, dbscan_labels, filenames, n=5)

    # Metriche (silhouette e Davies-Bouldin) evitando di calcolare se cluster insufficienti
    try:
        if len(unique_dbscan_clusters) >= 2:
            # Usiamo solo punti non-noise per silhouette e DB
            valid_mask = dbscan_labels != -1
            sil, dbi = computer_clustering_scores(features_reduced[valid_mask], dbscan_labels[valid_mask])
        else:
            print("Metriche DBSCAN saltate: meno di 2 cluster validi.")
            sil = None
            dbi = None
    except Exception as e:
        print(f"Errore calcolo metriche DBSCAN: {e}")
        sil = None
        dbi = None

    print("Generazione report Markdown DBSCAN...")
    report_dbscan = salva_risultati_markdown(
        filenames,
        features_reduced,
        dbscan_labels,
        feature_names=None,
        path=results_dir + "/report_DBSCAN.md",
        n_repr=5,
        generi=music_genres,
    )
    print(f"Report DBSCAN generato: {report_dbscan}")
    if report_detailed:
        report_dbscan_detailed = salva_risultati_markdown(
            filenames,
            features_norm_original,
            dbscan_labels,
            feature_names=features_names,
            path=results_dir + "/report_dettagliato_feature_originali_DBSCAN.md",
            n_repr=5,
            generi=music_genres,
        )
        print(f"Report dettagliato DBSCAN generato: {report_dbscan_detailed}")

    return dbscan_labels, dbscan_model, sil, dbi


__all__ = [
    'dbscan_clustering_classifier',
    'trova_brani_rappresentativi_dbscan',
    'silhouette_analysis_dbscan',
    'k_distance_plot_dbscan',
    'run_dbscan_clustering_pipeline',
]

