import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

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
    #model = HDBSCAN(min_cluster_size=min_samples, min_samples=min_samples, metric=metric, n_jobs=1, cluster_selection_method='eom', algorithm='auto', leaf_size=30, allow_single_cluster=False)
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


def sil_dbi_analysis_dbscan(features, eps_values, min_samples=5, metric='euclidean', fig_name='clustering_results/silhouette_analysis_dbscan.png', show_fig=False):
    """Valuta silhouette score e Davies-Bouldin Index al variare di eps.

    Considera solo risultati con almeno 2 cluster validi (escludendo -1) e meno del 90% di noise.
    Ritorna lista di tuple (eps, n_clusters, noise_ratio, silhouette or None, dbi or None).
    """
    risultati = []  # (eps, n_clusters, noise_ratio, sil, dbi)
    for eps in eps_values:
        labels, _ = dbscan_clustering_classifier(features, eps=eps, min_samples=min_samples, metric=metric)
        unique_clusters = [c for c in np.unique(labels) if c != -1]
        noise_ratio = np.sum(labels == -1) / len(labels)
        sil = None
        dbi = None
        if len(unique_clusters) >= 2 and noise_ratio < 0.9:
            valid_mask = labels != -1
            try:
                sil = silhouette_score(features[valid_mask], labels[valid_mask])  # silhouette sui soli punti non-noise
            except Exception:
                sil = None
            try:
                dbi = davies_bouldin_score(features[valid_mask], labels[valid_mask])  # DBI sui soli punti non-noise
            except Exception:
                dbi = None
        risultati.append((eps, len(unique_clusters), noise_ratio, sil, dbi))

    # Plot con doppio asse Y
    valid_sil = [(e, s) for (e, nc, nr, s, d) in risultati if s is not None]
    valid_dbi = [(e, d) for (e, nc, nr, s, d) in risultati if d is not None]

    if not valid_sil and not valid_dbi:
        # Nessun risultato da plottare
        return risultati

    fig, ax1 = plt.subplots(figsize=(10, 6))

    lines = []
    labels_lines = []

    best_sil_str = None
    best_dbi_str = None

    if valid_sil:
        x_sil = [v[0] for v in valid_sil]
        y_sil = [v[1] for v in valid_sil]
        l1 = ax1.plot(x_sil, y_sil, 'o-', color='tab:blue', label='Silhouette Score (no-noise)')
        lines += l1
        labels_lines += [str(l.get_label()) for l in l1]
        ax1.set_xlabel('eps')
        ax1.set_ylabel('Silhouette Score (solo punti non-noise)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.grid(True, which='both', linestyle='--', alpha=0.4)
        best_idx = int(np.argmax(y_sil))
        best_sil_str = f"best Sil eps={x_sil[best_idx]}, {y_sil[best_idx]:.3f}"

    ax2 = None
    if valid_dbi:
        x_dbi = [v[0] for v in valid_dbi]
        y_dbi = [v[1] for v in valid_dbi]
        ax2 = ax1.twinx()
        l2 = ax2.plot(x_dbi, y_dbi, 's--', color='tab:red', label='Davies-Bouldin Index (no-noise)')
        lines += l2
        labels_lines += [str(l.get_label()) for l in l2]
        ax2.set_ylabel('Davies-Bouldin Index (solo punti non-noise)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        best_idx_d = int(np.argmin(y_dbi))
        best_dbi_str = f"best DBI eps={x_dbi[best_idx_d]}, {y_dbi[best_idx_d]:.3f}"

    if lines:
        ax1.legend(lines, labels_lines, loc='best')

    # Titolo con riassunto dei best
    title_parts = []
    if best_sil_str:
        title_parts.append(best_sil_str)
    if best_dbi_str:
        title_parts.append(best_dbi_str)
    title_suffix = ' ; '.join(title_parts) if title_parts else ''
    plt.title(f'Analisi Silhouette & Davies-Bouldin DBSCAN{(" (" + title_suffix + ")") if title_suffix else ""}')

    fig.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    plt.close(fig)

    return risultati


def k_distance_plot_dbscan(features, k=5, metric='euclidean',
                           fig_name='clustering_results/k_distance_dbscan.png',
                           show_fig=False):
    """Genera il k-distance plot per stimare eps (esclude il punto stesso)."""
    n = features.shape[0]
    if n == 0:
        return np.array([])

    # usa k+1 vicini per escludere il punto stesso (distanza 0)
    k_eff = min(k + 1, n)
    neigh = NearestNeighbors(n_neighbors=k_eff, metric=metric)
    neigh.fit(features)
    distances, _ = neigh.kneighbors(features)

    # colonna del k-esimo vicino reale (escludendo self); fallback se k>=n
    col = min(k, k_eff - 1)
    k_dist = np.sort(distances[:, col])

    fig = plt.figure(figsize=(10, 6))
    plt.plot(k_dist)
    plt.xlabel('Punti ordinati')
    plt.ylabel(f'Distanza al {k}-esimo vicino (metrica: {metric})')
    plt.title(f'K-distance Plot (k={k}) per stima eps DBSCAN')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    if show_fig:
        plt.show()
    plt.close(fig)
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
        eps_values_to_compute_sil_and_dbi = np.linspace(0.1, 2.0, 25)
):
    """Esegue l'intera pipeline di DBSCAN e salva grafici/report.

    Ritorna: (labels, modello)
    """
    os.makedirs(results_dir, exist_ok=True)

    print("\nAnalisi esplorativa per DBSCAN (k-distance plot)...")
    k_distance_plot_dbscan(features_reduced, k=min_samples, metric=metric, fig_name=results_dir + "/k_distance_dbscan.png")

    print("Analisi silhouette su diversi eps per DBSCAN...")
    try:
        sil_dbi_analysis_dbscan(
            features_reduced,
            eps_values_to_compute_sil_and_dbi,
            min_samples=min_samples,
            metric=metric,
            fig_name=results_dir + "/sil_dbi_analysis_dbscan.png",
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


# --- Helper: scegli lo spazio per DBSCAN
def choose_dbscan_space(features_reduced: np.ndarray, features_norm: np.ndarray, mode: str) -> np.ndarray:
    if mode == 'normalized':
        # Usa le feature normalizzate pre-PCA
        return features_norm
    elif mode == 'reduced_minmax':
        # Riapplica MinMax alle componenti PCA
        return MinMaxScaler().fit_transform(features_reduced)
    else:  # 'reduced'
        # Usa direttamente le componenti PCA
        return features_reduced

__all__ = [
    'dbscan_clustering_classifier',
    'trova_brani_rappresentativi_dbscan',
    'sil_dbi_analysis_dbscan',
    'k_distance_plot_dbscan',
    'run_dbscan_clustering_pipeline',
    'choose_dbscan_space',
]
