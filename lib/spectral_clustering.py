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
    Crea un classificatore utilizzando l'algoritmo di spectral clustering.
    :param gamma: parametro per il kernel RBF
    :param n_clusters: numero di cluster da creare
    :param features: array delle feature
    :param random_state: seme per la riproducibilità
    :return: labels: etichette di cluster assegnate a ogni sample
    """
    # Calcolo della matrice di affinità con kernel RBF
    affinity_matrix = rbf_kernel(features, gamma=gamma)

    # Applica spectral clustering
    model = SpectralClustering(n_clusters=n_clusters,
                               affinity='precomputed',
                               random_state=random_state,
                               )

    # Addestra il modello e ottieni le etichette
    labels = model.fit_predict(affinity_matrix)
    return labels


def trova_brani_rappresentativi(features, labels, filenames, n=3):
    for cluster_id in np.unique(labels):
        # Calcola il centro del cluster
        cluster_idx = np.where(labels == cluster_id)[0]
        cluster_features = features[cluster_idx]
        centro = np.mean(cluster_features, axis=0)

        # Calcola la distanza di ogni brano dal centro
        distanze = np.linalg.norm(cluster_features - centro, axis=1)

        # Trova i brani più vicini al centro
        idx_piu_vicini = np.argsort(distanze)[:n]

        print(f"\nBrani rappresentativi del Cluster {cluster_id}:")
        for i in idx_piu_vicini:
            print(f"  - {filenames[cluster_idx[i]]}")



def sil_dbi_score_analysis_spectral_clustering(features, gamma=0.1, range_k=(2, 20), fig_name='clustering_results/silhouette_analysis_spectral_clustering.png', show_fig=False):
    """Esegue un'analisi con Silhouette Score e Davies-Bouldin Index per diversi k
    utilizzando lo spectral clustering. Restituisce i risultati come lista di tuple (k, silhouette_score, dbi).
    """
    risultati = []  # (k, sil, dbi)
    for k in range(range_k[0], range_k[1]):
        try:
            labels_k = spectral_clustering_classifier(features, n_clusters=k, gamma=gamma)
            sil = silhouette_score(features, labels_k)
            dbi = davies_bouldin_score(features, labels_k)
            risultati.append((k, sil, dbi))
        except Exception:
            # Alcune combinazioni di k/gamma possono fallire; ignoro e proseguo
            pass

    # Se non ci sono risultati validi, esci senza creare figure
    if not risultati:
        print("[silhouette/dbi] Nessun risultato valido ottenuto: salto la generazione della figura.")
        return risultati

    # Plot dei risultati con doppio asse Y
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

    # Silhouette (asse sinistro)
    line1 = ax1.plot(k_values, sil_values, 'o-', color='tab:blue', label='Silhouette Score')
    ax1.set_xlabel('Numero di cluster (k)')
    ax1.set_ylabel('Silhouette Score', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', alpha=0.4)

    # Davies-Bouldin (asse destro)
    ax2 = ax1.twinx()
    line2 = ax2.plot(k_values, dbi_values, 's--', color='tab:red', label='Davies-Bouldin Index')
    ax2.set_ylabel('Davies-Bouldin Index', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Legenda combinata
    lines = line1 + line2
    legend_labels: List[str] = [str(ln.get_label()) for ln in lines]
    ax1.legend(lines, legend_labels, loc='best')

    plt.title(
        f'Analisi Silhouette & Davies-Bouldin Spectral Clustering (best Sil k={best_sil_k}, {best_sil:.3f}; '
        f'best DBI k={best_dbi_k}, {best_dbi:.3f})'
    )

    fig.tight_layout()
    fig.savefig(fig_name)
    if show_fig:
        plt.show()
    # Chiude esplicitamente la figura per evitare accumulo in memoria
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
    """Esegue l'intera pipeline di spectral clustering e salva grafici/report.

    Ritorna: labels prodotti dallo spectral clustering.
    """
    os.makedirs(results_dir, exist_ok=True)

    print(f"Esecuzione del spectral clustering con {n_clusters} cluster...")
    spectral_clustering_labels = spectral_clustering_classifier(features=features_reduced, n_clusters=n_clusters, gamma=gamma)

    print("Spectral clustering classification completed!")
    plot_clusters_results(filenames, features_reduced, spectral_clustering_labels, results_dir + "/clusters_plot.png")

    print("Analisi dei cluster (Spectral)...")
    #trova_brani_rappresentativi(features_reduced, spectral_clustering_labels, filenames)
    plot_tsne_clustering(features_reduced, spectral_clustering_labels, filenames, results_dir + "/tsne_clusters_plot.png")
    sil, dbi = computer_clustering_scores(features_reduced, spectral_clustering_labels)

    print("Generazione report Markdown spectral clustering...")
    report_path = salva_risultati_markdown(
        filenames,
        features_reduced,
        spectral_clustering_labels,
        feature_names=None,
        path=results_dir + "/report_SC.md",
        n_repr=5,
        generi=music_genres,
    )
    print(f"Report generato: {report_path}")
    if report_detailed:
        report_detailed_path = salva_risultati_markdown(
            filenames,
            features_norm_original,
            spectral_clustering_labels,
            feature_names=features_names,
            path=results_dir + "/report_dettagliato_feature_originali_SC.md",
            n_repr=5,
            generi=music_genres,
        )
        print(f"Report dettagliato generato: {report_detailed_path}")

    # Analisi del silhouette score per diversi numeri di cluster
    sil_dbi_score_analysis_spectral_clustering(
        features_reduced,
        gamma=gamma,
        range_k=(2, 20),
        fig_name=results_dir + "/sil_dbi_analysis_spectral_clustering.png",
    )

    return spectral_clustering_labels, sil, dbi
