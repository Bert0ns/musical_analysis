import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt


def spectral_clustering_classifier(features, n_clusters=5, gamma=1.0):
    """
    Crea un classificatore utilizzando l'algoritmo di spectral clustering.

    Args:
        features: array delle feature audio
        n_clusters: numero di cluster da creare
        gamma: parametro per il kernel RBF

    Returns:
        labels: etichette di cluster assegnate ad ogni canzone
    """
    # Calcolo della matrice di affinità con kernel RBF
    affinity_matrix = rbf_kernel(features, gamma=gamma)

    # Applica spectral clustering
    model = SpectralClustering(n_clusters=n_clusters,
                               affinity='precomputed',
                               random_state=42,
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


def valuta_cluster(features, labels, fig_name='clustering_results/cluster_evaluation.png', show_fig=False):
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)

    print(f"Silhouette Score: {silhouette:.3f} (più alto è migliore, max 1)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (più basso è migliore)")

    # Prova diversi numeri di cluster
    risultati = []
    for k in range(2, 20):
        try:
            labels_k = spectral_clustering_classifier(features, n_clusters=k)
            sil = silhouette_score(features, labels_k)
            risultati.append((k, sil))
        except:
            pass

    # Visualizza i risultati
    k_values = [r[0] for r in risultati]
    sil_values = [r[1] for r in risultati]
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, sil_values, 'o-')
    plt.xlabel('Numero di cluster')
    plt.ylabel('Silhouette Score')
    plt.title('Valutazione del numero ottimale di cluster')
    plt.grid(True)
    plt.savefig(fig_name)
    if show_fig:
        plt.show()


def distribuzione_generi_per_cluster(labels, generi):
    """Calcola la distribuzione dei generi (sottocartelle) all'interno di ogni cluster.

    Ritorna:
        dict: {cluster_id: {genere: {'count': n, 'perc_cluster': p_cluster, 'perc_genere_in_tot': p_genere_su_tot_genere}}}
        dove:
          - 'perc_cluster' è la percentuale del genere sul totale del cluster
          - 'perc_genere_in_tot' è la percentuale dei brani di quel genere assegnati a quel cluster rispetto a tutti i brani di quel genere
    """
    if len(labels) != len(generi):
        raise ValueError("labels e generi devono avere stessa lunghezza")

    labels = np.asarray(labels)
    generi = np.asarray(generi)

    distribuzione = {}
    # Conteggio globale per genere
    conteggio_genere_tot = {}
    for g in generi:
        conteggio_genere_tot[g] = conteggio_genere_tot.get(g, 0) + 1

    for cid in np.unique(labels):
        idx_cluster = np.where(labels == cid)[0]
        generi_cluster = generi[idx_cluster]
        totale_cluster = len(idx_cluster)
        distribuzione[cid] = {}
        # Conteggi nel cluster
        conteggio_locale = {}
        for g in generi_cluster:
            conteggio_locale[g] = conteggio_locale.get(g, 0) + 1
        for g, c in sorted(conteggio_locale.items(), key=lambda x: (-x[1], x[0].lower())):
            perc_cluster = c / totale_cluster * 100 if totale_cluster else 0.0
            perc_genere_in_tot = c / conteggio_genere_tot[g] * 100 if conteggio_genere_tot[g] else 0.0
            distribuzione[cid][g] = {
                'count': c,
                'perc_cluster': perc_cluster,
                'perc_genere_in_tot': perc_genere_in_tot,
            }
    return distribuzione, conteggio_genere_tot


