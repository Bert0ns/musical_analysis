import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


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


def visualizza_risultati(filenames, features, labels):
    """
    Visualizza i risultati della clusterizzazione.
    """
    # Riduzione dimensionale per visualizzazione
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)

    # Mostra i cluster
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Clustering Spettrale delle Canzoni')
    plt.xlabel('Componente Principale 1')
    plt.ylabel('Componente Principale 2')
    plt.savefig('cluster_audio.png')
    plt.show()

    # Mostra i brani in ogni cluster
    print("\nContenuto dei cluster:")
    for cluster_id in np.unique(labels):
        cluster_files = [filenames[i] for i in range(len(filenames)) if labels[i] == cluster_id]
        print(f"\nCluster {cluster_id} ({len(cluster_files)} brani):")
        for file in cluster_files[:5]:  # Mostra solo i primi 5 per cluster
            print(f"  - {file}")
        if len(cluster_files) > 5:
            print(f"  ... e altri {len(cluster_files) - 5} brani")


def analizza_cluster(features, labels):
    for cluster_id in np.unique(labels):
        # Seleziona le feature dei brani in questo cluster
        cluster_features = features[labels == cluster_id]

        # Calcola statistiche per questo cluster
        media = np.mean(cluster_features, axis=0)
        std = np.std(cluster_features, axis=0)

        print(f"\nCluster {cluster_id} - {len(cluster_features)} brani:")
        print(f"  Media delle feature: {media}")
        print(f"  Deviazione standard: {std}")


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


def visualizza_tsne(features, labels, filenames):
    # Riduzione dimensionale con t-SNE (migliore per visualizzare cluster)
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)

    # Crea la visualizzazione
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Clustering dei brani (t-SNE)')

    # Aggiungi etichette per alcuni punti
    for i in range(0, len(filenames), len(filenames) // 20):  # Mostra circa 20 etichette
        plt.annotate(filenames[i].split('/')[-1], (features_2d[i, 0], features_2d[i, 1]))

    plt.savefig('tsne_clusters.png')
    plt.show()


def valuta_cluster(features, labels):
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
    plt.savefig('cluster_evaluation.png')
    plt.show()