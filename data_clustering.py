import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from datetime import datetime


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


def visualizza_risultati(filenames, features, labels, fig_name='clustering_results/clusters.png'):
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
    plt.savefig(fig_name)
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


def visualizza_tsne(features, labels, filenames, fig_name='clustering_results/clusters_tsne.png'):
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

    plt.savefig(fig_name)
    plt.show()


def valuta_cluster(features, labels, fig_name='clustering_results/cluster_evaluation.png'):
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
    plt.show()


def salva_risultati_markdown(filenames, features, labels, feature_names=None, path='clustering_results/report.md', n_repr=5):
    """Salva un report in formato Markdown con i risultati del clustering.

    Parametri:
        filenames: lista dei nomi dei file (brani)
        features: array numpy (feature usate per il clustering o trasformate)
        labels: array delle etichette di cluster
        feature_names: lista opzionale dei nomi delle feature (se lunghezza combacia con features.shape[1])
        path: percorso del file markdown in uscita
        n_repr: numero di brani rappresentativi (più vicini al centro) da mostrare per cluster
    """
    if len(filenames) != len(labels):
        raise ValueError("filenames e labels devono avere la stessa lunghezza")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_clusters = len(np.unique(labels))
    counts = {cid: int(np.sum(labels == cid)) for cid in np.unique(labels)}
    totale = len(labels)

    # Metriche globali (gestione try per sicurezza)
    try:
        sil = silhouette_score(features, labels)
    except Exception:
        sil = None
    try:
        dbi = davies_bouldin_score(features, labels)
    except Exception:
        dbi = None

    lines = []
    lines.append(f"# Report Clustering Musicale")
    lines.append("")
    lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append(f"Numero di brani: {totale}")
    lines.append("")
    lines.append(f"Numero di cluster: {n_clusters}")
    lines.append("")
    if sil is not None:
        lines.append(f"Silhouette Score: {sil:.3f} (più alto è migliore, max 1)")
        lines.append("")
    if dbi is not None:
        lines.append(f"Davies-Bouldin Index: {dbi:.3f} (più basso è migliore)")
        lines.append("")
    lines.append("")

    # Tabella riassuntiva dimensioni cluster
    lines.append("## Sommario Cluster")
    lines.append("")
    lines.append("| Cluster | Numero brani | Percentuale |")
    lines.append("|---------|--------------|-------------|")
    for cid in sorted(counts.keys()):
        perc = counts[cid] / totale * 100
        lines.append(f"| {cid} | {counts[cid]} | {perc:.1f}% |")
    lines.append("")

    # Dettaglio per cluster
    lines.append("## Dettaglio per Cluster")
    for cid in sorted(counts.keys()):
        cluster_idx = np.where(labels == cid)[0]
        cluster_features = features[cluster_idx]
        lines.append("")
        lines.append(f"### Cluster {cid}")
        lines.append(f"Brani nel cluster: {counts[cid]} ({counts[cid]/totale*100:.1f}% del totale)")

        # Calcolo centro e rappresentativi
        centro = np.mean(cluster_features, axis=0)
        distanze = np.linalg.norm(cluster_features - centro, axis=1)
        ord_idx = np.argsort(distanze)
        repr_global_idx = cluster_idx[ord_idx[:min(n_repr, len(ord_idx))]]

        lines.append("**Brani rappresentativi (più vicini al centro):**")
        for gi in repr_global_idx:
            lines.append(f"- {filenames[gi]}")

        # Elenco completo dei brani
        lines.append("")
        lines.append("**Tutti i brani nel cluster:**")
        for gi in cluster_idx:
            lines.append(f"- {filenames[gi]}")

        # Statistiche feature (solo se coerenti con feature_names)
        if feature_names is not None and len(feature_names) == features.shape[1]:
            media = np.mean(cluster_features, axis=0)
            std = np.std(cluster_features, axis=0)
            lines.append("")
            lines.append("**Statistiche Feature (media ± std):**")
            lines.append("")
            lines.append("| Feature | Media | Dev.Std |")
            lines.append("|---------|-------|---------|")

            for fname, m, s in list(zip(feature_names, media, std)):
                lines.append(f"| {fname} | {m:.3f} | {s:.3f} |")
        else:
            lines.append("")
            lines.append("(Statistiche dettagliate delle feature non mostrate: nomi non disponibili o dimensione non corrispondente.)")

    # Consigli finali
    lines.append("")
    lines.append("## Note Interpretative")
    lines.append("- I brani rappresentativi sono quelli più vicini al centro geometrico del cluster.")
    lines.append("- Se i cluster hanno dimensioni molto sbilanciate, considera di rivedere i parametri (gamma, numero di cluster, PCA, ecc.).")
    lines.append("- Usa i grafici generati per confrontare visivamente la separazione dei cluster.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return path
