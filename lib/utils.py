import itertools
import os
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score


def plot_clusters_results(filenames, features, labels, fig_name='clustering_results/clusters.png', show_fig=False):
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
    plt.title('Rappresentazione PCA a 2 componenti dei brani clusterizzati')
    plt.xlabel('Componente Principale 1')
    plt.ylabel('Componente Principale 2')
    plt.savefig(fig_name)
    if show_fig:
        plt.show()


def plot_tsne_clustering(features, labels, filenames, fig_name='clustering_results/clusters_tsne.png', show_fig=False):
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
    if show_fig:
        plt.show()


def salva_risultati_markdown(filenames, features, labels, feature_names=None, path='clustering_results/report.md', n_repr=5, generi=None):
    """Salva un report in formato Markdown con i risultati del clustering.

    Parametri:
        filenames: lista dei nomi dei file (brani)
        features: array numpy (feature usate per il clustering o trasformate)
        labels: array delle etichette di cluster
        feature_names: lista opzionale dei nomi delle feature (se lunghezza combacia con features.shape[1])
        path: percorso del file markdown in uscita
        n_repr: numero di brani rappresentativi (più vicini al centro) da mostrare per cluster
        generi: lista dei generi (sottocartelle) corrispondenti ai brani
    """
    if len(filenames) != len(labels):
        raise ValueError("filenames e labels devono avere la stessa lunghezza")
    if generi is not None and len(generi) != len(labels):
        raise ValueError("generi deve avere stessa lunghezza di labels")

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

    # Distribuzione generi per cluster (se disponibile)
    distrib_gen = None
    conteggio_genere_tot = None
    if generi is not None:
        distrib_gen, conteggio_genere_tot = distribuzione_generi_per_cluster(labels, generi)

    lines = []
    lines.append(f"# Report Clustering Musicale")
    lines.append("")
    lines.append(f"Generato: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Numero di brani: {totale}")
    lines.append(f"Numero di cluster: {n_clusters}")
    if sil is not None:
        lines.append(f"Silhouette Score: {sil:.3f} (più alto è migliore, max 1)")
    if dbi is not None:
        lines.append(f"Davies-Bouldin Index: {dbi:.3f} (più basso è migliore)")
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

    # Sommario globale generi
    if distrib_gen is not None:
        lines.append("## Sommario Generi (totale dataset)")
        lines.append("")
        lines.append("| Genere | Brani | Percentuale |")
        lines.append("|--------|-------|-------------|")
        for g, c in sorted(conteggio_genere_tot.items(), key=lambda x: (-x[1], x[0].lower())):
            lines.append(f"| {g} | {c} | {c / totale * 100:.1f}% |")
        lines.append("")

    # Dettaglio per cluster
    lines.append("## Dettaglio per Cluster")
    for cid in sorted(counts.keys()):
        cluster_idx = np.where(labels == cid)[0]
        cluster_features = features[cluster_idx]
        lines.append("")
        lines.append(f"### Cluster {cid}")
        lines.append(f"Brani nel cluster: {counts[cid]} ({counts[cid]/totale*100:.1f}% del totale)")

        # Distribuzione generi per cluster
        if distrib_gen is not None:
            lines.append("")
            lines.append("**Distribuzione generi nel cluster:**")
            lines.append("")
            lines.append("| Genere | Brani nel cluster | % del cluster | % dei brani del genere in questo cluster |")
            lines.append("|--------|-------------------|---------------|-------------------------------------------|")
            for g, info in distrib_gen[cid].items():
                lines.append(f"| {g} | {info['count']} | {info['perc_cluster']:.1f}% | {info['perc_genere_in_tot']:.1f}% |")

        # Calcolo centro e rappresentativi
        centro = np.mean(cluster_features, axis=0)
        distanze = np.linalg.norm(cluster_features - centro, axis=1)
        ord_idx = np.argsort(distanze)
        repr_global_idx = cluster_idx[ord_idx[:min(n_repr, len(ord_idx))]]

        lines.append("")
        lines.append("**Brani rappresentativi (più vicini al centro):**")
        for gi in repr_global_idx:
            lines.append(f"- {filenames[gi]}")

        # Elenco completo dei brani
        lines.append("")
        lines.append("**Tutti i brani nel cluster:**")
        for gi in cluster_idx:
            if generi is not None:
                lines.append(f"- {filenames[gi]} (genere: {generi[gi]})")
            else:
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
    lines.append("- Le percentuali per genere aiutano a capire se il cluster è omogeneo o misto.")
    lines.append("- Se i cluster hanno dimensioni molto sbilanciate, rivedere i parametri (gamma, numero cluster, PCA, ecc.).")
    lines.append("- Usa i grafici generati per confrontare visivamente la separazione dei cluster.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return path


def computer_clustering_scores(features, labels):
    silhouette = silhouette_score(features, labels)
    davies_bouldin = davies_bouldin_score(features, labels)
    print(f"Silhouette Score: {silhouette:.3f} (più alto è migliore, max 1)")
    print(f"Davies-Bouldin Index: {davies_bouldin:.3f} (più basso è migliore)")
    return silhouette, davies_bouldin


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


def _fmt_float(v: float) -> str:
    s = f"{v}"
    return s.replace('.', 'p')


def _param_product(grid: dict):
    keys = list(grid.keys())
    for values in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, values))
