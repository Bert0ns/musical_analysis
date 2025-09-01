"""
Script per generare immagini illustrative sull'uso del Silhouette Score nel clustering.

Figure prodotte (cartella: figs_silhouette/):

1. silhouette_good.png
   Silhouette plot per un clustering "buono" (K-Means su 4 blob ben separati).
2. silhouette_k_comparison.png
   Confronto silhouette plot per k=2, k=4 (vicino al valore ottimale) e k=6 (sovra-frammentazione).
3. silhouette_avg_vs_k.png
   Andamento del Silhouette medio al variare di k (metodo di supporto alla scelta del numero di cluster).
4. silhouette_metric_comparison.png
   Confronto della silhouette usando distanza euclidea vs coseno (dati normalizzati) per mostrare l’effetto della metrica.
5. silhouette_dbscan.png
   Esempio di silhouette con DBSCAN (noise escluso dal calcolo e visualizzato separatemente).
6. silhouette_pca_effect.png (opzionale)
   Effetto della riduzione dimensionale (PCA) sul Silhouette medio per K-Means a k fisso.

Dipendenze:
- Python 3.9+
- numpy
- matplotlib
- scikit-learn
- seaborn (solo per una palette opzionale, può essere omesso)

Installazione rapida (se mancano pacchetti):
pip install numpy matplotlib scikit-learn seaborn

Esecuzione:
python generate_silhouette_figures.py

Note:
- Il Silhouette score s(i) per un punto i è definito come:
      s(i) = (b(i) - a(i)) / max(a(i), b(i))
  dove a(i) è la distanza media di i dagli altri punti del proprio cluster,
  e b(i) è la distanza media più bassa verso un cluster alternativo (il "nearest cluster").
- Valori vicini a 1 indicano buona assegnazione; intorno a 0 indicano punti su un confine;
  negativi indicano possibile assegnazione errata.

Le didascalie suggerite sono stampate a fine esecuzione (vedere print()).
"""

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "figs_silhouette"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Utility per silhouette plot
# ---------------------------

def silhouette_plot(X, labels, metric="euclidean", title="", ax=None, show_avg=True, color_cycle=None):
    """
    Crea un silhouette plot per i cluster in 'labels'.
    Ignora (non disegna) cluster con un solo punto.
    Restituisce il silhouette medio (esclusi eventuali cluster di taglia 1).
    """
    if ax is None:
        ax = plt.gca()

    unique_clusters = [c for c in np.unique(labels) if c != -1]  # escludi noise se presente
    if color_cycle is None:
        color_cycle = sns.color_palette("tab10", len(unique_clusters))

    # Calcolo silhouette (se almeno 2 cluster)
    if len(unique_clusters) < 2:
        ax.text(0.5, 0.5, "Silhouette non definito (meno di 2 cluster)",
                ha="center", va="center")
        ax.set_xticks([]); ax.set_yticks([])
        return np.nan

    # Calcolo silhouette per tutti (inclusi noise) ma poi filtriamo
    s_values_all = silhouette_samples(X, labels, metric=metric)

    y_lower = 10
    cluster_avgs = []
    for idx, c in enumerate(sorted(unique_clusters)):
        mask = labels == c
        s_c = s_values_all[mask]
        # ordina valori per una visualizzazione a "barrette"
        s_c_sorted = np.sort(s_c)
        size_c = s_c_sorted.shape[0]
        if size_c <= 1:
            continue
        y_upper = y_lower + size_c
        color = color_cycle[idx % len(color_cycle)]
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, s_c_sorted,
                         facecolor=color, edgecolor=color, alpha=0.75)
        # etichetta cluster
        ax.text(-0.02, y_lower + 0.5 * size_c, str(c), fontsize=8, ha='right', va='center')
        cluster_avgs.append(np.mean(s_c_sorted))
        y_lower = y_upper + 10  # spazio tra cluster

    avg_s = np.mean(cluster_avgs) if cluster_avgs else np.nan
    if show_avg and not np.isnan(avg_s):
        ax.axvline(avg_s, color="red", linestyle="--", linewidth=1.2, label=f"Media = {avg_s:.3f}")
        ax.legend(loc="lower right", fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Punti ordinati per cluster")
    ax.set_xlim([-0.3, 1.0])
    ax.set_yticks([])
    return avg_s


# ---------------------------
# Figura 1: Silhouette "buono"
# ---------------------------

def fig_silhouette_good():
    X, y = make_blobs(n_samples=1200, centers=4, cluster_std=0.55, random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_s = silhouette_plot(X, labels, metric="euclidean", title="Silhouette plot - clustering ben separato", ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "silhouette_good.png"), dpi=170)
    plt.close(fig)
    return avg_s


# -------------------------------------------
# Figura 2 e 3: Confronto k e curva media vs k
# -------------------------------------------

def fig_silhouette_k_comparison_and_curve():
    X, _ = make_blobs(n_samples=1300, centers=4, cluster_std=[0.5, 0.6, 0.55, 0.5],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)
    ks = [2, 4, 6]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    avg_scores = {}
    for ax, k in zip(axes, ks):
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        avg_s = silhouette_plot(X, labels, title=f"k = {k}", ax=ax)
        avg_scores[k] = avg_s
    fig.suptitle("Confronto silhouette plot per k differenti", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "silhouette_k_comparison.png"), dpi=170)
    plt.close(fig)

    # Curva silhouette media vs k
    k_range = range(2, 11)
    mean_s = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        try:
            val = silhouette_score(X, labels)
        except Exception:
            val = np.nan
        mean_s.append(val)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(list(k_range), mean_s, marker="o")
    ax2.set_xlabel("k")
    ax2.set_ylabel("Silhouette medio")
    ax2.set_title("Andamento Silhouette medio al variare di k")
    # Evidenzia max
    best_k = list(k_range)[int(np.nanargmax(mean_s))]
    best_val = np.nanmax(mean_s)
    ax2.axvline(best_k, color="green", linestyle="--", alpha=0.7,
                label=f"Massimo a k={best_k} (s={best_val:.3f})")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "silhouette_avg_vs_k.png"), dpi=170)
    plt.close(fig2)


# --------------------------------------------------------
# Figura 4: Confronto metrica euclidea vs coseno normalizzato
# --------------------------------------------------------

def fig_metric_comparison():
    X, _ = make_blobs(n_samples=1100, centers=4, cluster_std=[0.6, 0.45, 0.5, 0.55],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    # Normalizziamo in norma unitaria per il coseno
    X_norm = normalize(X, norm="l2")

    k = 4
    km_eucl = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    labels_eucl = km_eucl.labels_

    # Per usare la silhouette con coseno utilizziamo direttamente X_norm e metric="cosine"
    km_cos = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X_norm)
    labels_cos = km_cos.labels_

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s1 = silhouette_plot(X, labels_eucl, metric="euclidean", title="Metrica Euclidea", ax=axes[0])
    s2 = silhouette_plot(X_norm, labels_cos, metric="cosine", title="Metrica Coseno (dati normalizzati)", ax=axes[1])
    fig.suptitle("Confronto silhouette: Euclidea vs Coseno", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "silhouette_metric_comparison.png"), dpi=170)
    plt.close(fig)
    return s1, s2


# ---------------------------------
# Figura 5: Silhouette con DBSCAN
# ---------------------------------

def fig_dbscan_example():
    X, _ = make_blobs(n_samples=900,
                      centers=[(-3, -2), (-1, 2.5), (2.5, -0.5)],
                      cluster_std=[0.5, 0.7, 0.55],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.5, min_samples=8)
    labels = db.fit_predict(X)

    # Maschera per escludere noise (-1)
    mask = labels != -1
    unique = np.unique(labels[mask])
    fig, ax = plt.subplots(figsize=(6, 4))

    if unique.size >= 2:
        avg_s = silhouette_plot(X[mask], labels[mask], metric="euclidean",
                                title="Silhouette DBSCAN (noise escluso)", ax=ax)
    else:
        ax.text(0.5, 0.5, "Silhouette non definibile (meno di 2 cluster)",
                ha="center", va="center")
        avg_s = np.nan

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "silhouette_dbscan.png"), dpi=170)
    plt.close(fig)
    return avg_s, np.sum(~mask)


# --------------------------------------------------
# Figura 6: Effetto PCA sul Silhouette (opzionale)
# --------------------------------------------------

def fig_pca_effect():
    X, _ = make_blobs(n_samples=1400, centers=5,
                      cluster_std=[0.7, 0.55, 0.6, 0.65, 0.5],
                      random_state=RANDOM_STATE)
    # Creiamo feature ridondanti per simulare un dataset "più largo"
    X = StandardScaler().fit_transform(X)
    # Aggiungiamo rumore e combinazioni lineari per aumentare dimensioni
    noise = 0.15 * np.random.randn(X.shape[0], 10)
    X_wide = np.hstack([X, X[:, :2] @ np.array([[1.2, -0.4], [0.3, 0.8]]), noise])

    ks = [3, 5]
    n_components_list = [2, 3, 5, 8, 12, X_wide.shape[1]]
    results = {k: [] for k in ks}

    for k in ks:
        for nc in n_components_list:
            pca = PCA(n_components=nc, random_state=RANDOM_STATE)
            Xp = pca.fit_transform(X_wide)
            km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE)
            labels = km.fit_predict(Xp)
            try:
                s_val = silhouette_score(Xp, labels)
            except Exception:
                s_val = np.nan
            results[k].append(s_val)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for k, vals in results.items():
        ax.plot(n_components_list, vals, marker="o", label=f"k={k}")
    ax.set_xlabel("Componenti PCA")
    ax.set_ylabel("Silhouette medio")
    ax.set_title("Effetto della riduzione dimensionale (PCA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "silhouette_pca_effect.png"), dpi=170)
    plt.close(fig)


def main():
    print("Generazione figure silhouette...")
    avg_good = fig_silhouette_good()
    fig_silhouette_k_comparison_and_curve()
    s_eucl, s_cos = fig_metric_comparison()
    s_dbscan, noise_count = fig_dbscan_example()
    fig_pca_effect()

    captions = {
        "silhouette_good.png": "Figura S.1: Silhouette plot per un clustering ben separato (K-Means, k=4). Le barre lunghe e positive indicano cluster compatti e distanti.",
        "silhouette_k_comparison.png": "Figura S.2: Confronto dei silhouette plot per k differenti. k troppo piccolo (sottocluster) e k troppo grande (sovra-frammentazione) degradano la qualità rispetto al k vicino all'ottimo.",
        "silhouette_avg_vs_k.png": "Figura S.3: Silhouette medio al variare di k. Il massimo locale suggerisce un intervallo ragionevole per il numero di cluster.",
        "silhouette_metric_comparison.png": "Figura S.4: Confronto tra metrica euclidea e coseno (dati normalizzati). La scelta della metrica può modificare la coesione percepita dei cluster.",
        "silhouette_dbscan.png": "Figura S.5: Esempio di silhouette con DBSCAN (noise escluso). Il numero di punti noise influenza l'interpretazione della qualità dei cluster.",
        "silhouette_pca_effect.png": "Figura S.6: Effetto del numero di componenti PCA sul Silhouette medio. Ridurre dimensionalità può rimuovere rumore e aumentare la separazione fino a un plateau."
    }

    print("\nDidascalie suggerite:")
    for k, v in captions.items():
        print(f"- {k}: {v}")

    print("\nRiepilogo valori medi (indicativi):")
    print(f"Silhouette clustering buono (k=4): {avg_good:.3f}")
    print(f"Silhouette confronto metriche: Euclidea={s_eucl:.3f} | Coseno={s_cos:.3f}")
    print(f"Silhouette DBSCAN (noise escluso): {s_dbscan:.3f} | Punti noise: {noise_count}")
    print("\nLe figure sono state salvate nella cartella:", OUT_DIR)

if __name__ == '__main__':
    # Disattiva alcuni warning non critici (ad es. edgecolor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()