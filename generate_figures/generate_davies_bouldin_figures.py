"""
Script per generare figure illustrative del Davies-Bouldin Index (DBI).

Figure prodotte (cartella: figs_dbi/):

1. dbi_good_vs_bad.png
   Confronto visivo tra un clustering "buono" (cluster compatti e separati) e uno "cattivo" (cluster sovrapposti / allungati). Riporta il valore del DBI in ciascun caso.

2. dbi_vs_k.png
   Curva del Davies-Bouldin index al variare di k (K-Means). Illustra come k influisce su compattezza/separazione; valori più bassi indicano configurazioni migliori.

3. dbi_component_example.png
   Esempio di calcolo concettuale: per un cluster i si mostra la dispersione intra-cluster (S_i) come cerchio medio, le distanze tra centroidi M_ij e si annota il valore R_ij = (S_i + S_j) / M_ij verso il cluster più “simile”.

4. dbi_outliers_effect.png
   Effetto degli outlier sul DBI: lo stesso dataset con e senza pochi punti estremi. Il DBI peggiora a causa dell’aumento della dispersione intra-cluster.

5. dbi_heatmap_k_pca.png
   Heatmap del DBI per una griglia di (k, n_componenti PCA) per mostrare come la riduzione dimensionale influenzi la qualità secondo DBI.

6. dbi_manual_check.png
   Confronto tra il DBI calcolato manualmente e quello di scikit-learn su più valori di k (verifica didattica).

Dipendenze:
  python 3.9+
  numpy
  matplotlib
  seaborn
  scikit-learn

Installazione (se necessario):
  pip install numpy matplotlib seaborn scikit-learn

Esecuzione:
  python generate_davies_bouldin_figures.py

Note teoriche sintetiche:
  DBI = (1 / k) * Σ_i max_{j != i} R_ij
  con R_ij = (S_i + S_j) / M_ij
  S_i = (1 / |C_i|) * Σ_{x in C_i} ||x - μ_i||     (dispersione media del cluster i)
  M_ij = || μ_i - μ_j ||                            (distanza tra centroidi)
  Più basso è il DBI, migliore (cluster compatti e ben separati).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score
from matplotlib.patches import Circle

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "../figs_dbi"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------

def compute_dbi_manual(X, labels):
    """
    Calcolo manuale del Davies-Bouldin Index per confronto didattico.
    Assunzione: metrica euclidea.
    """
    unique = np.unique(labels)
    k = unique.shape[0]
    # Centroidi e dispersioni
    centroids = []
    dispersions = []
    for c in unique:
        Xi = X[labels == c]
        mu = Xi.mean(axis=0)
        centroids.append(mu)
        # dispersione media (L2)
        Si = np.mean(np.linalg.norm(Xi - mu, axis=1))
        dispersions.append(Si)
    centroids = np.vstack(centroids)
    dispersions = np.array(dispersions)

    # R_ij = (S_i + S_j)/M_ij
    Rij = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                Rij[i, j] = -np.inf
            else:
                Mij = np.linalg.norm(centroids[i] - centroids[j])
                Rij[i, j] = (dispersions[i] + dispersions[j]) / (Mij + 1e-12)
    # Per ogni i, prendi max_j
    Ri = np.max(Rij, axis=1)
    DBI = np.mean(Ri)
    return DBI, centroids, dispersions, Rij

def create_well_separated(n_samples=900):
    X, _ = make_blobs(n_samples=n_samples,
                      centers=[(-4, -3), (-1, 3), (2.5, -1), (5, 3)],
                      cluster_std=[0.55, 0.6, 0.5, 0.6],
                      random_state=RANDOM_STATE)
    return StandardScaler().fit_transform(X)

def create_overlapping(n_samples=900):
    X, _ = make_blobs(n_samples=n_samples,
                      centers=[(-2, 0), (0, 1), (2, 0.5), (3.5, 1.2)],
                      cluster_std=[1.5, 1.2, 1.4, 1.1],
                      random_state=RANDOM_STATE)
    return StandardScaler().fit_transform(X)

def scatter_clusters(X, labels, ax, title="", palette="tab10"):
    uniq = np.unique(labels)
    colors = sns.color_palette(palette, len(uniq))
    for i, c in enumerate(uniq):
        ax.scatter(X[labels == c, 0], X[labels == c, 1], s=14, c=[colors[i]],
                   alpha=0.85, edgecolor="none", label=f"C{c}")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=7, frameon=True)

# ---------------------------------------------------------------------
# Figura 1: Buono vs Cattivo clustering
# ---------------------------------------------------------------------
def fig_dbi_good_vs_bad():
    X_good = create_well_separated()
    X_bad = create_overlapping()

    k = 4
    km_good = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    km_bad = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels_good = km_good.fit_predict(X_good)
    labels_bad = km_bad.fit_predict(X_bad)

    dbi_good = davies_bouldin_score(X_good, labels_good)
    dbi_bad = davies_bouldin_score(X_bad, labels_bad)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter_clusters(X_good, labels_good, axes[0], title=f"Clustering ben separato (DBI={dbi_good:.2f})")
    scatter_clusters(X_bad, labels_bad, axes[1], title=f"Clustering sovrapposto (DBI={dbi_bad:.2f})")

    fig.suptitle("Davies-Bouldin: confronto visivo buono vs cattivo", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "dbi_good_vs_bad.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figura 2: DBI vs k
# ---------------------------------------------------------------------
def fig_dbi_vs_k():
    X = create_well_separated()
    ks = range(2, 11)
    dbi_vals = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        dbi_vals.append(davies_bouldin_score(X, labels))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(list(ks), dbi_vals, marker="o")
    best_k = list(ks)[int(np.argmin(dbi_vals))]
    best_val = np.min(dbi_vals)
    ax.axvline(best_k, color="green", linestyle="--", alpha=0.7,
               label=f"Min DBI a k={best_k} ({best_val:.2f})")
    ax.set_xlabel("Numero di cluster k")
    ax.set_ylabel("Davies-Bouldin Index (↓ meglio)")
    ax.set_title("Andamento del DBI al variare di k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_vs_k.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figura 3: Componenti di DBI (R_ij) e dispersioni
# ---------------------------------------------------------------------
def fig_dbi_component_example():
    X = create_well_separated()
    k = 4
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    dbi, centroids, dispersions, Rij = compute_dbi_manual(X, labels)

    # Scegliamo un cluster i e il j che massimizza R_ij
    i = 0
    j = np.argmax(Rij[i])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    scatter_clusters(X, labels, ax, title=f"Esempio componenti DBI (DBI globale={dbi:.2f})")

    # Cerchio per dispersione S_i
    Si = dispersions[i]
    circ_i = Circle(centroids[i], radius=Si, facecolor="none", edgecolor="red", linestyle="--", linewidth=1.2)
    ax.add_patch(circ_i)
    ax.text(centroids[i,0], centroids[i,1]-1.5*Si, f"S_i={Si:.2f}", color="red", ha="center", fontsize=8)

    # Cerchio per dispersione S_j
    Sj = dispersions[j]
    circ_j = Circle(centroids[j], radius=Sj, facecolor="none", edgecolor="purple", linestyle="--", linewidth=1.2)
    ax.add_patch(circ_j)
    ax.text(centroids[j,0], centroids[j,1]-1.5*Sj, f"S_j={Sj:.2f}", color="purple", ha="center", fontsize=8)

    # Linea tra centroidi
    ax.plot([centroids[i,0], centroids[j,0]], [centroids[i,1], centroids[j,1]],
            color="black", linewidth=1.0, linestyle=":")
    Mij = np.linalg.norm(centroids[i] - centroids[j])
    Rij_val = Rij[i, j]

    ax.text((centroids[i,0]+centroids[j,0])/2,
            (centroids[i,1]+centroids[j,1])/2,
            f"M_ij={Mij:.2f}\nR_ij={(dispersions[i]+dispersions[j])/Mij:.2f}",
            fontsize=8, color="black", ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_title("Dispersioni (S_i, S_j) e distanza centroidi (M_ij)\nContributo R_ij = (S_i+S_j)/M_ij", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_component_example.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figura 4: Effetto outlier
# ---------------------------------------------------------------------
def fig_dbi_outliers_effect():
    X = create_well_separated()

    k = 4
    km_clean = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels_clean = km_clean.fit_predict(X)
    dbi_clean = davies_bouldin_score(X, labels_clean)

    # Aggiungiamo outlier
    n_out = 25
    outliers = np.random.uniform(low=-10, high=10, size=(n_out, X.shape[1]))
    X_out = np.vstack([X, outliers])
    km_out = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels_out = km_out.fit_predict(X_out)
    dbi_out = davies_bouldin_score(X_out, labels_out)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter_clusters(X, labels_clean, axes[0], title=f"Senza outlier (DBI={dbi_clean:.2f})")
    scatter_clusters(X_out, labels_out, axes[1], title=f"Con outlier (DBI={dbi_out:.2f})")
    for ax in axes:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    fig.suptitle("Effetto degli outlier sul Davies-Bouldin Index", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "dbi_outliers_effect.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figura 5: Heatmap DBI (k vs PCA components)
# ---------------------------------------------------------------------
def fig_dbi_heatmap_k_pca():
    # Dataset più “largo” con rumore
    X, _ = make_blobs(n_samples=1200,
                      centers=[(-4, -3), (-1, 3), (2.5, -1), (5, 3)],
                      cluster_std=[0.55, 0.6, 0.5, 0.6],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)
    # Aggiungo feature ridondanti + rumore
    noise = 0.2 * np.random.randn(X.shape[0], 8)
    Xw = np.hstack([X, noise, X[:, :2] * 0.3])

    k_values = [2, 3, 4, 5, 6, 7]
    pca_components = [2, 3, 5, 8, 12, Xw.shape[1]]
    heat = np.zeros((len(pca_components), len(k_values)))

    for i, nc in enumerate(pca_components):
        pca = PCA(n_components=nc, random_state=RANDOM_STATE)
        Xp = pca.fit_transform(Xw)
        for j, k in enumerate(k_values):
            km = KMeans(n_clusters=k, n_init=15, random_state=RANDOM_STATE)
            labels = km.fit_predict(Xp)
            dbi = davies_bouldin_score(Xp, labels)
            heat[i, j] = dbi

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "DBI (↓ meglio)"},
                xticklabels=k_values, yticklabels=pca_components, ax=ax)
    ax.set_xlabel("k")
    ax.set_ylabel("Componenti PCA")
    ax.set_title("Heatmap Davies-Bouldin: effetto di k e riduzione dimensionale")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_heatmap_k_pca.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figura 6: Confronto manuale vs scikit-learn
# ---------------------------------------------------------------------
def fig_dbi_manual_check():
    X = create_well_separated()
    ks = [2, 3, 4, 5, 6, 7]
    manual_vals = []
    sklearn_vals = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        dbi_sklearn = davies_bouldin_score(X, labels)
        dbi_manual, *_ = compute_dbi_manual(X, labels)
        manual_vals.append(dbi_manual)
        sklearn_vals.append(dbi_sklearn)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, sklearn_vals, marker="o", label="scikit-learn")
    ax.plot(ks, manual_vals, marker="s", linestyle="--", label="manuale")
    ax.set_xlabel("k")
    ax.set_ylabel("DBI")
    ax.set_title("Verifica calcolo DBI manuale vs scikit-learn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_manual_check.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Generazione figure Davies-Bouldin...")
    fig_dbi_good_vs_bad()
    fig_dbi_vs_k()
    fig_dbi_component_example()
    fig_dbi_outliers_effect()
    fig_dbi_heatmap_k_pca()
    fig_dbi_manual_check()
    print(f"Figure salvate in: {OUT_DIR}")

    print("\nDidascalie suggerite (adatta numerazione in tesi):\n")
    captions = [
        ("dbi_good_vs_bad.png",
         "Figura DBI.1: Confronto tra clustering ben separato (DBI basso) e clustering sovrapposto (DBI alto). Cluster compatti e distanti riducono l'indice."),
        ("dbi_vs_k.png",
         "Figura DBI.2: Andamento del Davies-Bouldin al variare di k. Il minimo relativo individua una configurazione più equilibrata per compattezza e separazione."),
        ("dbi_component_example.png",
         "Figura DBI.3: Illustrazione delle componenti di DBI: dispersioni intra-cluster (S_i, S_j) e distanza fra centroidi (M_ij). Il rapporto R_ij = (S_i+S_j)/M_ij influenza il massimo per il cluster i."),
        ("dbi_outliers_effect.png",
         "Figura DBI.4: Effetto degli outlier: pochi punti estremi aumentano la dispersione intra-cluster e peggiorano il DBI."),
        ("dbi_heatmap_k_pca.png",
         "Figura DBI.5: Heatmap del DBI per combinazioni di k e numero di componenti PCA. Evidenzia regioni parametriche più favorevoli (DBI più basso)."),
        ("dbi_manual_check.png",
         "Figura DBI.6: Verifica del calcolo manuale del DBI rispetto all'implementazione scikit-learn: le curve coincidono entro differenze numeriche trascurabili.")
    ]
    for fname, cap in captions:
        print(f"- {fname}: {cap}")

if __name__ == "__main__":
    main()