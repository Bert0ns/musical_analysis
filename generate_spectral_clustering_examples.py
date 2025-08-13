import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize

from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "figs"
os.makedirs(OUT_DIR, exist_ok=True)

# Utility plotting
def scatter2d(X, labels=None, title="", ax=None, s=14, cmap="tab10"):
    if ax is None:
        ax = plt.gca()
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=s, c="gray", alpha=0.8, edgecolor="none")
    else:
        scatter = ax.scatter(X[:, 0], X[:, 1], s=s, c=labels, cmap=cmap, alpha=0.9, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    return ax

def symmetrize(W):
    return (W + W.T) / 2

def build_knn_affinity(X, n_neighbors=15, mode="connectivity", include_self=False):
    # mode="connectivity" -> 0/1 adjacency; mode="distance" -> distances
    G = kneighbors_graph(X, n_neighbors=n_neighbors, mode=mode, include_self=include_self, n_jobs=-1)
    # Make symmetric
    G = G.maximum(G.T)
    return G

def rbf_affinity_from_dist(D2, gamma):
    # D2 = squared distances
    A = np.exp(-gamma * D2)
    np.fill_diagonal(A, 0.0)
    return A

def self_tuning_affinity(X, k_local=7):
    # Zelnik-Manor & Perona (2004): sigma_i = distanza al k_local-esimo vicino
    D = pairwise_distances(X, metric="euclidean")
    # ordina le distanze per riga
    sortD = np.sort(D, axis=1)
    # indice 1.. => 0 è distanza a se stessi
    k_idx = min(k_local, sortD.shape[1]-1)
    sigma = sortD[:, k_idx] + 1e-12
    # matrice sigma_i * sigma_j
    Sigma = np.outer(sigma, sigma)
    A = np.exp(-(D ** 2) / Sigma)
    np.fill_diagonal(A, 0.0)
    # opzionale: sparsifica tenendo solo mutui k-NN per stabilità
    knn = kneighbors_graph(X, n_neighbors=k_local, mode="connectivity", include_self=False, n_jobs=-1)
    mask = knn.maximum(knn.T).toarray().astype(bool)
    A = A * mask
    return A

# Figura 2.1 – Two moons: Originale vs K-Means vs Spectral
def fig_2_1():
    X, y_true = make_moons(n_samples=600, noise=0.06, random_state=RANDOM_STATE)

    km = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE)
    y_km = km.fit_predict(X)

    sp = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                            n_neighbors=15, assign_labels="kmeans",
                            random_state=RANDOM_STATE)
    y_sp = sp.fit_predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    scatter2d(X, None, "Spazio originale", ax=axes[0])
    scatter2d(X, y_km, "K-Means (k=2)", ax=axes[1], cmap="tab10")
    scatter2d(X, y_sp, "Spectral (k-NN=15)", ax=axes[2], cmap="tab20")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_01_moons_kmeans_vs_spectral.png"), dpi=160)
    plt.close(fig)

# Figura 2.2 – Effetto di k nel grafo k-NN (matrice di adiacenza)
def fig_2_2():
    X, _ = make_moons(n_samples=400, noise=0.06, random_state=RANDOM_STATE)
    ks = [5, 30]
    figs, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, k in zip(axes, ks):
        G = build_knn_affinity(X, n_neighbors=k, mode="connectivity", include_self=False)
        # Visualizza la matrice di adiacenza
        ax.imshow(G.toarray(), cmap="Greys", interpolation="nearest")
        ax.set_title(f"Adiacenza k-NN (k={k})")
        ax.set_xticks([]); ax.set_yticks([])
    figs.tight_layout()
    figs.savefig(os.path.join(OUT_DIR, "02_02_knn_adjacency_k5_k30.png"), dpi=160)
    plt.close(figs)

# Figura 2.3 – Effetto del parametro RBF (gamma)
def fig_2_3():
    X, _ = make_moons(n_samples=600, noise=0.06, random_state=RANDOM_STATE)
    # gamma = 1/(2*sigma^2)
    gammas = [0.5, 50.0]  # sigma grande vs sigma piccolo
    titles = [f"RBF gamma={gammas[0]} (sigma grande)", f"RBF gamma={gammas[1]} (sigma piccolo)"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, g, title in zip(axes, gammas, titles):
        sp = SpectralClustering(n_clusters=2, affinity="rbf", gamma=g,
                                assign_labels="kmeans", random_state=RANDOM_STATE)
        y_sp = sp.fit_predict(X)
        scatter2d(X, y_sp, title, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_03_rbf_gamma_effect.png"), dpi=160)
    plt.close(fig)

# Figura 2.4 – Eigengap su Laplaciano normalizzato
def fig_2_4():
    X, _ = make_moons(n_samples=500, noise=0.06, random_state=RANDOM_STATE)
    G = build_knn_affinity(X, n_neighbors=15, mode="connectivity", include_self=False)
    # Costruisci affinità pesata semplice (0/1 connessi) oppure usa distanze -> RBF locale minima
    # Qui usiamo una pesatura semplice uniformando a 1 per archi (connettività).
    W = G  # già simmetrica
    # Laplaciano normalizzato
    L = csgraph.laplacian(W, normed=True)
    # Calcola i più piccoli autovalori (escludendo la componente nulla numericamente instabile)
    k_eval = 10
    # eigsh su matrice sparsa simmetrica
    evals, _ = eigsh(L, k=k_eval, which="SM", tol=1e-4, maxiter=5000)
    evals = np.sort(np.real(evals))

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(1, k_eval + 1), evals, marker="o")
    # Evidenzia il gap massimo
    gaps = np.diff(evals)
    if len(gaps) > 0:
        idx = np.argmax(gaps) + 1  # gap tra idx e idx+1 => suggerisce k=idx
        ax.axvline(idx, color="tomato", linestyle="--", alpha=0.7, label=f"eigengap → k≈{idx}")
        ax.legend(loc="best")
    ax.set_xlabel("Indice autovalore")
    ax.set_ylabel("Autovalore (L_norm)")
    ax.set_title("Eigengap per la scelta di k")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_04_eigengap.png"), dpi=160)
    plt.close(fig)

# Figura 2.5 – Embedding spettrale: normalizzazione di riga
def fig_2_5():
    X, _ = make_moons(n_samples=600, noise=0.06, random_state=RANDOM_STATE)
    # Costruisci affinità k-NN (connettività) e Laplaciano normalizzato
    W = build_knn_affinity(X, n_neighbors=15, mode="connectivity", include_self=False)
    L = csgraph.laplacian(W, normed=True)

    # Prendi i 2 autovettori associati ai più piccoli autovalori (evita quello costante se presente)
    k = 2
    evals, evecs = eigsh(L, k=k+1, which="SM", tol=1e-4, maxiter=5000)
    # Scarta il primo autovettore (associato tipicamente a autovalore ~0)
    order = np.argsort(evals)
    evecs = evecs[:, order]
    U = evecs[:, 1:k+1]  # prendi le 2 componenti non banali

    U_row = normalize(U, norm="l2", axis=1)

    km_raw = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit(U)
    y_raw = km_raw.labels_
    km_norm = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit(U_row)
    y_norm = km_norm.labels_

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    axes[0].scatter(U[:, 0], U[:, 1], c=y_raw, cmap="tab10", s=14, edgecolor="none")
    axes[0].set_title("Embedding U (non normalizzato)")
    axes[0].set_xticks([]); axes[0].set_yticks([])
    axes[1].scatter(U_row[:, 0], U_row[:, 1], c=y_norm, cmap="tab10", s=14, edgecolor="none")
    axes[1].set_title("Embedding U normalizzato (righe a norma 1)")
    axes[1].set_xticks([]); axes[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_05_embedding_row_normalization.png"), dpi=160)
    plt.close(fig)

# Figura 2.6 (opzionale) – Densità variabile e self-tuning
def fig_2_6_optional():
    X, _ = make_blobs(n_samples=[250, 250, 250],
                      centers=[[0, 0], [4, 0], [2, 2.5]],
                      cluster_std=[0.30, 0.90, 0.50],
                      random_state=RANDOM_STATE)

    # Spettro con RBF globale
    D2 = pairwise_distances(X, squared=True)
    A_global = rbf_affinity_from_dist(D2, gamma=2.0)  # gamma scelto arbitrariamente
    sp_global = SpectralClustering(n_clusters=3, affinity="precomputed",
                                   assign_labels="kmeans", random_state=RANDOM_STATE)
    y_global = sp_global.fit_predict(A_global)

    # Spettro con scala locale (self-tuning)
    A_local = self_tuning_affinity(X, k_local=7)
    sp_local = SpectralClustering(n_clusters=3, affinity="precomputed",
                                  assign_labels="kmeans", random_state=RANDOM_STATE)
    y_local = sp_local.fit_predict(A_local)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    scatter2d(X, y_global, "Affinità RBF (scala globale)", ax=axes[0])
    scatter2d(X, y_local, "Affinità self-tuning (scala locale)", ax=axes[1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "02_06_self_tuning_optional.png"), dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    fig_2_1()
    fig_2_2()
    fig_2_3()
    fig_2_4()
    fig_2_5()
    # facoltativa (può richiedere più memoria/tempo su alcune macchine)
    fig_2_6_optional()
    print(f"Figure salvate in: {OUT_DIR}")