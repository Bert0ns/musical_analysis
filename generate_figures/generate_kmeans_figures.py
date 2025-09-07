import os
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "../figs/figs_kmeans"
os.makedirs(OUT_DIR, exist_ok=True)

def scatter2d(X, labels=None, title="", ax=None, s=14, cmap="tab10"):
    if ax is None:
        ax = plt.gca()
    if labels is None:
        ax.scatter(X[:, 0], X[:, 1], s=s, c="gray", alpha=0.85, edgecolor="none")
    else:
        ax.scatter(X[:, 0], X[:, 1], s=s, c=labels, cmap=cmap, alpha=0.9, edgecolor="none")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
    return ax

def plot_centroids(ax, centers, color="k", marker="X", s=90, edgecolor="white", linewidth=1.0):
    ax.scatter(centers[:, 0], centers[:, 1], c=color, s=s, marker=marker, edgecolor=edgecolor, linewidth=linewidth, zorder=5)

# Figura K.1 – Effetto della standardizzazione
def fig_k1_scaling_effect():
    # Dati con scale diverse sulle due dimensioni
    X, y = make_blobs(n_samples=600, centers=[(-3, 0), (3, 0), (0, 5)], cluster_std=[1.0, 2.0, 0.7], random_state=RANDOM_STATE)
    # Allunghiamo l'asse x per accentuare lo squilibrio di scala
    X_unscaled = X.copy()
    X_unscaled[:, 0] *= 8.0

    # K-Means su dati non scalati
    km_raw = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    y_raw = km_raw.fit_predict(X_unscaled)

    # Standardizzazione e K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_unscaled)
    km_scaled = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE)
    y_scaled = km_scaled.fit_predict(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0 = scatter2d(X_unscaled, y_raw, "Senza standardizzazione", ax=axes[0])
    plot_centroids(ax0, km_raw.cluster_centers_)
    ax1 = scatter2d(X_scaled, y_scaled, "Con standardizzazione (z-score)", ax=axes[1])
    plot_centroids(ax1, km_scaled.cluster_centers_)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "K1_scaling_effect.png"), dpi=160)
    plt.close(fig)

# Figura K.2 – Elbow e silhouette vs k
def fig_k2_elbow_silhouette():
    X, _ = make_blobs(n_samples=1000, centers=4, cluster_std=1.1, random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    ks = range(2, 11)
    inertias = []
    silhouettes = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        # Silhouette definita per k >= 2
        silhouettes.append(silhouette_score(X, labels, metric="euclidean"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(list(ks), inertias, marker="o")
    axes[0].set_title("Metodo elbow (inertia)")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia ↓")

    axes[1].plot(list(ks), silhouettes, marker="o", color="tab:green")
    axes[1].set_title("Silhouette media")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("Silhouette ↑")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "K2_elbow_silhouette.png"), dpi=160)
    plt.close(fig)

# Figura K.3 – Sensibilità all’inizializzazione
def fig_k3_init_sensitivity():
    X, _ = make_blobs(n_samples=800, centers=[(-4, -2), (0, 3), (4, -1)], cluster_std=[1.0, 1.2, 0.9], random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    # Due soluzioni con semi diversi, n_init=1 per evidenziare l'effetto del seed
    km_a = KMeans(n_clusters=3, n_init=1, random_state=10)
    km_b = KMeans(n_clusters=3, n_init=1, random_state=999)
    y_a = km_a.fit_predict(X)
    y_b = km_b.fit_predict(X)

    # Distribuzione dell'inertia su molte esecuzioni (n_init=1)
    runs = 30
    inertias = []
    for seed in np.random.randint(0, 10_000, size=runs):
        km = KMeans(n_clusters=3, n_init=1, random_state=int(seed))
        km.fit(X)
        inertias.append(km.inertia_)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    scatter2d(X, y_a, "Soluzione A (seed=10)", ax=axes[0])
    plot_centroids(axes[0], km_a.cluster_centers_)
    scatter2d(X, y_b, "Soluzione B (seed=999)", ax=axes[1])
    plot_centroids(axes[1], km_b.cluster_centers_)
    axes[2].hist(inertias, bins=8, color="tab:blue", alpha=0.85)
    axes[2].set_title("Distribuzione inertia (30 esecuzioni)")
    axes[2].set_xlabel("Inertia"); axes[2].set_ylabel("Frequenza")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "K3_init_sensitivity.png"), dpi=160)
    plt.close(fig)

# Figura K.4 – Impatto degli outlier
def fig_k4_outliers_effect():
    X, _ = make_blobs(n_samples=600, centers=[(-2, 0), (2, 0)], cluster_std=[0.8, 0.8], random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    # Aggiungiamo pochi outlier molto lontani
    n_out = 8
    outliers = np.array([[8.0, 8.0], [9.0, 8.5], [8.5, 9.5], [-8.5, 9.0], [-9.0, -8.5], [-8.5, -9.0], [9.0, -8.0], [8.5, -9.5]])
    X_with_out = np.vstack([X, outliers])

    # Stesso K-Means
    km_clean = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit(X)
    km_out = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit(X_with_out)

    # Calcolo spostamento centroidi (matching per distanza minima)
    C0 = km_clean.cluster_centers_
    C1 = km_out.cluster_centers_
    # associa ogni centroide di C0 al più vicino in C1
    from scipy.spatial.distance import cdist
    D = cdist(C0, C1)
    match_idx = np.argmin(D, axis=1)
    shifts = np.sqrt(np.sum((C0 - C1[match_idx])**2, axis=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax0 = scatter2d(X, km_clean.labels_, "Senza outlier", ax=axes[0])
    plot_centroids(ax0, C0)
    ax1 = scatter2d(X_with_out, km_out.labels_, f"Con outlier (shift medio={np.mean(shifts):.2f})", ax=axes[1])
    plot_centroids(ax1, C1, color="tab:red")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "K4_outliers_effect.png"), dpi=160)
    plt.close(fig)

# Figura K.5 – Caso avverso: forme non sferiche (two moons e cluster anisotropi)
def fig_k5_adverse_cases():
    # Two moons
    X_m, _ = make_moons(n_samples=600, noise=0.06, random_state=RANDOM_STATE)
    km_m = KMeans(n_clusters=2, n_init=20, random_state=RANDOM_STATE).fit(X_m)
    y_m = km_m.labels_

    # Cluster anisotropi: trasformazione lineare di blob
    X_b, _ = make_blobs(n_samples=600, centers=3, cluster_std=[1.0, 0.6, 0.8], random_state=RANDOM_STATE)
    T = np.array([[0.6, -0.6], [-0.2, 0.8]])  # shear + scaling
    X_a = X_b @ T.T
    X_a = StandardScaler().fit_transform(X_a)
    km_a = KMeans(n_clusters=3, n_init=20, random_state=RANDOM_STATE).fit(X_a)
    y_a = km_a.labels_

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter2d(X_m, y_m, "Due lune (K-Means fatica)", ax=axes[0])
    plot_centroids(axes[0], km_m.cluster_centers_)
    scatter2d(X_a, y_a, "Cluster anisotropi (frontiere non sferiche)", ax=axes[1])
    plot_centroids(axes[1], km_a.cluster_centers_)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "K5_adverse_cases.png"), dpi=160)
    plt.close(fig)

if __name__ == "__main__":
    fig_k1_scaling_effect()
    fig_k2_elbow_silhouette()
    fig_k3_init_sensitivity()
    fig_k4_outliers_effect()
    fig_k5_adverse_cases()
    print(f"Figure salvate in: {OUT_DIR}")