"""
Script to generate illustrative figures for the Davies-Bouldin Index (DBI).

Figures produced (folder: figs_dbi/):

1. dbi_good_vs_bad.png
    Visual comparison between a "good" clustering (compact, separated clusters) and a "bad" one (overlapping / elongated clusters). Shows DBI in each case.

2. dbi_vs_k.png
    Davies-Bouldin curve over k (K-Means). Shows how k affects compactness/separation; lower values indicate better configurations.

3. dbi_component_example.png
    Conceptual computation example: for a cluster i, shows intra-cluster dispersion (S_i) as a mean circle, centroid distances M_ij, and annotates R_ij = (S_i + S_j) / M_ij toward the most similar cluster.

4. dbi_outliers_effect.png
    Outlier effect on DBI: same dataset with and without a few extreme points. DBI worsens due to increased intra-cluster dispersion.

5. dbi_heatmap_k_pca.png
    DBI heatmap over a grid of (k, n_PCA_components) to show how dimensionality reduction affects DBI quality.

6. dbi_manual_check.png
    Comparison between manually computed DBI and scikit-learn DBI across multiple k values (didactic check).

Dependencies:
  python 3.9+
  numpy
  matplotlib
  seaborn
  scikit-learn

Install (if needed):
  pip install numpy matplotlib seaborn scikit-learn

Run:
  python generate_davies_bouldin_figures.py

Concise theory notes:
  DBI = (1 / k) * Σ_i max_{j != i} R_ij
  where R_ij = (S_i + S_j) / M_ij
  S_i = (1 / |C_i|) * Σ_{x in C_i} ||x - μ_i||     (mean dispersion of cluster i)
  M_ij = || μ_i - μ_j ||                            (distance between centroids)
  Lower DBI is better (compact, well-separated clusters).
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

OUT_DIR = "../figs/figs_dbi"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def compute_dbi_manual(X, labels):
    """
    Manual Davies-Bouldin Index computation for didactic comparison.
    Assumption: Euclidean metric.
    """
    unique = np.unique(labels)
    k = unique.shape[0]
    # Centroids and dispersions
    centroids = []
    dispersions = []
    for c in unique:
        Xi = X[labels == c]
        mu = Xi.mean(axis=0)
        centroids.append(mu)
        # mean dispersion (L2)
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
    # For each i, take max_j
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
# Figure 1: Good vs bad clustering
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
    scatter_clusters(X_good, labels_good, axes[0], title=f"Well-separated clustering (DBI={dbi_good:.2f})")
    scatter_clusters(X_bad, labels_bad, axes[1], title=f"Overlapping clustering (DBI={dbi_bad:.2f})")

    fig.suptitle("Davies-Bouldin: visual comparison good vs bad", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "dbi_good_vs_bad.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figure 2: DBI vs k
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
               label=f"Min DBI at k={best_k} ({best_val:.2f})")
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("Davies-Bouldin Index (lower is better)")
    ax.set_title("DBI over k")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_vs_k.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figure 3: DBI components (R_ij) and dispersions
# ---------------------------------------------------------------------
def fig_dbi_component_example():
    X = create_well_separated()
    k = 4
    km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    dbi, centroids, dispersions, Rij = compute_dbi_manual(X, labels)

    # Pick a cluster i and the j that maximizes R_ij
    i = 0
    j = np.argmax(Rij[i])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    scatter_clusters(X, labels, ax, title=f"DBI components example (global DBI={dbi:.2f})")

    # Circle for dispersion S_i
    Si = dispersions[i]
    circ_i = Circle(centroids[i], radius=Si, facecolor="none", edgecolor="red", linestyle="--", linewidth=1.2)
    ax.add_patch(circ_i)
    ax.text(centroids[i,0], centroids[i,1]-1.5*Si, f"S_i={Si:.2f}", color="red", ha="center", fontsize=8)

    # Circle for dispersion S_j
    Sj = dispersions[j]
    circ_j = Circle(centroids[j], radius=Sj, facecolor="none", edgecolor="purple", linestyle="--", linewidth=1.2)
    ax.add_patch(circ_j)
    ax.text(centroids[j,0], centroids[j,1]-1.5*Sj, f"S_j={Sj:.2f}", color="purple", ha="center", fontsize=8)

    # Line between centroids
    ax.plot([centroids[i,0], centroids[j,0]], [centroids[i,1], centroids[j,1]],
            color="black", linewidth=1.0, linestyle=":")
    Mij = np.linalg.norm(centroids[i] - centroids[j])
    Rij_val = Rij[i, j]

    ax.text((centroids[i,0]+centroids[j,0])/2,
            (centroids[i,1]+centroids[j,1])/2,
            f"M_ij={Mij:.2f}\nR_ij={(dispersions[i]+dispersions[j])/Mij:.2f}",
            fontsize=8, color="black", ha="center", va="center",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    ax.set_title("Dispersions (S_i, S_j) and centroid distance (M_ij)\nContribution R_ij = (S_i+S_j)/M_ij", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_component_example.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figure 4: Outlier effect
# ---------------------------------------------------------------------
def fig_dbi_outliers_effect():
    X = create_well_separated()

    k = 4
    km_clean = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels_clean = km_clean.fit_predict(X)
    dbi_clean = davies_bouldin_score(X, labels_clean)

    # Add outliers
    n_out = 25
    outliers = np.random.uniform(low=-10, high=10, size=(n_out, X.shape[1]))
    X_out = np.vstack([X, outliers])
    km_out = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
    labels_out = km_out.fit_predict(X_out)
    dbi_out = davies_bouldin_score(X_out, labels_out)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    scatter_clusters(X, labels_clean, axes[0], title=f"Without outliers (DBI={dbi_clean:.2f})")
    scatter_clusters(X_out, labels_out, axes[1], title=f"With outliers (DBI={dbi_out:.2f})")
    for ax in axes:
        ax.set_xlim(-4, 4); ax.set_ylim(-4, 4)
    fig.suptitle("Outlier effect on Davies-Bouldin Index", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "dbi_outliers_effect.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figure 5: DBI heatmap (k vs PCA components)
# ---------------------------------------------------------------------
def fig_dbi_heatmap_k_pca():
    # Wider dataset with noise
    X, _ = make_blobs(n_samples=1200,
                      centers=[(-4, -3), (-1, 3), (2.5, -1), (5, 3)],
                      cluster_std=[0.55, 0.6, 0.5, 0.6],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)
    # Add redundant features + noise
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
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="viridis", cbar_kws={"label": "DBI (lower is better)"},
                xticklabels=k_values, yticklabels=pca_components, ax=ax)
    ax.set_xlabel("k")
    ax.set_ylabel("PCA components")
    ax.set_title("Davies-Bouldin heatmap: effect of k and dimensionality reduction")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_heatmap_k_pca.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# Figure 6: Manual vs scikit-learn comparison
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
    ax.plot(ks, manual_vals, marker="s", linestyle="--", label="manual")
    ax.set_xlabel("k")
    ax.set_ylabel("DBI")
    ax.set_title("Manual DBI check vs scikit-learn")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbi_manual_check.png"), dpi=170)
    plt.close(fig)

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    print("Generating Davies-Bouldin figures...")
    fig_dbi_good_vs_bad()
    fig_dbi_vs_k()
    fig_dbi_component_example()
    fig_dbi_outliers_effect()
    fig_dbi_heatmap_k_pca()
    fig_dbi_manual_check()
        print(f"Figures saved to: {OUT_DIR}")

        print("\nSuggested captions (adjust numbering in thesis):\n")
    captions = [
        ("dbi_good_vs_bad.png",
            "Figure DBI.1: Comparison between well-separated clustering (low DBI) and overlapping clustering (high DBI). Compact, distant clusters reduce the index."),
        ("dbi_vs_k.png",
            "Figure DBI.2: Davies-Bouldin over k. A relative minimum indicates a more balanced configuration for compactness and separation."),
        ("dbi_component_example.png",
            "Figure DBI.3: Illustration of DBI components: intra-cluster dispersions (S_i, S_j) and centroid distance (M_ij). The ratio R_ij = (S_i+S_j)/M_ij influences the max for cluster i."),
        ("dbi_outliers_effect.png",
            "Figure DBI.4: Outlier effect: a few extreme points increase intra-cluster dispersion and worsen DBI."),
        ("dbi_heatmap_k_pca.png",
            "Figure DBI.5: DBI heatmap for combinations of k and number of PCA components. Highlights more favorable parameter regions (lower DBI)."),
        ("dbi_manual_check.png",
            "Figure DBI.6: Manual DBI calculation vs scikit-learn implementation: curves match within negligible numerical differences.")
    ]
    for fname, cap in captions:
        print(f"- {fname}: {cap}")

if __name__ == "__main__":
    main()