"""
Script to generate illustrative images for the Silhouette Score in clustering.

Figures produced (folder: figs_silhouette/):

1. silhouette_good.png
    Silhouette plot for a "good" clustering (K-Means on 4 well-separated blobs).
2. silhouette_k_comparison.png
    Silhouette plot comparison for k=2, k=4 (near optimal), and k=6 (over-fragmentation).
3. silhouette_avg_vs_k.png
    Mean silhouette over k (support method for choosing the number of clusters).
4. silhouette_metric_comparison.png
    Silhouette comparison using Euclidean vs cosine distance (normalized data) to show metric impact.
5. silhouette_dbscan.png
    Example silhouette with DBSCAN (noise excluded from calculation and shown separately).
6. silhouette_pca_effect.png (optional)
    Effect of dimensionality reduction (PCA) on mean silhouette for K-Means at fixed k.

Dependencies:
- Python 3.9+
- numpy
- matplotlib
- scikit-learn
- seaborn (only for an optional palette; can be omitted)

Quick install (if packages are missing):
pip install numpy matplotlib scikit-learn seaborn

Run:
python generate_silhouette_figures.py

Notes:
- The silhouette score s(i) for point i is defined as:
        s(i) = (b(i) - a(i)) / max(a(i), b(i))
  where a(i) is the average distance of i to other points in its cluster,
  and b(i) is the lowest average distance to an alternative cluster (the "nearest cluster").
- Values near 1 indicate good assignment; around 0 indicate boundary points;
  negative values indicate possible misassignment.

Suggested captions are printed at the end (see print()).
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

OUT_DIR = "../figs/figs_silhouette"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------
# Utilities for silhouette plots
# ---------------------------


def silhouette_plot(
    X, labels, metric="euclidean", title="", ax=None, show_avg=True, color_cycle=None
):
    """
    Create a silhouette plot for clusters in 'labels'.
    Ignore (do not draw) clusters with a single point.
    Returns the mean silhouette (excluding size-1 clusters).
    """
    if ax is None:
        ax = plt.gca()

    unique_clusters = [
        c for c in np.unique(labels) if c != -1
    ]  # exclude noise if present
    if color_cycle is None:
        color_cycle = sns.color_palette("tab10", len(unique_clusters))

    # Compute silhouette (if at least 2 clusters)
    if len(unique_clusters) < 2:
        ax.text(
            0.5,
            0.5,
            "Silhouette undefined (fewer than 2 clusters)",
            ha="center",
            va="center",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        return np.nan

    # Compute silhouette for all (including noise), then filter
    s_values_all = silhouette_samples(X, labels, metric=metric)

    y_lower = 10
    cluster_avgs = []
    for idx, c in enumerate(sorted(unique_clusters)):
        mask = labels == c
        s_c = s_values_all[mask]
        # Sort values for bar-like visualization
        s_c_sorted = np.sort(s_c)
        size_c = s_c_sorted.shape[0]
        if size_c <= 1:
            continue
        y_upper = y_lower + size_c
        color = color_cycle[idx % len(color_cycle)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            s_c_sorted,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
        )
        # Cluster label
        ax.text(
            -0.02, y_lower + 0.5 * size_c, str(c), fontsize=8, ha="right", va="center"
        )
        cluster_avgs.append(np.mean(s_c_sorted))
        y_lower = y_upper + 10  # spacing between clusters

    avg_s = np.mean(cluster_avgs) if cluster_avgs else np.nan
    if show_avg and not np.isnan(avg_s):
        ax.axvline(
            avg_s,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=f"Mean = {avg_s:.3f}",
        )
        ax.legend(loc="lower right", fontsize=8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Silhouette value")
    ax.set_ylabel("Points ordered by cluster")
    ax.set_xlim([-0.3, 1.0])
    ax.set_yticks([])
    return avg_s


# ---------------------------
# Figure 1: "good" silhouette
# ---------------------------


def fig_silhouette_good():
    X, y = make_blobs(
        n_samples=1200, centers=4, cluster_std=0.55, random_state=RANDOM_STATE
    )
    X = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    fig, ax = plt.subplots(figsize=(6, 4))
    avg_s = silhouette_plot(
        X,
        labels,
        metric="euclidean",
        title="Silhouette plot - well-separated clustering",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "silhouette_good.png"), dpi=170)
    plt.close(fig)
    return avg_s


# -------------------------------------------
# Figures 2 and 3: k comparison and mean curve vs k
# -------------------------------------------


def fig_silhouette_k_comparison_and_curve():
    X, _ = make_blobs(
        n_samples=1300,
        centers=4,
        cluster_std=[0.5, 0.6, 0.55, 0.5],
        random_state=RANDOM_STATE,
    )
    X = StandardScaler().fit_transform(X)
    ks = [2, 4, 6]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    avg_scores = {}
    for ax, k in zip(axes, ks):
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        avg_s = silhouette_plot(X, labels, title=f"k = {k}", ax=ax)
        avg_scores[k] = avg_s
    fig.suptitle("Silhouette plot comparison for different k", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "silhouette_k_comparison.png"), dpi=170)
    plt.close(fig)

    # Mean silhouette curve vs k
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
    ax2.set_ylabel("Mean silhouette")
    ax2.set_title("Mean silhouette vs k")
    # Highlight max
    best_k = list(k_range)[int(np.nanargmax(mean_s))]
    best_val = np.nanmax(mean_s)
    ax2.axvline(
        best_k,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Max at k={best_k} (s={best_val:.3f})",
    )
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(OUT_DIR, "silhouette_avg_vs_k.png"), dpi=170)
    plt.close(fig2)


# --------------------------------------------------------
# Figure 4: Euclidean vs cosine metric comparison
# --------------------------------------------------------


def fig_metric_comparison():
    X, _ = make_blobs(
        n_samples=1100,
        centers=4,
        cluster_std=[0.6, 0.45, 0.5, 0.55],
        random_state=RANDOM_STATE,
    )
    X = StandardScaler().fit_transform(X)

    # Normalize to unit norm for cosine
    X_norm = normalize(X, norm="l2")

    k = 4
    km_eucl = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X)
    labels_eucl = km_eucl.labels_

    # For cosine silhouette, use X_norm with metric="cosine"
    km_cos = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE).fit(X_norm)
    labels_cos = km_cos.labels_

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    s1 = silhouette_plot(
        X, labels_eucl, metric="euclidean", title="Euclidean metric", ax=axes[0]
    )
    s2 = silhouette_plot(
        X_norm,
        labels_cos,
        metric="cosine",
        title="Cosine metric (normalized data)",
        ax=axes[1],
    )
    fig.suptitle("Silhouette comparison: Euclidean vs Cosine", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "silhouette_metric_comparison.png"), dpi=170)
    plt.close(fig)
    return s1, s2


# ---------------------------------
# Figure 5: Silhouette with DBSCAN
# ---------------------------------


def fig_dbscan_example():
    X, _ = make_blobs(
        n_samples=900,
        centers=[(-3, -2), (-1, 2.5), (2.5, -0.5)],
        cluster_std=[0.5, 0.7, 0.55],
        random_state=RANDOM_STATE,
    )
    X = StandardScaler().fit_transform(X)

    db = DBSCAN(eps=0.5, min_samples=8)
    labels = db.fit_predict(X)

    # Mask to exclude noise (-1)
    mask = labels != -1
    unique = np.unique(labels[mask])
    fig, ax = plt.subplots(figsize=(6, 4))

    if unique.size >= 2:
        avg_s = silhouette_plot(
            X[mask],
            labels[mask],
            metric="euclidean",
            title="DBSCAN silhouette (noise excluded)",
            ax=ax,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "Silhouette undefined (fewer than 2 clusters)",
            ha="center",
            va="center",
        )
        avg_s = np.nan

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "silhouette_dbscan.png"), dpi=170)
    plt.close(fig)
    return avg_s, np.sum(~mask)


# --------------------------------------------------
# Figure 6: PCA effect on silhouette (optional)
# --------------------------------------------------


def fig_pca_effect():
    X, _ = make_blobs(
        n_samples=1400,
        centers=5,
        cluster_std=[0.7, 0.55, 0.6, 0.65, 0.5],
        random_state=RANDOM_STATE,
    )
    # Create redundant features to simulate a "wider" dataset
    X = StandardScaler().fit_transform(X)
    # Add noise and linear combinations to increase dimensionality
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
    ax.set_xlabel("PCA components")
    ax.set_ylabel("Mean silhouette")
    ax.set_title("Effect of dimensionality reduction (PCA)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "silhouette_pca_effect.png"), dpi=170)
    plt.close(fig)


def main():
    print("Generating silhouette figures...")
    avg_good = fig_silhouette_good()
    fig_silhouette_k_comparison_and_curve()
    s_eucl, s_cos = fig_metric_comparison()
    s_dbscan, noise_count = fig_dbscan_example()
    fig_pca_effect()

    captions = {
        "silhouette_good.png": "Figure S.1: Silhouette plot for a well-separated clustering (K-Means, k=4). Long positive bars indicate compact, well-separated clusters.",
        "silhouette_k_comparison.png": "Figure S.2: Silhouette plot comparison for different k. k too small (subclusters) and k too large (over-fragmentation) reduce quality versus k near optimal.",
        "silhouette_avg_vs_k.png": "Figure S.3: Mean silhouette vs k. A local maximum suggests a reasonable range for the number of clusters.",
        "silhouette_metric_comparison.png": "Figure S.4: Euclidean vs cosine metric comparison (normalized data). Metric choice can alter perceived cluster cohesion.",
        "silhouette_dbscan.png": "Figure S.5: DBSCAN silhouette example (noise excluded). The number of noise points affects cluster quality interpretation.",
        "silhouette_pca_effect.png": "Figure S.6: Effect of PCA components on mean silhouette. Reducing dimensionality can remove noise and increase separation up to a plateau.",
    }

    print("\nSuggested captions:")
    for k, v in captions.items():
        print(f"- {k}: {v}")

    print("\nSummary of mean values (indicative):")
    print(f"Silhouette good clustering (k=4): {avg_good:.3f}")
    print(f"Silhouette metric comparison: Euclidean={s_eucl:.3f} | Cosine={s_cos:.3f}")
    print(
        f"DBSCAN silhouette (noise excluded): {s_dbscan:.3f} | Noise points: {noise_count}"
    )
    print("\nFigures saved to folder:", OUT_DIR)


if __name__ == "__main__":
    # Disable some non-critical warnings (e.g., edgecolor)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
