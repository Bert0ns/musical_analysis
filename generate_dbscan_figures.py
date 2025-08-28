"""
Generazione figure per illustrare DBSCAN.

Figure prodotte:

1. dbscan_core_border_noise.png
   - Visualizza un dataset con cluster + rumore evidenziando punti core, border e noise.

2. dbscan_eps_variation.png
   - Mostra l'effetto di eps variando eps (min_samples fisso).

3. dbscan_min_samples_variation.png
   - Mostra l'effetto di min_samples variando min_samples (eps fisso).

4. dbscan_k_distance_plot.png
   - K-distance plot (distanza al k-esimo vicino) con stima "ginocchiata" per suggerire eps.

5. dbscan_variable_density.png
   - Dataset a densità variabile: fallimento di un singolo eps (cluster uno unito o cluster sparsi/noise).

6. dbscan_param_grid_heatmap.png
   - Heatmap silhouette media (solo punti non-noise) su una griglia (eps, min_samples).

Dipendenze:
- Python 3.9+
- numpy
- scikit-learn
- matplotlib
- seaborn (solo per heatmap)
- kneed (facoltativo; se non presente viene usato un fallback semplice per stimare il knee)

Installazione pacchetti (esempio):
pip install numpy scikit-learn matplotlib seaborn kneed

Esecuzione:
python generate_dbscan_figures.py

Le immagini vengono salvate nella cartella figs_dbscan/.
"""

import os
import warnings
import numpy as np
import matplotlib
from numpy.ma.extras import apply_along_axis

matplotlib.use('Agg')  # Usa backend non interattivo
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

OUT_DIR = "figs_dbscan"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- Utility ----------

def scatter_dbscan(X, labels, title="", ax=None, core_samples_mask=None):
    """
    Visualizza i risultati DBSCAN.
    - labels: etichette cluster con -1 per noise.
    - core_samples_mask: boolean array per i punti core (True se core).
    """
    if ax is None:
        ax = plt.gca()

    unique_labels = np.unique(labels)
    palette = pyplot.get_cmap("tab10", len(unique_labels))

    # Se non fornita maschera core, consideriamo tutti non-core
    if core_samples_mask is None:
        core_samples_mask = np.zeros_like(labels, dtype=bool)

    for idx, lbl in enumerate(unique_labels):
        class_member_mask = labels == lbl
        if lbl == -1:
            # noise in grigio
            col = (0.55, 0.55, 0.55, 0.6)
        else:
            col = palette(idx)

        # Core points
        xy_core = X[class_member_mask & core_samples_mask]
        # Border points
        xy_border = X[class_member_mask & ~core_samples_mask]

        if xy_core.size > 0:
            ax.scatter(xy_core[:, 0], xy_core[:, 1],
                       s=28, c=[col], marker="o", edgecolor="black", linewidth=0.4,
                       alpha=0.95, label=f"Cluster {lbl} core" if lbl != -1 else "Noise (core?)")
        if xy_border.size > 0:
            ax.scatter(xy_border[:, 0], xy_border[:, 1],
                       s=20, c=[col], marker="o", edgecolor="none", alpha=0.55,
                       label=f"Cluster {lbl} border" if lbl != -1 else "Noise")

    # Evita doppioni in legenda
    handles, labels_leg = ax.get_legend_handles_labels()
    legend_clean = []
    seen = set()
    for h, l in zip(handles, labels_leg):
        if l not in seen and ("core" in l or "border" in l or "Noise" in l):
            legend_clean.append((h, l))
            seen.add(l)
    if len(legend_clean) <= 12:
        ax.legend([h for h, _ in legend_clean], [l for _, l in legend_clean],
                  loc="best", fontsize=8, frameon=True)
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

def compute_core_samples_mask(dbscan_model, n_samples):
    mask = np.zeros(n_samples, dtype=bool)
    if hasattr(dbscan_model, "core_sample_indices_"):
        mask[dbscan_model.core_sample_indices_] = True
    return mask

# ---------- Figure 1: Core / Border / Noise ----------

def custom_scatter_dbscan(X, labels, cluster_colors, title="", ax=None, core_samples_mask=None):
    if ax is None:
        ax = plt.gca()
    unique_labels = np.unique(labels)
    if core_samples_mask is None:
        core_samples_mask = np.zeros_like(labels, dtype=bool)
    for lbl in unique_labels:
        class_member_mask = labels == lbl
        col = cluster_colors.get(lbl, (0.2, 0.2, 0.2))
        xy_core = X[class_member_mask & core_samples_mask]
        xy_border = X[class_member_mask & ~core_samples_mask]
        if xy_core.size > 0:
            ax.scatter(xy_core[:, 0], xy_core[:, 1],
                       s=28, c=[col], marker="o", edgecolor="black", linewidth=0.4,
                       alpha=0.95, label=f"Cluster {lbl} core" if lbl != -1 else "Noise (core?)")
        if xy_border.size > 0:
            ax.scatter(xy_border[:, 0], xy_border[:, 1],
                       s=20, c=[col], marker="o", edgecolor="none", alpha=0.55,
                       label=f"Cluster {lbl} border" if lbl != -1 else "Noise")
    handles, labels_leg = ax.get_legend_handles_labels()
    legend_clean = []
    seen = set()
    for h, l in zip(handles, labels_leg):
        if l not in seen and ("core" in l or "border" in l or "Noise" in l):
            legend_clean.append((h, l))
            seen.add(l)
    if len(legend_clean) <= 12:
        ax.legend([h for h, _ in legend_clean], [l for _, l in legend_clean],
                  loc="best", fontsize=8, frameon=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_core_border_noise():
    X, _ = make_blobs(n_samples=550,
                      centers=[(-4, -2), (-1, 2), (2.5, -1)],
                      cluster_std=[0.6, 0.9, 0.5],
                      random_state=RANDOM_STATE)
    # Aggiungiamo rumore sparso
    noise = np.random.uniform(low=-6, high=4, size=(50, 2))
    X = np.vstack([X, noise])
    X = StandardScaler().fit_transform(X)

    # Parametri scelti per distinguere cluster e identificare noise
    eps = 0.35
    min_samples = 8
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(X)
    core_mask = compute_core_samples_mask(db, X.shape[0])

    # Mappa colori: 0=viola, 1=arancione, 2=azzurro, -1=grigio
    cluster_colors = {
        0: (148/255, 0/255, 211/255),     # viola
        1: (255/255, 140/255, 0/255),     # arancione
        2: (30/255, 144/255, 255/255),    # azzurro
        -1: (0.55, 0.55, 0.55, 0.6)       # noise (grigio)
    }

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    custom_scatter_dbscan(
        X, labels,
        title=f"Core, border e noise (DBSCAN) | eps={eps}, min_samples={min_samples}",
        cluster_colors=cluster_colors,
        ax=ax, core_samples_mask=core_mask
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbscan_core_border_noise.png"), dpi=170)
    plt.close(fig)

# ---------- Figure 2: Variazione eps (min_samples fisso) ----------

def fig_eps_variation():
    X, _ = make_blobs(n_samples=600,
                      centers=[(-3, -2), (-1, 3), (2.2, 0.5)],
                      cluster_std=[0.5, 1.1, 0.6],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    eps_values = [0.20, 0.30, 0.40, 0.70]
    min_samples = 8

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()
    for ax, eps in zip(axes, eps_values):
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X)
        core_mask = compute_core_samples_mask(db, X.shape[0])
        valid = labels != -1
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        sil = np.nan
        if np.unique(labels[valid]).size > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if np.sum(valid) > 1:
                    try:
                        sil = silhouette_score(X[valid], labels[valid])
                    except Exception:
                        sil = np.nan
        title = f"eps={eps:.2f} | cluster={n_clusters} | silhouette={sil:.2f}" if not np.isnan(sil) else f"eps={eps:.2f} | cluster={n_clusters}"
        scatter_dbscan(X, labels, title=title, ax=ax, core_samples_mask=core_mask)

    fig.suptitle(f"Influenza di eps (min_samples={min_samples})", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, "dbscan_eps_variation.png"), dpi=170)
    plt.close(fig)

# ---------- Figure 3: Variazione min_samples (eps fisso) ----------

def fig_min_samples_variation():
    X, _ = make_blobs(n_samples=600,
                      centers=[(-2.5, -1.5), (-0.5, 2.5), (2.5, 0.2)],
                      cluster_std=[0.55, 0.85, 0.55],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    min_samples_list = [4, 8, 15, 30]
    eps = 0.50

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.ravel()

    for ax, ms in zip(axes, min_samples_list):
        db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
        labels = db.fit_predict(X)
        core_mask = compute_core_samples_mask(db, X.shape[0])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        valid = labels != -1
        sil = np.nan
        if np.unique(labels[valid]).size > 1:
            try:
                sil = silhouette_score(X[valid], labels[valid])
            except Exception:
                sil = np.nan
        title = f"min_samples={ms} | cluster={n_clusters} | silhouette={sil:.2f}" if not np.isnan(sil) else f"min_samples={ms} | cluster={n_clusters}"
        scatter_dbscan(X, labels, title=title, ax=ax, core_samples_mask=core_mask)

    fig.suptitle(f"Influenza di min_samples (eps={eps})", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT_DIR, "dbscan_min_samples_variation.png"), dpi=170)
    plt.close(fig)

# ---------- Figure 4: k-distance plot (stima eps) ----------

def knee_index(y):
    """
    Ritorna un indice 'ginocchiata' semplice usando la massima distanza
    di ogni punto dalla corda (primo-ultimo) nel grafico (metodo geometrico).
    """
    x = np.arange(len(y))
    # retta fra primo e ultimo
    p1 = np.array([x[0], y[0]], dtype=float)
    p2 = np.array([x[-1], y[-1]], dtype=float)
    line_vec = p2 - p1
    line_vec_norm = line_vec / (np.linalg.norm(line_vec) + 1e-12)
    distances = []
    for i in range(len(y)):
        p = np.array([x[i], y[i]], dtype=float)
        proj_len = np.dot(p - p1, line_vec_norm)
        proj_point = p1 + proj_len * line_vec_norm
        dist = np.linalg.norm(p - proj_point)
        distances.append(dist)
    return int(np.argmax(distances))

def fig_k_distance_plot():
    X, _ = make_blobs(n_samples=600,
                      centers=[(-3, -2), (-1, 3), (2.5, 0.5)],
                      cluster_std=[0.6, 1.0, 0.55],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    min_samples = 8
    k = min_samples  # spesso si usa k = min_samples
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, _ = nbrs.kneighbors(X)
    # prendiamo la distanza all'ultimo vicino (k-esimo)
    k_dists = distances[:, -1]
    k_dists_sorted = np.sort(k_dists)

    # Stima ginocchiata
    idx_knee = knee_index(k_dists_sorted)
    eps_est = k_dists_sorted[idx_knee]

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(k_dists_sorted, linewidth=1.2)
    ax.axvline(idx_knee, color="tomato", linestyle="--", label=f"indice knee={idx_knee}")
    ax.axhline(eps_est, color="green", linestyle="--", label=f"eps stimato≈{eps_est:.3f}")
    ax.set_title(f"k-distance plot (k={k})")
    ax.set_xlabel("Punti ordinati")
    ax.set_ylabel(f"Distanza al {k}-esimo vicino")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbscan_k_distance_plot.png"), dpi=170)
    plt.close(fig)

# ---------- Figure 5: Densità variabile ----------

def fig_variable_density():
    # Creiamo cluster con densità molto diversa
    X1, _ = make_blobs(n_samples=500, centers=[(-2, -2)], cluster_std=0.30, random_state=RANDOM_STATE)
    X2, _ = make_blobs(n_samples=300, centers=[(2, -1)], cluster_std=0.80, random_state=RANDOM_STATE)
    X3, _ = make_blobs(n_samples=200, centers=[(0, 2.5)], cluster_std=1.10, random_state=RANDOM_STATE)
    X = np.vstack([X1, X2, X3])
    X = StandardScaler().fit_transform(X)

    eps_values = [0.25, 0.50, 0.90]
    min_samples = 8

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, eps in zip(axes, eps_values):
        db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
        labels = db.fit_predict(X)
        core_mask = compute_core_samples_mask(db, X.shape[0])
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        scatter_dbscan(X, labels, title=f"eps={eps} | cluster={n_clusters}", ax=ax, core_samples_mask=core_mask)

    fig.suptitle("Dataset a densità variabile: un singolo eps è subottimale", y=0.97)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT_DIR, "dbscan_variable_density.png"), dpi=170)
    plt.close(fig)

# ---------- Figure 6: Heatmap silhouette vs (eps, min_samples) ----------

def fig_param_grid_heatmap():
    X, _ = make_blobs(n_samples=700,
                      centers=[(-3, -2), (-1, 2.5), (2.2, -0.5)],
                      cluster_std=[0.5, 0.8, 0.6],
                      random_state=RANDOM_STATE)
    X = StandardScaler().fit_transform(X)

    eps_grid = np.linspace(0.25, 0.80, 12)
    min_samples_grid = [4, 6, 8, 10, 14, 18, 24]

    heat = np.full((len(min_samples_grid), len(eps_grid)), np.nan)

    for i, ms in enumerate(min_samples_grid):
        for j, eps in enumerate(eps_grid):
            db = DBSCAN(eps=eps, min_samples=ms, n_jobs=-1)
            labels = db.fit_predict(X)
            # Calcoliamo silhouette sui soli non-noise se almeno due cluster
            mask = labels != -1
            unique_valid = np.unique(labels[mask])
            if unique_valid.size > 1 and mask.sum() > len(unique_valid):
                try:
                    sil = silhouette_score(X[mask], labels[mask])
                    heat[i, j] = sil
                except Exception:
                    heat[i, j] = np.nan

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(heat, annot=False, cmap="viridis", xticklabels=[f"{e:.2f}" for e in eps_grid],
                yticklabels=min_samples_grid, ax=ax, cbar_kws={"label": "Silhouette (non-noise)"})
    ax.set_xlabel("eps")
    ax.set_ylabel("min_samples")
    ax.set_title("Silhouette media vs eps e min_samples")
    # Evidenzia il massimo
    if np.isfinite(heat).any():
        max_idx = np.nanargmax(heat)
        mi, mj = np.unravel_index(max_idx, heat.shape)
        ax.scatter(mj + 0.5, mi + 0.5, color="red", marker="o", s=60, linewidth=1.2,
                   edgecolor="white", label="Massimo")
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "dbscan_param_grid_heatmap.png"), dpi=170)
    plt.close(fig)

def main():
    fig_core_border_noise()
    fig_eps_variation()
    fig_min_samples_variation()
    fig_k_distance_plot()
    fig_variable_density()
    fig_param_grid_heatmap()
    print(f"Figure salvate in: {OUT_DIR}")

if __name__ == "__main__":
    main()