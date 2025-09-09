import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from lib.spectral_clustering import spectral_clustering_classifier, trova_brani_rappresentativi, \
    silhouette_score_analysis_spectral_clustering
from lib.extract_data_features import get_audio_features
from lib.utils import salva_risultati_markdown, plot_tsne_clustering, plot_clusters_results, computer_clustering_scores
from lib.k_means_clustering import (
    kmeans_clustering_classifier,
    trova_brani_rappresentativi_kmeans,
    silhouette_score_analysis_kmeans,
    elbow_method_kmeans,
)
from lib.dbscan_clustering import (
    dbscan_clustering_classifier,
    trova_brani_rappresentativi_dbscan,
    silhouette_analysis_dbscan,
    k_distance_plot_dbscan,
)

CSV_FEATURE_FILENAME = "dataset/songs_features/songs_features_trap.csv"
SONGS_DIR = "dataset/songs/trap/"  # Cambia con il percorso della tua cartella di canzoni

RESULTS_SC = "clustering_results/spectral_clustering"
RESULTS_KM = "clustering_results/kmeans"
RESULTS_DBSCAN = "clustering_results/dbscan"

N_CLUSTERS = 5  # Numero di cluster da creare
PCA_COMPONENTS = 0.98 # Percentuale di varianza da mantenere con PCA

SPECTRAL_CLUSTERING_GAMMA = 0.2  # Parametro gamma per lo spectral clustering

# Parametri DBSCAN (valori di default, puoi regolarli dopo aver guardato i grafici)
DBSCAN_EPS = 0.7
DBSCAN_MIN_SAMPLES = 8
DBSCAN_METRIC = 'euclidean'

if __name__ == "__main__":
    print("Caricamento delle feature audio...")
    filenames, music_genres, features, features_names = get_audio_features(SONGS_DIR, CSV_FEATURE_FILENAME)
    print("Shape feature array:", features.shape)

    # Rimozione dei duplicati
    features, unique_indices = np.unique(features, axis=0, return_index=True)
    filenames = [filenames[i] for i in unique_indices]
    music_genres = [music_genres[i] for i in unique_indices]
    print("Shape feature array dopo rimozione duplicati:", features.shape)

    # Normalizzazione delle feature (MinMax: range 0-1)
    scaler = MinMaxScaler()
    features_norm = scaler.fit_transform(features)

    # Copia per report dettagliato (prima della PCA)
    features_norm_original = features_norm.copy()

    # Riduzione della dimensionalità con PCA
    pca = PCA(n_components=PCA_COMPONENTS, svd_solver='full')
    features_reduced = pca.fit_transform(features_norm)
    print("Shape feature array dopo PCA:", features_reduced.shape)

    # =========================== SPECTRAL CLUSTERING ===========================
    print(f"Esecuzione del spectral clustering con {N_CLUSTERS} cluster...")
    spectral_clustering_labels = spectral_clustering_classifier(features=features_reduced, n_clusters=N_CLUSTERS, gamma=SPECTRAL_CLUSTERING_GAMMA)

    print("Spectral clustering classification completed!")
    plot_clusters_results(filenames, features_reduced, spectral_clustering_labels, RESULTS_SC + "/clusters_plot.png")

    print("Analisi dei cluster (Spectral)...")
    trova_brani_rappresentativi(features_reduced, spectral_clustering_labels, filenames)
    plot_tsne_clustering(features_reduced, spectral_clustering_labels, filenames, RESULTS_SC + "/tsne_clusters_plot.png")
    computer_clustering_scores(features_reduced, spectral_clustering_labels)

    print("Generazione report Markdown spectral clustering...")
    report_path = salva_risultati_markdown(filenames, features_reduced, spectral_clustering_labels, feature_names=None, path=RESULTS_SC + "/report_SC.md", n_repr=5, generi=music_genres)
    report_detailed_path = salva_risultati_markdown(filenames, features_norm_original, spectral_clustering_labels, feature_names=features_names, path=RESULTS_SC + "/report_dettagliato_feature_originali_SC.md", n_repr=5, generi=music_genres)
    print(f"Report generato: {report_path}")
    print(f"Report dettagliato generato: {report_detailed_path}")

    # Analisi del silhouette score per diversi numeri di cluster, spectral clustering
    silhouette_score_analysis_spectral_clustering(features_reduced, gamma=SPECTRAL_CLUSTERING_GAMMA, range_k=(2, 20),
                                                  fig_name=RESULTS_SC + "/silhouette_analysis_spectral_clustering.png")

    # ============================= K-MEANS CLUSTERING =============================
    print(f"\nEsecuzione del K-Means clustering con {N_CLUSTERS} cluster...")
    kmeans_labels, kmeans_centers = kmeans_clustering_classifier(features_reduced, n_clusters=N_CLUSTERS)
    print("K-Means clustering completed!")

    plot_clusters_results(filenames, features_reduced, kmeans_labels, RESULTS_KM + "/clusters_plot_kmeans.png")
    plot_tsne_clustering(features_reduced, kmeans_labels, filenames, RESULTS_KM + "/tsne_clusters_plot_kmeans.png")

    print("Analisi dei cluster (K-Means)...")
    trova_brani_rappresentativi_kmeans(features_reduced, kmeans_labels, filenames, n=5, centers=kmeans_centers)
    computer_clustering_scores(features_reduced, kmeans_labels)

    print("Generazione report Markdown K-Means...")
    report_km = salva_risultati_markdown(filenames, features_reduced, kmeans_labels, feature_names=None, path=RESULTS_KM + "/report_KM.md", n_repr=5, generi=music_genres)
    report_km_detailed = salva_risultati_markdown(filenames, features_norm_original, kmeans_labels, feature_names=features_names, path=RESULTS_KM + "/report_dettagliato_feature_originali_KM.md", n_repr=5, generi=music_genres)
    print(f"Report K-Means generato: {report_km}")
    print(f"Report dettagliato K-Means generato: {report_km_detailed}")

    # Analisi silhouette K-Means
    silhouette_score_analysis_kmeans(features_reduced, range_k=(2, 20), fig_name=RESULTS_KM + "/silhouette_analysis_kmeans.png")
    # Elbow method K-Means
    elbow_method_kmeans(features_reduced, range_k=(2, 20), fig_name=RESULTS_KM + "/elbow_analysis_kmeans.png")

    # ============================= DBSCAN CLUSTERING =============================
    print("\nAnalisi esplorativa per DBSCAN (k-distance plot)...")
    try:
        k_distance_plot_dbscan(features_reduced, k=DBSCAN_MIN_SAMPLES, fig_name=RESULTS_DBSCAN + "/k_distance_dbscan.png")
    except Exception as e:
        print(f"Errore k-distance plot DBSCAN: {e}")

    print("Analisi silhouette su diversi eps per DBSCAN...")
    eps_values = np.linspace(0.1, 2.0, 25)
    try:
        silhouette_analysis_dbscan(features_reduced, eps_values, min_samples=DBSCAN_MIN_SAMPLES, metric=DBSCAN_METRIC, fig_name=RESULTS_DBSCAN + "/silhouette_analysis_dbscan.png")
    except Exception as e:
        print(f"Errore silhouette analysis DBSCAN: {e}")

    print(f"Esecuzione DBSCAN finale (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    dbscan_labels, dbscan_model = dbscan_clustering_classifier(features_reduced, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric=DBSCAN_METRIC)
    unique_dbscan_clusters = [c for c in np.unique(dbscan_labels) if c != -1]
    print(f"Cluster trovati (senza noise): {len(unique_dbscan_clusters)} - con noise label -1 totale classi: {len(np.unique(dbscan_labels))}")
    noise_ratio = np.sum(dbscan_labels == -1) / len(dbscan_labels)
    print(f"Percentuale noise: {noise_ratio*100:.1f}%")

    plot_clusters_results(filenames, features_reduced, dbscan_labels, RESULTS_DBSCAN + "/clusters_plot_dbscan.png")
    plot_tsne_clustering(features_reduced, dbscan_labels, filenames, RESULTS_DBSCAN + "/tsne_clusters_plot_dbscan.png")

    trova_brani_rappresentativi_dbscan(features_reduced, dbscan_labels, filenames, n=5)

    # Metriche (silhouette e Davies-Bouldin) evitando di calcolare se cluster insufficienti
    try:
        if len(unique_dbscan_clusters) >= 2:
            # Usiamo solo punti non-noise per silhouette e DB
            valid_mask = dbscan_labels != -1
            computer_clustering_scores(features_reduced[valid_mask], dbscan_labels[valid_mask])
        else:
            print("Metriche DBSCAN saltate: meno di 2 cluster validi.")
    except Exception as e:
        print(f"Errore calcolo metriche DBSCAN: {e}")

    print("Generazione report Markdown DBSCAN...")
    report_dbscan = salva_risultati_markdown(filenames, features_reduced, dbscan_labels, feature_names=None, path=RESULTS_DBSCAN + "/report_DBSCAN.md", n_repr=5, generi=music_genres)
    report_dbscan_detailed = salva_risultati_markdown(filenames, features_norm_original, dbscan_labels, feature_names=features_names, path=RESULTS_DBSCAN + "/report_dettagliato_feature_originali_DBSCAN.md", n_repr=5, generi=music_genres)
    print(f"Report DBSCAN generato: {report_dbscan}")
    print(f"Report dettagliato DBSCAN generato: {report_dbscan_detailed}")

    print("\nPipeline completata (Spectral + K-Means + DBSCAN)")
