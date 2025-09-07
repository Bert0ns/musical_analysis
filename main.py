import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from data_clustering import spectral_clustering_classifier, trova_brani_rappresentativi, valuta_cluster
from extract_data_features import get_audio_features
from utils import salva_risultati_markdown, visualizza_tsne, analizza_cluster, visualizza_risultati

CSV_FEATURE_FILENAME = "dataset/audio_features.csv"
SONGS_DIR = "dataset/songs"  # Cambia con il percorso della tua cartella di canzoni
RESULTS = "clustering_results"

N_CLUSTERS = 3  # Numero di cluster da creare
PCA_COMPONENTS = 0.98  # Percentuale di varianza da mantenere con PCA

SPECTRAL_CLUSTERING_GAMMA = 0.1  # Parametro gamma per lo spectral clustering

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

    print(f"Esecuzione del spectral clustering con {N_CLUSTERS} cluster...")
    labels = spectral_clustering_classifier(features=features_reduced, n_clusters=N_CLUSTERS, gamma=SPECTRAL_CLUSTERING_GAMMA)

    print("Classificazione completata!")
    visualizza_risultati(filenames, features_reduced, labels, RESULTS + "/clusters_plot.png")

    print("Analisi dei cluster...")
    analizza_cluster(features_reduced, labels)
    trova_brani_rappresentativi(features_reduced, labels, filenames)
    visualizza_tsne(features_reduced, labels, filenames, RESULTS + "/tsne_clusters_plot.png")
    valuta_cluster(features_reduced, labels, RESULTS + "/cluster_evaluation.png")

    print("Generazione report Markdown...")
    # Report principale (sulle feature ridotte usate per il clustering)
    report_path = salva_risultati_markdown(filenames, features_reduced, labels, feature_names=None, path=RESULTS + "/report.md", n_repr=5, generi=music_genres)
    # Report opzionale con statistiche sulle feature originali normalizzate (senza PCA) usando i nomi
    report_detailed_path = salva_risultati_markdown(filenames, features_norm_original, labels, feature_names=features_names, path=RESULTS + "/report_dettagliato_feature_originali.md", n_repr=5, generi=music_genres)
    print(f"Report generato: {report_path}")
    print(f"Report dettagliato generato: {report_detailed_path}")
