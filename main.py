from data_clustering import spectral_clustering_classifier, visualizza_risultati, analizza_cluster, \
    trova_brani_rappresentativi, visualizza_tsne, valuta_cluster
from extract_data_features import get_audio_features

CSV_FEATURE_FILENAME = "audio_features.csv"
SONGS_DIR = "songs"  # Cambia con il percorso della tua cartella di canzoni

N_CLUSTERS = 6  # Numero di cluster da creare

if __name__ == "__main__":
    print("Caricamento delle feature audio...")
    filenames, features = get_audio_features(SONGS_DIR, CSV_FEATURE_FILENAME)
    print("Shape feature array:", features.shape)
    print(f"Esecuzione del spectral clustering con {N_CLUSTERS} cluster...")
    labels = spectral_clustering_classifier(features=features, n_clusters=N_CLUSTERS, gamma=1)

    print("Classificazione completata!")
    visualizza_risultati(filenames, features, labels)

    print("Analisi dei cluster...")
    analizza_cluster(features, labels)
    trova_brani_rappresentativi(features, labels, filenames)
    visualizza_tsne(features, labels, filenames)
    valuta_cluster(features, labels)