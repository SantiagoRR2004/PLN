# P0 - Estadísticas del corpus

import matplotlib.pyplot as plt

# TODO 1: Obtener los tokens del corpus del archivo "tiny_cc_news.txt", en el orden original

# TODO 2: Leer los tokens correspondientes a stopwords desde el archivo "stopwords.txt"

# TODO 3: Obtener diccionarios de frecuencias para tokens y tokens que no son stopwords

# TODO 4: Obtener estadísticas básicas del corpus:
# * Número de documentos
# * Número de tokens
# * Número de stopwords
# * Número de tokens que no son stopwords
# * Proporción de stopwords
# * Tamaño del vocabulario
# * Tamaño del vocabulario sin stopwords
# * Longitud media de documento
# * Longitud mínima/máxima de documento
# * Longitud media de token
# * Longitud mínima/máxima de token
# * Longitud media de tokens sin stopwords
# * Longitud mínima/máxima de tokens sin stopwords

# TODO 5: Obtener métricas de riqueza léxica
# * Type-Token Ratio (TTR)
# * Hapax legomena

# TODO 6: Obtener los 10 tokens más frecuentes y los 10 tokens más frecuentes sin stopwords

# TODO 7: Obtener bigramas y trigramas de tokens y de tokens sin stopwords

# TODO 8: Obtener los 10 bigramas y trigramas más frecuentes


def plot_zipfs_law(sorted_freqs, ranks):
    plt.figure(figsize=(6, 4))
    plt.plot(ranks, sorted_freqs, marker=".", linestyle="solid")
    plt.title("Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


# TODO 9: Mostrar gráfica de la Ley de Zipf
