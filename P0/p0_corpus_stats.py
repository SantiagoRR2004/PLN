# P0 - Estadísticas del corpus

import matplotlib.pyplot as plt
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))

# 1: Obtener los tokens del corpus del archivo "tiny_cc_news.txt", en el orden original
with open(
    os.path.join(currentDirectory, "tiny_cc_news.txt"), "r", encoding="utf-8"
) as file:
    corpus = file.read()

# Now all lowercase
corpus = corpus.lower()

corpus = corpus.split("\n\n")


# 2: Leer los tokens correspondientes a stopwords desde el archivo "stopwords.txt"
with open(
    os.path.join(currentDirectory, "stopwords.txt"), "r", encoding="utf-8"
) as file:
    stopwords = file.readlines()

stopwords = [word.strip() for word in stopwords]

# 3: Obtener diccionarios de frecuencias para tokens y tokens que no son stopwords
fTokens = {}
fTokensNoStopwords = {}

for document in corpus:
    tokens = document.split()
    for token in tokens:
        # Frecuencia de tokens
        if token in fTokens:
            fTokens[token] += 1
        else:
            fTokens[token] = 1
        # Frecuencia de tokens que no son stopwords
        if token not in stopwords:
            if token in fTokensNoStopwords:
                fTokensNoStopwords[token] += 1
            else:
                fTokensNoStopwords[token] = 1

# 4: Obtener estadísticas básicas del corpus:
# * Número de documentos
print(f"Número de documentos: {len(corpus)}")
# * Número de tokens
nTokens = sum(fTokens.values())
print(f"Número de tokens: {nTokens}")
# * Número de stopwords
nTokensNoStopwords = sum(fTokensNoStopwords.values())
nTokensStopwords = nTokens - nTokensNoStopwords
print(f"Número de stopwords: {nTokensStopwords}")
# * Número de tokens que no son stopwords
print(f"Número de tokens que no son stopwords: {nTokensNoStopwords}")
# * Proporción de stopwords
print(f"Proporción de stopwords: {nTokensStopwords / nTokens:.2%}")
# * Tamaño del vocabulario
print(f"Tamaño del vocabulario: {len(fTokens)}")
# * Tamaño del vocabulario sin stopwords
print(f"Tamaño del vocabulario sin stopwords: {len(fTokensNoStopwords)}")
# * Longitud media de documento
print(f"Longitud media de documento: {nTokens / len(corpus):.2f}")
# * Longitud mínima/máxima de documento
print(f"Longitud mínima de documento: {min(len(doc.split()) for doc in corpus)}")
print(f"Longitud máxima de documento: {max(len(doc.split()) for doc in corpus)}")
# * Longitud media de token
print(f"Longitud media de token: {nTokens / sum(len(token) for token in fTokens):.2f}")
# * Longitud mínima/máxima de token
print(f"Longitud mínima de token: {min(len(token) for token in fTokens)}")
print(f"Longitud máxima de token: {max(len(token) for token in fTokens)}")
# * Longitud media de tokens sin stopwords
print(
    f"Longitud media de tokens sin stopwords: {nTokensNoStopwords / sum(len(token) for token in fTokensNoStopwords):.2f}"
)
# * Longitud mínima/máxima de tokens sin stopwords
print(
    f"Longitud mínima de tokens sin stopwords: {min(len(token) for token in fTokensNoStopwords)}"
)
print(
    f"Longitud máxima de tokens sin stopwords: {max(len(token) for token in fTokensNoStopwords)}"
)

print(60 * "=")

# 5: Obtener métricas de riqueza léxica
# * Type-Token Ratio (TTR)
print(f"Type-Token Ratio (TTR): {len(fTokens) / nTokens:.2%}")
# * Hapax legomena https://en.wikipedia.org/wiki/Hapax_legomenon
nOneToken = sum(1 for freq in fTokens.values() if freq == 1)
print(f"Hapax legomena: {nOneToken / nTokens:.2%}")

print(60 * "=")

# 6: Obtener los 10 tokens más frecuentes y los 10 tokens más frecuentes sin stopwords
fTokens = dict(sorted(fTokens.items(), key=lambda item: item[1], reverse=True))
fTokensNoStopwords = dict(
    sorted(fTokensNoStopwords.items(), key=lambda item: item[1], reverse=True)
)

print("10 tokens más frecuentes:")
for i, (token, freq) in enumerate(fTokens.items()):
    if i >= 10:
        break
    print(f"\t{token}: {freq}")

print("10 tokens más frecuentes sin stopwords:")
for i, (token, freq) in enumerate(fTokensNoStopwords.items()):
    if i >= 10:
        break
    print(f"\t{token}: {freq}")

print(60 * "=")


# 7: Obtener bigramas y trigramas de tokens y de tokens sin stopwords
fBigrams = {}
fTrigrams = {}
fBigramsNoStopwords = {}
fTrigramsNoStopwords = {}

for document in corpus:
    words = document.split()
    for i in range(len(words) - 1):

        bigram = (words[i], words[i + 1])
        if bigram in fBigrams:
            fBigrams[bigram] += 1
        else:
            fBigrams[bigram] = 1
        if all(word not in stopwords for word in bigram):
            if bigram in fBigramsNoStopwords:
                fBigramsNoStopwords[bigram] += 1
            else:
                fBigramsNoStopwords[bigram] = 1

        if i < len(words) - 2:
            trigram = (words[i], words[i + 1], words[i + 2])
            if trigram in fTrigrams:
                fTrigrams[trigram] += 1
            else:
                fTrigrams[trigram] = 1
            if all(word not in stopwords for word in trigram):
                if trigram in fTrigramsNoStopwords:
                    fTrigramsNoStopwords[trigram] += 1
                else:
                    fTrigramsNoStopwords[trigram] = 1

# 8: Obtener los 10 bigramas y trigramas más frecuentes
fBigrams = dict(sorted(fBigrams.items(), key=lambda item: item[1], reverse=True))
fTrigrams = dict(sorted(fTrigrams.items(), key=lambda item: item[1], reverse=True))
fBigramsNoStopwords = dict(
    sorted(fBigramsNoStopwords.items(), key=lambda item: item[1], reverse=True)
)
fTrigramsNoStopwords = dict(
    sorted(fTrigramsNoStopwords.items(), key=lambda item: item[1], reverse=True)
)

print("10 bigramas más frecuentes:")
for i, (bigram, freq) in enumerate(fBigrams.items()):
    if i >= 10:
        break
    print(f"\t{' '.join(bigram)}: {freq}")

print("10 trigramas más frecuentes:")
for i, (trigram, freq) in enumerate(fTrigrams.items()):
    if i >= 10:
        break
    print(f"\t{' '.join(trigram)}: {freq}")

print("10 bigramas más frecuentes sin stopwords:")
for i, (bigram, freq) in enumerate(fBigramsNoStopwords.items()):
    if i >= 10:
        break
    print(f"\t{' '.join(bigram)}: {freq}")

print("10 trigramas más frecuentes sin stopwords:")
for i, (trigram, freq) in enumerate(fTrigramsNoStopwords.items()):
    if i >= 10:
        break
    print(f"\t{' '.join(trigram)}: {freq}")


# 9: Mostrar gráfica de la Ley de Zipf
def plot_zipfs_law(sorted_freqs, ranks):
    plt.figure(figsize=(6, 4))
    plt.loglog(ranks, sorted_freqs, marker=".", linestyle="solid")
    plt.title("Zipf's Law")
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()


plot_zipfs_law(list(fTokens.values()), list(range(1, len(fTokens) + 1)))
