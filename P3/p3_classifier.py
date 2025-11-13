from P1 import ByteLevelBPE
import numpy as np
import pickle
import os

currentDirectory = os.path.dirname(os.path.abspath(__file__))


# 1: Implementa funciones para cargar token embeddings de un modelo entrenado.
def loadEmbeddings(
    filePath: str = os.path.join(
        os.path.dirname(currentDirectory), "skipgram_embeddings.txt"
    )
) -> np.ndarray:
    """
    Load token embeddings from a text file.

    The first line of the file is expected to contain two integers:
    the vocabulary size and the embedding dimension.

    Args:
        - filePath (str): Path to the embeddings file.

    Returns:
        - np.ndarray: A 2D array containing the embeddings.
    """
    with open(filePath, "r", encoding="utf-8") as f:
        firstLine = f.readline().strip()
        vocabSize, embeddingDim = map(int, firstLine.split())
        embeddings = np.zeros((vocabSize, embeddingDim))

        # Read each embedding vector
        for i in range(vocabSize):
            parts = f.readline().strip().split()
            vector = parts[-embeddingDim:]
            embeddings[i] = np.array(vector, dtype=np.float32)

    return embeddings


# 1: Implementa funciones  para obtener una sola embedding por texto de entrada.
# Puedes usar la función de agregación que quieras.
def obtainEmbeddings(
    text: str,
    embeddings: np.ndarray,
    modelPath: str = os.path.join(os.path.dirname(currentDirectory), "bpe_model.pkl"),
) -> np.ndarray:
    """
    Obtain a single embedding for the input text by aggregating token embeddings.

    Args:
        - text (str): The input text.
        - embeddings (np.ndarray): The token embeddings matrix.
        - modelPath (str): Path to the BPE model file.

    Returns:
        - np.ndarray: The aggregated embedding for the input text.
    """
    with open(modelPath, "rb") as f:
        bpe: ByteLevelBPE = pickle.load(f)

    tokenIds = bpe.encode(text)

    return np.mean(embeddings[tokenIds], axis=0)


# TODO 2: Implementa la clase LogisticRegression con los siguientes componentes:
class LogisticRegression:
    def __init__(self):
        # * Campos para almacenar los pesos, sesgo y learning rate.
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        # * Método `forward`, que implemente la combinación lineal de los pesos y sesgo con la entrada seguida de la función logística.
        pass

    def backward(self, X: np.ndarray, yPred: np.ndarray, yTrue: np.ndarray) -> None:
        # * Método `backward` que, dada la entrada, la salida obtenida y la salida deseada, modifique los parámetros del modelo.
        pass

    def compute_loss(self, yPred: np.ndarray, yTrue: np.ndarray) -> float:
        # * Método `compute_loss`, que implemente la función de entropía cruzada binaria.
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # * Método `fit`, que recibe los datos de entrenamiento y optimiza el modelo mediante descenso de gradiente.
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # # Método `predict`, que usa `forward` y obtiene la salida final de inferencia en `{0, 1}`.
        pass


# TODO 3: Implementa una función principal que realice todos los pasos necesarios para entrenar y evaluar un modelo de regresión logística que usa agregados de token embeddings como características.

# NOTA: no es necesario almacenar el modelo de regresión.
