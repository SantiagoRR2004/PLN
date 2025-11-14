import matplotlib.pyplot as plt
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


# 2: Implementa la clase LogisticRegression con los siguientes componentes:
class LogisticRegression:

    def __init__(self, learningRate: float = 0.01, epochs: int = 100000) -> None:
        """
        Campos para almacenar los pesos, sesgo y learning rate.

        Args:
            - learningRate (float): The learning rate for gradient descent.
            - epochs (int): The number of training epochs.

        Returns:
            - None
        """
        self.weights: np.ndarray = None
        self.bias: float = 0.0
        self.lr = learningRate
        self.epochs = epochs

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Método `forward`, que implemente la combinación lineal
        de los pesos y sesgo con la entrada seguida de la función logística.

        Args:
            - X (np.ndarray): Input data of shape (numSamples, numFeatures).

        Returns:
            - np.ndarray: Predicted probabilities of shape (numSamples,).
        """
        output = X @ self.weights + self.bias
        activations = 1 / (1 + np.exp(-output))
        return activations

    def backward(self, X: np.ndarray, yPred: np.ndarray, yTrue: np.ndarray) -> None:
        """
        Método `backward` que, dada la entrada, la salida obtenida y
        la salida deseada, modifique los parámetros del modelo.

        Args:
            - X (np.ndarray): Input data of shape (numSamples, numFeatures).
            - yPred (np.ndarray): Predicted probabilities of shape (numSamples,).
            - yTrue (np.ndarray): True class labels of shape (numSamples,).

        Returns:
            - None
        """
        numSamples = X.shape[0]
        error = yPred - yTrue

        # Compute gradients
        dW = (1 / numSamples) * (X.T @ error)
        dB = (1 / numSamples) * np.sum(error)

        # Update weights and bias
        self.weights -= self.lr * dW
        self.bias -= self.lr * dB

    def compute_loss(self, yPred: np.ndarray, yTrue: np.ndarray) -> float:
        """
        Método `compute_loss`, que implemente la función de entropía cruzada binaria.

        Args:
            - yPred (np.ndarray): Predicted probabilities of shape (numSamples,).
            - yTrue (np.ndarray): True class labels of shape (numSamples,).

        Returns:
            - float: Computed binary cross-entropy loss.
        """
        positiveLoss = yTrue * np.log(yPred)
        negativeLoss = (1 - yTrue) * np.log(1 - yPred)

        return -np.mean(positiveLoss + negativeLoss)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Método `fit`, que recibe los datos de entrenamiento y
        optimiza el modelo mediante descenso de gradiente.

        Args:
            - X (np.ndarray): Training data of shape (numSamples, numFeatures).
            - y (np.ndarray): True class labels of shape (numSamples,).

        Returns:
            - None
        """
        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        self.losses = []

        for _ in range(self.epochs):
            # Get predictions
            yPred = self.forward(X)

            # Gradient descent step
            self.backward(X, yPred, y)

            # Compute and store loss
            self.losses.append(self.compute_loss(yPred, y))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Método `predict`, que usa `forward` y obtiene la salida final de inferencia en `{0, 1}`.

        Args:
            - X (np.ndarray): Input data of shape (numSamples, numFeatures).

        Returns:
            - np.ndarray: Predicted class labels of shape (numSamples,).
        """
        return (self.forward(X) >= 0.5).astype(int)


def main() -> None:
    """
    3: Implementa una función principal que realice todos
    los pasos necesarios para entrenar y evaluar un modelo de
    regresión logística que usa agregados de token embeddings como características.

    NOTA: no es necesario almacenar el modelo de regresión.
    """
    sentimentFile = os.path.join(currentDirectory, "sentiment_analysis.tsv")

    # For the final graph
    losses = {}

    for mode in ["skipgram", "cbow"]:

        features = []
        labels = []
        embeddings = loadEmbeddings(
            filePath=os.path.join(
                os.path.dirname(currentDirectory), f"{mode}_embeddings.txt"
            )
        )

        with open(sentimentFile, "r", encoding="utf-8") as f:
            for line in f:
                text, label = line.strip().split("\t")
                embedding = obtainEmbeddings(text, embeddings)
                features.append(embedding)
                labels.append(int(label))

        features = np.array(features)
        labels = np.array(labels)

        # Split data into training and testing sets
        splitIndex = int(0.8 * len(features))
        XTrain, XTest = features[:splitIndex], features[splitIndex:]
        yTrain, yTest = labels[:splitIndex], labels[splitIndex:]

        model = LogisticRegression()
        model.fit(XTrain, yTrain)
        yPred = model.predict(XTest)
        accuracy = np.mean(yPred == yTest)
        print(f"{mode.capitalize()} Accuracy: {accuracy:.4f}")

        losses[mode] = model.losses

    # Plotting the losses
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Sentiment Analysis")
    for mode in losses:
        plt.plot(losses[mode], label=f"{mode.capitalize()}")
    plt.legend()
    plt.savefig(os.path.join(currentDirectory, "loss.png"))


if __name__ == "__main__":
    main()
