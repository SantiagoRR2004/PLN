import matplotlib.pyplot as plt
from P1 import ByteLevelBPE
import numpy as np
import p3_finetune
import pickle
import utils
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


class LearningRateScheduler:
    """
    Adaptive learning rate scheduler based on loss progression.
    """

    def __init__(
        self,
        initialLR: float,
    ) -> None:
        """
        Initialize the learning rate scheduler.

        Args:
            - initialLR (float): Initial learning rate.

        Returns:
            - None
        """
        self.currentLR = initialLR

    def step(self, losses: list) -> float:
        """
        Compute the next learning rate based on loss progression.

        Args:
            - losses (list): List of loss values from training history.

        Returns:
            - float: Updated learning rate.
        """
        if len(losses) == 0:
            return self.currentLR

        # If the last 5 losses are too flat, try to exit stagnation
        if len(losses) >= 5 and max(losses[-5:]) - min(losses[-5:]) < 1e-4:
            self.currentLR += 1
            return self.currentLR

        descendingLosses = self.longestDecreasingSuffix(losses)

        # Need at least 3 points to compute slopes
        if len(descendingLosses) < 3:
            return self.currentLR

        slopes = [
            descendingLosses[i + 1] - descendingLosses[i]
            for i in range(len(descendingLosses) - 1)
        ]

        # If it is too close to a straight line, increase by 10%
        # If it is not, decrease by 10%
        if np.std(slopes) < 1e-4:
            self.currentLR *= 1.1  # Increase by 10%
        else:
            self.currentLR *= 0.9  # Decrease by 10%

        return self.currentLR

    def longestDecreasingSuffix(self, nums: list) -> list:
        """
        Find the longest amount of losses that have been decreasing from the end.

        Args:
            - nums (list): List of loss values.

        Returns:
            - list: The longest decreasing suffix of the input list.
        """
        if len(nums) < 2:
            return nums

        i = len(nums) - 1
        while i > 0 and nums[i] < nums[i - 1]:
            i -= 1

        return nums[i:]


# 2: Implementa la clase LogisticRegression con los siguientes componentes:
class LogisticRegression:

    def __init__(self, learningRate: float = 0.01, epochs: int = 1000) -> None:
        """
        Campos para almacenar los pesos, sesgo y learning rate.

        Args:
            - learningRate (float): The initial learning rate for gradient descent.
            - epochs (int): The number of training epochs.

        Returns:
            - None
        """
        self.weights: np.ndarray = None
        self.bias: float = 0.0
        self.lr = learningRate
        self.epochs = epochs

        self.scheduler = LearningRateScheduler(
            initialLR=learningRate,
        )

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

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        XEval: np.ndarray = None,
        yEval: np.ndarray = None,
    ) -> None:
        """
        Método `fit`, que recibe los datos de entrenamiento y
        optimiza el modelo mediante descenso de gradiente.

        Args:
            - X (np.ndarray): Training data of shape (numSamples, numFeatures).
            - y (np.ndarray): True class labels of shape (numSamples,).
            - XEval (np.ndarray, optional): Evaluation data of shape (numEvalSamples, numFeatures).
            - yEval (np.ndarray, optional): True class labels for evaluation data of shape (numEvalSamples,).

        Returns:
            - None
        """
        # Initialize weights
        self.weights = np.zeros(X.shape[1])

        self.losses = []
        self.learningRates = []  # Track learning rate changes

        # For evaluation during training
        evaluation = XEval is not None and yEval is not None
        if evaluation:
            self.evalLosses = []
            self.accuracies = []
            bestAccuracy = 0.0
            bestWeights = self.weights.copy()
            bestBias = self.bias

        for _ in range(self.epochs):
            # Update learning rate based on loss progression
            self.lr = self.scheduler.step(self.losses)

            self.learningRates.append(self.lr)

            # Get predictions
            yPred = self.forward(X)

            # Gradient descent step
            self.backward(X, yPred, y)

            # Compute and store loss
            self.losses.append(self.compute_loss(yPred, y))

            # Evaluation
            if evaluation:
                yEvalPred = self.forward(XEval)

                # Compute loss on evaluation set
                evalLoss = self.compute_loss(yEvalPred, yEval)
                self.evalLosses.append(evalLoss)

                # Compute accuracy
                yEvalLabels = (yEvalPred >= 0.5).astype(int)
                accuracy = np.mean(yEvalLabels == yEval)
                self.accuracies.append(accuracy)

                # Save best model
                if accuracy > bestAccuracy:
                    bestAccuracy = accuracy
                    bestWeights = self.weights.copy()
                    bestBias = self.bias

        # Keep best model after evaluation
        if evaluation:
            self.weights = bestWeights
            self.bias = bestBias

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

    modes = ["skipgram", "cbow"]

    # Load embeddings
    embeddings = {}
    for mode in modes:
        embeddings[mode] = loadEmbeddings(
            filePath=os.path.join(
                os.path.dirname(currentDirectory), f"{mode}_embeddings.txt"
            )
        )

    datasets = {}

    # The basic dataset
    with open(sentimentFile, "r", encoding="utf-8") as f:
        lines = map(str.strip, f)
        texts, labels = zip(*[line.split("\t") for line in lines])
        labels = list(map(int, labels))
        labels = np.array(labels)

    for mode in modes:
        features = utils.obtainEmbeddingsParallelism(
            texts,
            embeddings[mode],
            name="Basic " + mode.capitalize(),
            filename=f"{mode}_basic.npy",
        )

        # Split data into training and testing sets
        # There needs to be 50 samples in the training set
        splitIndex = 50
        XTrain, XTest = features[-splitIndex:], features[:-splitIndex]
        yTrain, yTest = labels[-splitIndex:], labels[:-splitIndex]

        assert len(XTrain) == splitIndex

        datasets["Basic " + mode] = {
            "XTrain": XTrain,
            "yTrain": yTrain,
            "XTest": XTest,
            "yTest": yTest,
        }

    # The IMDb dataset
    IMDbDataset = p3_finetune.obtainDataset()

    # Only the train
    for mode in modes:
        # Tokenize and embed XTrain
        XTrain = utils.obtainEmbeddingsParallelism(
            IMDbDataset["train"]["text"],
            embeddings[mode],
            name="IMDb Train " + mode.capitalize(),
            filename=f"{mode}_imdb_train.npy",
        )
        yTrain = np.array(IMDbDataset["train"]["label"])

        XTest = np.concatenate(
            (datasets["Basic " + mode]["XTest"], datasets["Basic " + mode]["XTrain"])
        )
        yTest = np.concatenate(
            (datasets["Basic " + mode]["yTest"], datasets["Basic " + mode]["yTrain"])
        )

        datasets["IMDb Train " + mode] = {
            "XTrain": XTrain,
            "yTrain": yTrain,
            "XTest": XTest,
            "yTest": yTest,
        }

    # The full IMDb dataset
    for mode in modes:
        # Tokenize and embed XTest
        XTestIMDb = utils.obtainEmbeddingsParallelism(
            IMDbDataset["test"]["text"],
            embeddings[mode],
            name="IMDb Test " + mode.capitalize(),
            filename=f"{mode}_imdb_test.npy",
        )
        yTestIMDb = np.array(IMDbDataset["test"]["label"])

        XTrain = np.concatenate(
            (
                datasets["IMDb Train " + mode]["XTrain"],
                XTestIMDb,
            )
        )
        yTrain = np.concatenate(
            (
                datasets["IMDb Train " + mode]["yTrain"],
                yTestIMDb,
            )
        )

        XTest = datasets["IMDb Train " + mode]["XTest"]
        yTest = datasets["IMDb Train " + mode]["yTest"]

        datasets["IMDb Whole " + mode] = {
            "XTrain": XTrain,
            "yTrain": yTrain,
            "XTest": XTest,
            "yTest": yTest,
        }

        datasets["IMDb " + mode] = {
            "XTrain": datasets["IMDb Train " + mode]["XTrain"],
            "yTrain": datasets["IMDb Train " + mode]["yTrain"],
            "XTest": XTestIMDb,
            "yTest": yTestIMDb,
        }

    # For the final graphs
    losses = {}
    lossesEval = {}
    accuracies = {}
    learningRates = {}

    for datasetName in datasets:
        model = LogisticRegression()
        model.fit(
            datasets[datasetName]["XTrain"],
            datasets[datasetName]["yTrain"],
            XEval=datasets[datasetName]["XTest"],
            yEval=datasets[datasetName]["yTest"],
        )
        yPred = model.predict(datasets[datasetName]["XTest"])
        accuracy = np.mean(yPred == datasets[datasetName]["yTest"])
        print(f"{datasetName} Accuracy: {accuracy:.4f}")

        losses[datasetName] = model.losses
        lossesEval[datasetName] = model.evalLosses
        accuracies[datasetName] = model.accuracies
        learningRates[datasetName] = model.learningRates

    # Plotting the losses
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Sentiment Analysis")
    for mode in losses:
        plt.plot(losses[mode], label=f"{mode.title()}")
    plt.legend()
    plt.savefig(os.path.join(currentDirectory, "loss.png"))
    plt.clf()

    # Plotting the evaluation losses
    plt.xlabel("Epoch")
    plt.ylabel("Evaluation Loss")
    plt.title("Evaluation Loss Sentiment Analysis")
    for mode in lossesEval:
        plt.plot(lossesEval[mode], label=f"{mode.title()}")
    plt.legend()
    plt.savefig(os.path.join(currentDirectory, "lossEval.png"))
    plt.clf()

    # Plotting the accuracies
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Evaluation Accuracy Sentiment Analysis")
    for mode in accuracies:
        plt.plot(accuracies[mode], label=f"{mode.title()}")
    plt.legend()
    plt.savefig(os.path.join(currentDirectory, "accuracy.png"))
    plt.clf()

    # Plotting the learning rates
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Adaptation")
    for mode in learningRates:
        plt.plot(learningRates[mode], label=f"{mode.title()}")
    plt.legend()
    plt.savefig(os.path.join(currentDirectory, "learningRate.png"))
    plt.clf()


if __name__ == "__main__":
    main()
