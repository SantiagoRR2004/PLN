import matplotlib.pyplot as plt
from typing import List
from P1 import ByteLevelBPE
import numpy as np
import pickle
import tqdm
import os

# Una embedding es una codificación que hace la red neuronal para una entrada dada.
current_directory = os.path.dirname(os.path.abspath(__file__))


def sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ez = np.exp(x[neg])
    out[neg] = ez / (1.0 + ez)
    return out


# TODO 1: Implementa un método de entrenamiento simple, esto es, con learning rate (LR) constante y ventana estática.
class Trainer:
    def _neg_sampling_fix(self):
        # TODO 2: Inicializa `self.neg_prob`, que será usado como distribución de probabilidad
        # a la hora de hacer el muestreo negativo, de modo que contenga las frecuencias
        # relativas de cada token del vocabulario elevadas a 2/3.
        pass

    def _subsample_data(self):
        # TODO 3: Reduce la ocurrencia de los tokens más frecuentes usando la siguiente fórmula:
        # `p_keep = (np.sqrt(t / f) + t / f) if f > 0 else 1.0`
        # donde `t = 1e-5` y `f` es la frecuencia relativa del token.
        pass

    def __init__(
        self,
        corpus_fpath: str,
        rng: np.random.Generator,
        embedding_dim: int,
        window_size: int,
        epochs: int,
        lr: float,
        lr_min_factor: float,
        neg_samples: int,
    ):
        self.corpus_fpath = corpus_fpath
        self.rng = rng
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.epochs = epochs
        self.lr = lr
        self.lr_min_factor = lr_min_factor
        self.neg_samples = neg_samples

        self.setup_plot()

        # 1.1: Carga el corpus y tokenízalo usando el tokenizador BPE de la práctica anterior.
        # El corpus debería quedar codificado como una secuencia de ids de tokens.
        with open("bpe_model.pkl", "rb") as f:
            self.bpe: ByteLevelBPE = pickle.load(f)

        with open(corpus_fpath, "r", encoding="utf-8") as f:
            corpus = f.read()

        self.corpus = [self.bpe.encode(c) for c in corpus.split("\n\n")]

        # Aplica ajustes para evitar la sobreponderancia de tokens frecuentes
        self._neg_sampling_fix()
        self._subsample_data()

    def setup_plot(self) -> None:
        """
        Show the live plot for loss visualization.

        Args:
            - None

        Returns:
            - None
        """
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Loss during Training")
        self.fig.show()
        plt.pause(0.1)

    def update_plot(self) -> None:
        """
        Update the live plot with the latest loss values.

        Args:
            - None

        Returns:
            - None
        """
        self.ax.clear()
        # Plot the losses
        self.ax.plot(
            range(1, len(self.losses) + 1), self.losses, label="Loss", color="red"
        )
        self.ax.set_xlabel("Epoch")
        self.ax.set_ylabel("Loss")
        self.ax.set_title("Loss during Training")
        self.ax.legend()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def sample_neg(self, forbidden) -> List[int]:
        # 1.2: Obtén una muestra negativa de tokens, evitando seleccionar aquellos en
        # `forbidden`, que serán los que estén dentro de la ventana actual.
        samples = []
        while len(samples) < self.neg_samples:
            token = self.rng.choice(list(self.bpe.vocab.values()))
            if token not in forbidden and token not in samples:
                samples.append(token)
        return samples

    def train(self):
        # 1.3: Inicializa dos matrices de `self.vocab_size` x `self.embedding_dim` para tokens
        # centrales y contexto.
        centralEmbeddings = self.rng.normal(
            loc=0.0,
            scale=0.1,
            size=(len(self.bpe.vocab), self.embedding_dim),
        ).astype(np.float32)
        contextEmbeddings = self.rng.normal(
            loc=0.0,
            scale=0.1,
            size=(len(self.bpe.vocab), self.embedding_dim),
        ).astype(np.float32)

        barTrain = tqdm.tqdm(
            total=self.epochs * len(self.corpus), desc="Training", position=0
        )

        self.losses = []

        # 1.4: Para cada `epoch` y para cada token en el corpus
        for epoch in range(self.epochs):
            loss = 0.0
            barEpoch = tqdm.tqdm(
                total=len(self.corpus),
                desc=f"Epoch {epoch+1}",
                position=1,
                leave=epoch >= self.epochs - 1,
            )

            for sentence in self.corpus:

                for i, centralToken in enumerate(sentence):
                    """
                    Para cada token en el contexto del token actual,
                    es decir, para cada token dentro de los `self.window_size`
                    tokens a la derecha e izquieda del actual, sin contar este
                    """
                    window = (
                        sentence[max(i - self.window_size, 0) : i]
                        + sentence[i + 1 : min(i + self.window_size + 1, len(sentence))]
                    )

                    # Calcular el producto escalar entre las embeddings del token central y token de contexto.
                    scalar = np.dot(
                        centralEmbeddings[centralToken],
                        contextEmbeddings[window].T,
                    )

                    # Pasar el resultado por la función `sigmoid`, obteniendo `pos_score`.
                    pos_score = sigmoid(scalar)

                    """
                    Muestra positiva: actualizar las embeddings del token central y
                    token contexto usando el LR, `(1 - pos_score)` y la embedding
                    (¡original!) del otro token.
                    """
                    centralEmbeddings[centralToken] += self.lr * np.sum(
                        (1 - pos_score)[:, np.newaxis] * contextEmbeddings[window],
                        axis=0,
                    )
                    contextEmbeddings[window] += (
                        self.lr
                        * (1 - pos_score)[:, np.newaxis]
                        * centralEmbeddings[centralToken]
                    )

                    # Muestras negativas
                    # Obtener muestras negativas para el token central
                    negativeSamples = self.sample_neg(forbidden=window + [centralToken])
                    # Para cada una, realizar un proceso similar al de la muestra positiva
                    # Ahora `pos_score` es `neg_score`
                    neg_score = sigmoid(
                        np.dot(
                            centralEmbeddings[centralToken],
                            contextEmbeddings[negativeSamples].T,
                        )
                    )

                    # Se usa `-neg_score` para actualizar las embeddings.
                    centralEmbeddings[centralToken] += self.lr * np.sum(
                        (-neg_score)[:, np.newaxis]
                        * contextEmbeddings[negativeSamples],
                        axis=0,
                    )
                    contextEmbeddings[negativeSamples] += (
                        self.lr
                        * -neg_score[:, np.newaxis]
                        * centralEmbeddings[centralToken]
                    )

                    """
                    4: Usa una ventana de contexto dinámica,
                    con tamaños que varíen aleatoriamente dentro
                    del rango de la ventana estática original.
                    """
                    self.window_size = self.rng.integers(1, self.window_size + 1)

                    # Criterion: entropy loss
                    loss += -np.sum(np.log(pos_score + 1e-10)) - np.sum(
                        np.log(1 - neg_score + 1e-10)
                    )

                barEpoch.update(1)
                barTrain.update(1)

            # 5: Haz que el LR disminuya progresivamente durante el entrenamiento (linear decay).
            self.lr += (self.lr_min_factor - self.lr) / (self.epochs - epoch)
            self.losses.append(loss / sum(len(samples) for samples in self.corpus))
            self.update_plot()

        plt.ioff()
        plt.savefig(os.path.join(current_directory, "loss.png"))

        # 1.5: Devuelve las dos matrices de embeddings.
        return centralEmbeddings, contextEmbeddings


def dump_embeddings(
    E: np.ndarray,
    bpe: ByteLevelBPE,
    file_path: str = os.path.join(current_directory, "skipgram_embeddings.txt"),
) -> None:
    """
    1.6: Escribe las embeddings en un fichero de texto donde,
    en la primera fila, aparezca el tamaño del vocabulario y el
    número de dimensiones de las embeddings y, en el resto de filas,
    cada token seguido de su correspondiente embedding, separando
    cada elemento con espacios simples.
    Ojo, los tokens pueden contener espacios.

    Args:
        - E (np.ndarray): Matriz de embeddings.
        - bpe (ByteLevelBPE): Modelo BPE usado para tokenizar.
        - file_path (str): Ruta del fichero donde se guardarán las embeddings.

    Returns:
        - None
    """
    with open(file_path, "w", encoding="utf-8") as f:
        vocab_size, embedding_dim = E.shape
        f.write(f"{vocab_size} {embedding_dim}\n")
        for token, token_id in bpe.vocab.items():
            # Map takes each number of the vector and transforms it in a str
            # Join puts a space between each element of the list
            embedding = " ".join(map(str, E[token_id]))
            readable_token = bpe.decode([token_id])
            f.write(f"{token} {token_id} {readable_token} {embedding}\n")


def print_similar_embeddings(bpe: ByteLevelBPE, E: np.ndarray, top_k: int = 10):

    # Add new axis to get the matrix of differences n x n x dimensions
    # Then compute the norm along the last axis
    diff = E[:, np.newaxis, :] - E[np.newaxis, :, :]

    dists = np.linalg.norm(diff, axis=-1)

    # Avoid zero distances (self-distances)
    np.fill_diagonal(dists, np.inf)

    # Get the indices of the top_k closest embeddings
    flat_indices = np.argsort(dists, axis=None)[:top_k]
    pairs = np.array(np.unravel_index(flat_indices, dists.shape)).T

    for i, j in pairs:
        print(
            f"Token 1: {bpe.decode([i])}, id: {i} | Token 2: {bpe.decode([j])}, id: {j}"
        )


def main():
    trainer = Trainer(
        corpus_fpath="P0/tiny_cc_news.txt",
        rng=np.random.default_rng(42),
        embedding_dim=100,
        window_size=5,
        epochs=5,
        lr=0.05,
        lr_min_factor=0.0001,
        neg_samples=5,
    )

    T, C = trainer.train()
    E = (T + C) / 2.0  # Matriz final de embeddings
    dump_embeddings(
        E,
        trainer.bpe,
    )

    print_similar_embeddings(trainer.bpe, E, top_k=10)


if __name__ == "__main__":
    main()
