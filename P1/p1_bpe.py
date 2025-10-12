from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter
import argparse
import pickle
import tqdm


class ByteLevelBPE:
    """
    Implementación básica de BPE a nivel de bytes.
    # https://en.wikipedia.org/wiki/Byte-pair_encoding
    - Los tokens iniciales son bytes individuales (0..255).
    - Durante el entrenamiento se obtienen los pares de tokens adyacentes más frecuentes y se fusionan, todo ello de forma iterativa.
    - La codificación (`encode`) aplica las fusiones aprendidas en orden.
    """

    def __init__(self):
        self.merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self.vocab: Dict[Tuple[int, ...], int] = {(i,): i for i in range(256)}
        self.id2bytes: List[Tuple[int, ...]] = [(i,) for i in range(256)]

    @staticmethod
    def _to_byte_tokens(s: str) -> List[Tuple[int, ...]]:
        """
        Devuelve una lista de tokens como tuplas de bytes individuales
        """
        b = s.encode("utf-8")
        return [(x,) for x in b]

    def _count_pairs(
        self,
        lines_tokens: List[List[Tuple[int, ...]]],
    ) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        """
        Obtiene las frecuencias de pares de tokens adyacentes en todas las líneas

        Calcular los pares en cada iteración tarda alrededor de 1 minuto y 10
        segundos para obtener un vocabulario de 1000 tokens.

        Con el nuevo método de ir actualizando en cada fusión, tarda
        alrededor de 30 segundos para 1000 tokens.
        """
        if hasattr(self, "pairs"):
            return self.pairs
        else:
            self.pairs = dict(
                Counter(
                    (line[i], line[i + 1])
                    for line in lines_tokens
                    for i in range(len(line) - 1)
                )
            )
        return self.pairs

    def _merge_in_line(
        self, line: List[Tuple[int, ...]], pair: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        """
        Fusiona todas ocurrencias del par `pair` en una línea (sin solapamiento)

        Vamos actualizando también el diccionario de pares.

        Args:
            - line: list of tokens
            - pair: pair of tokens to merge

        Returns:
            - new line with the pair merged
        """
        newLine = []

        i = 0
        # Iterate finding pairs
        while i < len(line):
            if i < len(line) - 1 and (line[i], line[i + 1]) == pair:
                merged_token = pair[0] + pair[1]
                newLine.append(merged_token)
                # if we merged (A, B) -> (AB), we need to reduce the count of (A, B)
                # because we removed one occurrence of it
                if pair in self.pairs:
                    self.pairs[pair] -= 1
                    if self.pairs[pair] <= 0:
                        del self.pairs[pair]
                # Updating pairs Counter around the merged pair
                if i > 0:
                    # Reduce count of the pair to the left (X, A)
                    old_left_pair = (line[i - 1], pair[0])
                    if old_left_pair in self.pairs:
                        self.pairs[old_left_pair] -= 1
                        if self.pairs[old_left_pair] <= 0:
                            del self.pairs[old_left_pair]
                    # Add new pair (X, AB)
                    self.pairs[(line[i - 1], merged_token)] = (
                        self.pairs.get((line[i - 1], merged_token), 0) + 1
                    )
                if i < len(line) - 2:
                    # Reduce count of the pair to the right (B, Y)
                    old_right_pair = (pair[1], line[i + 2])
                    if old_right_pair in self.pairs:
                        self.pairs[old_right_pair] -= 1
                        if self.pairs[old_right_pair] <= 0:
                            del self.pairs[old_right_pair]
                    # Pair to the right of the merged pair
                    self.pairs[(merged_token, line[i + 2])] = (
                        self.pairs.get((merged_token, line[i + 2]), 0) + 1
                    )
                i += 2
            else:
                newLine.append(line[i])
                i += 1

        return newLine

    def train(
        self,
        lines: Iterable[str],
        vocab_size: int = 1000,
        max_merges: Optional[int] = None,
    ) -> None:
        """
        Aprende las fusiones del BPE y construye los vocabularios.
        """
        # Convert lines to tokens
        lines_tokens = [self._to_byte_tokens(line) for line in lines]

        maxIterations = min(
            vocab_size - len(self.vocab), max_merges if max_merges else vocab_size
        )

        for _ in tqdm.tqdm(range(maxIterations), desc="Training BPE"):
            # Count pairs
            pairs = self._count_pairs(lines_tokens)

            # Get most common pair
            mostCommon = max(pairs, key=pairs.get, default=None)

            # Add to merges
            self.merges.append(mostCommon)

            # Delete from the pairs dict
            del self.pairs[mostCommon]

            # Create new token
            newToken = mostCommon[0] + mostCommon[1]
            self.vocab[newToken] = len(self.vocab)
            self.id2bytes.append(newToken)

            # Update the line_tokens
            lines_tokens = [
                self._merge_in_line(line, mostCommon) for line in lines_tokens
            ]

    def encode(self, text: str) -> List[int]:
        """
        Convierte el texto de entrada en una lista de token IDs.
        """
        tokens = self._to_byte_tokens(text)

        # Now we apply merges in order
        for merge in self.merges:
            tokens = self._merge_in_line(tokens, merge)

        # Convert tokens to IDs
        return [self.vocab[token] for token in tokens]

    def decode(self, ids: List[int]) -> str:
        """
        Convierte una lista de token IDs en texto.
        """
        intTokens = [self.id2bytes[i] for i in ids]
        return "".join([bytes(b).decode("utf-8") for b in intTokens])

    def tokenize(self, text: str) -> List[Tuple[int, ...]]:
        """
        Tokeniza un texto.
        """
        tokensIDs = self.encode(text)
        return [self.id2bytes[i] for i in tokensIDs]


if __name__ == "__main__":
    # Uso:
    # python p1_bpe.py train <input_train_corpus> <output_model_file>
    # python p1_bpe.py eval <input_model_file> <input_text>
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser train
    parser_train = subparsers.add_parser("train", help="Entrenar modelo")
    parser_train.add_argument(
        "input_train_corpus", help="Archivo de texto de entrenamiento"
    )
    parser_train.add_argument(
        "output_model_file", help="Archivo donde guardar el modelo entrenado"
    )

    # Subparser eval
    parser_eval = subparsers.add_parser("eval", help="Evaluar modelo")
    parser_eval.add_argument("input_model_file", help="Archivo del modelo entrenado")
    parser_eval.add_argument("input_text", help="Texto a codificar y decodificar")

    args = parser.parse_args()

    if args.command == "train":
        input_train_corpus = args.input_train_corpus
        output_model_file = args.output_model_file
        bpe = ByteLevelBPE()
        with open(input_train_corpus, "r", encoding="utf-8") as f:
            lines = f.readlines()
        bpe.train(lines, vocab_size=1000)
        with open(output_model_file, "wb") as f:
            pickle.dump(bpe, f)
    elif args.command == "eval":
        input_model_file = args.input_model_file
        input_text = args.input_text
        with open(input_model_file, "rb") as f:
            bpe = pickle.load(f)

        # Encode the text
        token_ids = bpe.encode(input_text)
        print(f"Original text: {input_text}")
        print()
        print(f"Token IDs: {token_ids}")
        print()

        # Show tokenization
        tokens = bpe.tokenize(input_text)
        print(f"Tokens: {tokens}")
        print()

        readable_tokens = [bytes(token).decode("utf-8") for token in tokens]
        print(f"Tokens (readable): {readable_tokens}")
        print()
        # Decode back to verify
        decoded_text = bpe.decode(token_ids)
        print(f"Decoded text: {decoded_text}")

    exit(0)
