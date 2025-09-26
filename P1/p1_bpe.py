from typing import Dict, Iterable, List, Optional, Tuple
from collections import Counter

import argparse
import pickle


class ByteLevelBPE:
    """
    Implementación básica de BPE a nivel de bytes.
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

    @staticmethod
    def _count_pairs(
        lines_tokens: List[List[Tuple[int, ...]]],
    ) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        """
        Obtiene las frecuencias de pares de tokens adyacentes en todas las líneas
        """
        return Counter(
            (line[i], line[i + 1])
            for line in lines_tokens
            for i in range(len(line) - 1)
        )

    @staticmethod
    def _merge_in_line(
        line: List[Tuple[int, ...]], pair: Tuple[Tuple[int, ...], Tuple[int, ...]]
    ) -> List[Tuple[int, ...]]:
        """
        Fusiona todas ocurrencias del par `pair` en una línea (sin solapamiento)

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
                newLine.append(pair[0] + pair[1])
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
    ):
        """
        Aprende las fusiones del BPE y construye los vocabularios.
        """
        # TODO

    def encode(self, text: str) -> List[int]:
        """
        Convierte el texto de entrada en una lista de token IDs.
        """
        return []  # TODO

    def decode(self, ids: List[int]) -> str:
        """
        Convierte una lista de token IDs en texto.
        """
        return ""  # TODO

    def tokenize(self, text: str) -> List[str]:
        """
        Tokeniza un texto.
        """
        return [""]  # TODO


if __name__ == "__main__":
    # TODO
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

    exit(1)
