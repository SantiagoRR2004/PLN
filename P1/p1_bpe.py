from typing import Dict, Iterable, List, Optional, Tuple


class ByteLevelBPE:
    """
    Implementación básica de BPE a nivel de bytes.
    - Los tokens iniciales son bytes individuales (0..255).
    - Durante el entrenamiento se obtienen los pares de tokens adyacentes más frecuentes y se fusionan, todo ello de forma iterativa.
    - La codificación (`encode`) aplica las fusiones aprendidas en orden.
    """

    def __init__(self):
        self.merges: List[Tuple[Tuple[int, ...], Tuple[int, ...]]] = []
        self.vocab: Dict[Tuple[int, ...], int] = {}
        self.id2bytes: List[Tuple[int, ...]] = []

    @staticmethod
    def _to_byte_tokens(s: str) -> List[Tuple[int, ...]]:
        """
        Devuelve una lista de tokens como tuplas de bytes individuales
        """
        b = s.encode("utf-8")
        return [(x,) for x in b]

    @staticmethod
    def _count_pairs(lines_tokens: List[List[Tuple[int, ...]]]) -> Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], int]:
        """
        Obtiene las frecuencias de pares de tokens adyacentes en todas las líneas
        """
        return {}  # TODO

    @staticmethod
    def _merge_in_line(line: List[Tuple[int, ...]],
                       pair: Tuple[Tuple[int, ...], Tuple[int, ...]]) -> List[Tuple[int, ...]]:
        """
        Fusiona todas ocurrencias del par `pair` en una línea (sin solapamiento)
        """
        return []  # TODO

    def train(self, lines: Iterable[str], vocab_size: int = 1000, max_merges: Optional[int] = None):
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
    exit(1)
