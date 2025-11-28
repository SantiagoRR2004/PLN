from concurrent.futures import ProcessPoolExecutor
import p3_classifier
import numpy as np
import torch
import os


def canUseGPU() -> str:
    """
    Check if a GPU is available and can be used by PyTorch.

    Args:
        - None

    Returns:
        - str: "cuda" if a GPU is available and usable, otherwise "cpu".
    """
    if torch.cuda.is_available():
        try:
            # Create small tensors directly on the GPU
            a = torch.randn((100, 100), device="cuda")
            b = torch.randn((100, 100), device="cuda")

            # Run a computation (matrix multiplication)
            c = torch.matmul(a, b)

            # Force synchronization to trigger any CUDA errors
            torch.cuda.synchronize()

        except Exception:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            torch.set_default_device("cpu")
            torch.cuda.is_available = lambda: False
            torch.cuda.is_available()

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_default_device("cpu")
    torch.cuda.is_available = lambda: False
    torch.cuda.is_available()


def obtainEmbeddingsParallelism(
    texts: list[str],
    embeddings: np.ndarray,
) -> np.ndarray:
    futures = []
    with ProcessPoolExecutor() as executor:
        for text in texts:
            futures.append(
                executor.submit(
                    p3_classifier.obtainEmbeddings,
                    text,
                    embeddings,
                )
            )

    results = [future.result() for future in futures]
    return np.array(results)


if __name__ == "__main__":
    canUseGPU()
