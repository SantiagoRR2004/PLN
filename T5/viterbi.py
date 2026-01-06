import numpy as np


def viterbi(start: np.ndarray, A: np.ndarray, B: np.ndarray) -> None:
    """
    Viterbi algorithm implementation for a Hidden Markov Model.

    Args:
        - start (np.ndarray): Initial state probabilities.
        - A (np.ndarray): State transition probability matrix.
        - B (np.ndarray): Emission probability matrix.

    Returns:
        - None: Prints the most probable states.
    """
    for _ in range(5):
        maxIndex1, maxValue1 = max(enumerate(start * A[:, 0]), key=lambda x: x[1])
        maxIndex2, maxValue2 = max(enumerate(start * A[:, 1]), key=lambda x: x[1])
        maxIndex3, maxValue3 = max(enumerate(start * A[:, 2]), key=lambda x: x[1])

        print(maxIndex1, maxIndex2, maxIndex3)
        start = np.array([maxValue1, maxValue2, maxValue3]) * B


if __name__ == "__main__":

    start = np.array([0.25, 0.5, 0.25])

    A = np.array(
        [
            [0.25, 0.25, 0.5],
            [0, 0.25, 0.75],
            [0.5, 0.5, 0],
        ]
    )

    B = np.array([0.5, 0.25, 0.75])

    # Run the Viterbi algorithm
    viterbi(start, A, B)
