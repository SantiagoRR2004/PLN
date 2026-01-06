from collections.abc import Iterable
import numpy as np


def viterbi(
    sequence: Iterable[int], start: np.ndarray, A: np.ndarray, B: np.ndarray
) -> None:
    """
    Viterbi algorithm implementation for a Hidden Markov Model.

    Args:
        - sequence (Iterable[int]): Observed sequence of emissions.
        - start (np.ndarray): Initial state probabilities.
        - A (np.ndarray): State transition probability matrix.
            The rows represent the current state, and the columns represent the next state.
        - B (np.ndarray): Emission probability matrix.
            The rows represent the states, and the columns represent the emissions.

    Returns:
        - None: Prints the most probable states.
    """
    # Check dimensions
    assert (
        max(sequence) < B.shape[1]
    ), "Emission indices exceed emission matrix dimensions."
    assert A.shape[0] == A.shape[1], "Transition matrix A must be square."
    assert (
        A.shape[0] == B.shape[0]
    ), "Transition matrix A and emission matrix B must have compatible dimensions."
    assert (
        start.shape[0] == A.shape[0]
    ), "Start probabilities must match number of states."
    assert len(sequence) > 0, "The observed sequence must not be empty."

    # In the first iteration, there are no transitions
    start = (start * B[:, sequence[0]]).ravel()

    for i in range(1, len(sequence)):
        maxIndex1, maxValue1 = max(enumerate(start * A[:, 0]), key=lambda x: x[1])
        maxIndex2, maxValue2 = max(enumerate(start * A[:, 1]), key=lambda x: x[1])
        maxIndex3, maxValue3 = max(enumerate(start * A[:, 2]), key=lambda x: x[1])

        print(maxIndex1, maxIndex2, maxIndex3)
        start = (
            np.array([maxValue1, maxValue2, maxValue3]) * B[:, sequence[i]]
        ).ravel()


if __name__ == "__main__":

    start = np.array([0.25, 0.5, 0.25])

    sequence = [0, 0, 0, 0, 0]

    A = np.array(
        [
            [0.25, 0.25, 0.5],
            [0, 0.25, 0.75],
            [0.5, 0.5, 0],
        ]
    )

    B = np.array([[0.5, 0.25, 0.75], [0.5, 0.75, 0.25]]).T

    # Run the Viterbi algorithm
    viterbi(sequence, start, A, B)
