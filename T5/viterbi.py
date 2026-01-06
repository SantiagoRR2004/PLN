from collections.abc import Iterable
import matplotlib.pyplot as plt
import numpy as np
import random
import os


def viterbi(
    sequence: Iterable[int], start: np.ndarray, A: np.ndarray, B: np.ndarray
) -> list[int]:
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

    mostProbable = []

    for i in range(1, len(sequence)):

        # Multiply the probabilities with the transition matrix
        mult = start[:, None] * A

        # Get the maximum probabilities and their indices
        maxIndeces = np.argmax(mult, axis=0)
        start = (mult[maxIndeces, range(A.shape[1])] * B[:, sequence[i]]).ravel()

        mostProbable.append(maxIndeces)

    finalState = np.argmax(start)
    optimalPath = [finalState]

    for states in reversed(mostProbable):
        optimalPath.append(states[optimalPath[-1]])

    return [int(x + 1) for x in optimalPath[::-1]]


def randomViterbi() -> dict:
    """
    Generates random parameters for a Hidden Markov Model and an observed sequence.

    Args:
        - None

    Returns:
        - dict: A dictionary containing the observed sequence, start probabilities,
                transition matrix, and emission matrix.
    """
    nStates = random.randint(2, 10)
    nEvents = random.randint(2, 10)
    seqLength = random.randint(5, 100)

    start = np.random.dirichlet(np.ones(nStates))
    A = np.array([np.random.dirichlet(np.ones(nStates)) for _ in range(nStates)])
    B = np.array([np.random.dirichlet(np.ones(nStates)) for _ in range(nEvents)]).T
    sequence = [random.randint(0, nEvents - 1) for _ in range(seqLength)]

    return {
        "sequence": sequence,
        "start": start,
        "A": A,
        "B": B,
    }


def testRepeatedSequences() -> None:
    """
    Test the Viterbi algorithm while always having the same emission.

    Args:
        - None

    Returns:
        - None
    """
    lengths = {}

    for _ in range(1000):
        example = randomViterbi()
        example["sequence"] = [0 for _ in range(len(example["sequence"]))]

        path = viterbi(**example)[1:]

        n = len(path)
        for size in range(1, n):
            pattern = path[:size]
            matches = True
            for i in range(n):
                if path[i] != pattern[i % size]:
                    matches = False
                    break
            if matches:
                n = size
                break

        lengths[n] = lengths.get(n, 0) + 1

    # Sort lengths by key
    lengths = dict(sorted(lengths.items()))

    # Plot the results as a plot
    plt.plot(lengths.keys(), lengths.values())
    plt.xlabel("Length of Repeated Pattern")
    plt.ylabel("Frequency")
    plt.title("Frequency of Repeated Pattern Lengths in Viterbi Paths")
    plt.savefig(os.path.join(os.path.dirname(__file__), "viterbi.png"))


if __name__ == "__main__":

    start = np.array([0.25, 0.5, 0.25])

    sequence = [0, 0, 0, 0, 1, 1, 0, 1]

    A = np.array(
        [
            [0.25, 0.25, 0.5],
            [0, 0.25, 0.75],
            [0.5, 0.5, 0],
        ]
    )

    B = np.array([[0.5, 0.25, 0.75], [0.5, 0.75, 0.25]]).T

    # Run the Viterbi algorithm
    print(viterbi(sequence, start, A, B))

    testRepeatedSequences()
