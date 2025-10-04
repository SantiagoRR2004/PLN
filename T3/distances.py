from collections import Counter


def hamming_distance(s1: str, s2: str) -> int:
    if len(s1) != len(s2):
        raise ValueError("Strings must be of equal length")

    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))


def hamming_example_usage():
    print(hamming_distance("karolin", "kathrin"))
    print(hamming_distance("1011101", "1001001"))
    try:
        hamming_distance("this", "those")
    except ValueError as ve:
        print(f"Error: {ve}")


# Wagner-Fischer algorithm
def levenshtein_distance(s1: str, s2: str) -> int:
    len_s1, len_s2 = len(s1), len(s2)

    # Initialize matrix of size (len_s1+1) x (len_s2+1)
    dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]

    # Base cases
    for i in range(len_s1 + 1):
        dp[i][0] = i
    for j in range(len_s2 + 1):
        dp[0][j] = j

    # Fill matrix
    for i in range(1, len_s1 + 1):
        for j in range(1, len_s2 + 1):
            subst_cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + subst_cost,  # Substitution
            )
        # The previous adjacent cells contain the number of operations to make the strings equal
        # For the new cell, you need at most 1 more operation on top of the previous minimal number of operations

    return dp[len_s1][len_s2]


def levenshtein_example_usage():
    print(levenshtein_distance("kitten", "sitting"))
    print(levenshtein_distance("flaw", "lawn"))


def bigrams(s: str):
    """Generate character bigrams from a string."""
    return [s[i : i + 2] for i in range(len(s) - 1)]


def bigram_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0

    bigrams1 = Counter(bigrams(s1))
    bigrams2 = Counter(bigrams(s2))

    intersection = sum((bigrams1 & bigrams2).values())

    return (2.0 * intersection) / (sum(bigrams1.values()) + sum(bigrams2.values()))


def dice_example_usage():
    print(bigram_similarity("gato", "gatito"))
    print(bigram_similarity("context", "contact"))
    print(bigram_similarity("hello", "hello"))


if __name__ == "__main__":
    hamming_example_usage()
    levenshtein_example_usage()
    dice_example_usage()
