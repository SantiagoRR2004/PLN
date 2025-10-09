import math
from collections import Counter


class UnigramLMTokenizer:
    def __init__(self):
        self.token_probs = {}

    def train(self, corpus, vocab_size=50, max_subword_length=10):
        # Step 1: Build initial large vocabulary of substrings
        vocab = set()
        for sentence in corpus:
            text = sentence.strip()
            for i in range(len(text)):
                for j in range(i + 1, min(len(text), i + max_subword_length) + 1):
                    vocab.add(text[i:j])

        self.token_probs = {tok: 1.0 / len(vocab) for tok in vocab}

        # Ideally, you would just use relative frequencies of tokens to get the maximum likelihood estimation.
        # But you can't do this here since there are multiple possible tokenizations with our token vocabulary.
        # Hence you need the "soft counting" (expected counts) of the forward-backward algorithm in an expectation-maximization (EM) loop.
        # The forward array, alpha(i), accumulates the probability mass of all possible tokenizations from the start up to each point i.
        # The backward array, beta(j), accumulates the probability mass of all possible tokenizations from each point j up until the end.
        # These accumulated probabilities can then be used to estimate the contribution of each token, or soft count, which can be normalized into a probability.

        # Step 2: Iterative pruning until target vocab size
        while len(vocab) > vocab_size:
            """
            forward[i] = probabilidad total de todas las tokenizaciones que cubren el prefijo hasta la posición i.
            backward[i] = probabilidad total de todas las tokenizaciones que cubren el sufijo desde i hasta el final.
            Para cada token text[i:j], la contribución esperada (soft count) se estima como: (forward[i] * p(token) * backward[j]) / Z
            donde Z = forward[n] es la probabilidad total de la secuencia.
            """
            counts = Counter()
            total = 0

            for sentence in corpus:
                text = sentence.strip()
                n = len(text)

                # Forward probabilities
                forward = [0.0] * (n + 1)
                forward[0] = 1.0
                for i in range(n):
                    if forward[i] == 0:
                        continue
                    for j in range(i + 1, min(n, i + max_subword_length) + 1):
                        token = text[i:j]
                        if token not in vocab:
                            continue  # skip pruned tokens
                        prob = self.token_probs[token]
                        forward[j] += forward[i] * prob
                # Explanation: The forward array is initialized with forward[0] = 1.0, representing the start of the string.
                # For each position i in the string, if forward[i] is non-zero, it iterates over possible end positions j for tokens starting at i.
                # If the substring text[i:j] is in the vocabulary, it updates forward[j] by adding the product of forward[i] and the token's probability.
                # This way, forward[j] accumulates the total probability of all tokenizations that can lead to position j.

                # Backward pass
                backward = [0.0] * (n + 1)
                backward[n] = 1.0
                for i in range(n - 1, -1, -1):
                    for j in range(i + 1, min(n, i + max_subword_length) + 1):
                        if backward[j] == 0:
                            continue
                        token = text[i:j]
                        if token not in vocab:
                            continue  # skip pruned tokens
                        prob = self.token_probs[token]
                        backward[i] += prob * backward[j]
                # Explanation: The backward array is initialized with backward[n] = 1.0, representing the end of the string.
                # It iterates backwards from position n-1 to 0. For each position i, it checks possible end positions j for tokens starting at i.
                # If backward[j] is non-zero and the substring text[i:j] is in the vocabulary, it updates backward[i] by adding the product of the
                # token's probability and backward[j].
                # This way, backward[i] accumulates the total probability of all tokenizations that can lead to position i.

                Z = forward[n]  # total probability of the string
                if Z == 0:
                    continue
                # Explanation: Z represents the total probability of generating the entire string using the current token probabilities.
                # If Z is zero, it means that the current token probabilities cannot generate the string,
                # so the algorithm skips to the next sentence to avoid division by zero in subsequent calculations.

                # Expected counts
                for i in range(n):
                    for j in range(i + 1, min(n, i + max_subword_length) + 1):
                        token = text[i:j]
                        if token not in self.token_probs:
                            continue  # skip pruned tokens
                        prob = self.token_probs[token]
                        contrib = (forward[i] * prob * backward[j]) / Z
                        counts[token] += contrib
                        total += contrib
                # Explanation: This nested loop calculates the expected counts for each token in the string.
                # For each possible token starting at position i and ending at position j, it checks if the token is in the current vocabulary.
                # If it is, it computes the contribution of that token to the expected count using the formula:
                # (forward[i] * prob * backward[j]) / Z, where prob is the token's probability.
                # This contribution is added to counts[token], and total is updated to reflect the sum of all contributions.

            # Update probabilities
            self.token_probs = {
                t: counts[t] / total if total > 0 else 1e-8 for t in vocab
            }

            # Compute the expected loss contribution per token, balancing frequency with information content
            contrib = {}
            for t in vocab:
                contrib[t] = counts[t] * -math.log(
                    self.token_probs.get(t, 1e-8)
                )  # less probable = more value
            # Rank tokens by their contribution
            # Why multiply by -math.log(prob)? Balance between frequency and information content.
            # stopwords have high count but low information content (low -log(prob))

            ranked = sorted(
                list(vocab), key=lambda t: contrib.get(t, math.inf)
            )  # Use of math.inf in case there is a token no evaluated. For now, skip it.
            # Explanation: This line sorts the vocabulary based on the contribution values computed earlier.
            # Tokens with lower contribution values (higher information content) will come first.

            # Remove 20% least useful tokens
            remove_count = max(1, len(ranked) // 5)
            for t in ranked[:remove_count]:
                vocab.remove(t)
                self.token_probs.pop(t, None)
            # Explanation: This block removes the least useful tokens from the vocabulary.
            # It calculates the number of tokens to remove (20% of the current vocabulary size, but at least 1),
            # and then iterates over the first remove_count tokens in the ranked list (the least useful ones),
            # removing them from both the vocabulary set and the token_probs dictionary.

    def tokenize(self, text):
        n = len(text)
        best_score = [math.inf] * (n + 1)  # -log(inf) = -infinito
        best_edge = [None] * (n + 1)
        best_score[0] = 0  # seed to start -Log(0) = infinito

        for i in range(n):
            if best_score[i] == math.inf:
                continue
            for j in range(i + 1, n + 1):
                word = text[i:j]
                if word in self.token_probs:
                    prob = -math.log(self.token_probs[word] + 1e-8)
                    score = best_score[i] + prob
                    if score < best_score[j]:
                        best_score[j] = score
                        best_edge[j] = (i, word)

        # Backtrack
        tokens = []
        idx = n
        while idx > 0:
            i, word = best_edge[idx]
            tokens.append(word)
            idx = i
        tokens.reverse()
        return tokens


# ------------------ DEMO ------------------
if __name__ == "__main__":
    corpus = [
        "one cat sat on the mat",
        "the cat sat on the log",
        "the dog sat on the mat",
        "the dog sat on the log",
    ]

    tokenizer = UnigramLMTokenizer()
    tokenizer.train(corpus, vocab_size=20)

    print("Probs:", tokenizer.token_probs)

    test_sentence = "the cat on the mat"
    print("Tokenized:", tokenizer.tokenize(test_sentence))
