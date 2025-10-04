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

                Z = forward[n]  # total probability of the string
                if Z == 0:
                    continue

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

            # Update probabilities
            self.token_probs = {t: counts[t] / total if total > 0 else 1e-8 for t in vocab}

            # Compute the expected loss contribution per token, balancing frequency with information content
            contrib = {}
            for t in vocab:
                contrib[t] = counts[t] * -math.log(self.token_probs.get(t, 1e-8))

            ranked = sorted(list(vocab), key=lambda t: contrib.get(t, math.inf))

            # Remove 20% least useful tokens
            remove_count = max(1, len(ranked) // 5)
            for t in ranked[:remove_count]:
                vocab.remove(t)
                self.token_probs.pop(t, None)

    def tokenize(self, text):
        n = len(text)
        best_score = [math.inf] * (n + 1)
        best_edge = [None] * (n + 1)
        best_score[0] = 0

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
