class DFA:
    def __init__(self, words):
        self.initial = ""  # empty prefix
        self.states = set()
        self.final = set(words)
        self.transitions = {}

        # Build alphabet
        alphabet = []
        for w in words:
            alphabet += list(w)
        alphabet = set(alphabet)

        # Build states
        self.states = {""}
        for w in words:
            for i in range(1, len(w) + 1):
                self.states.add(w[:i])

        # Build transitions
        for state in self.states:
            for sym in alphabet:
                next_state = state + sym
                if next_state in self.states:
                    self.transitions[(state, sym)] = next_state

    def recognize(self, word):
        state = self.initial
        for ch in word:
            if (state, ch) not in self.transitions:
                return False
            state = self.transitions[(state, ch)]
        return state in self.final


if __name__ == "__main__":
    words = {"casa", "a", "asa", "saca"}

    dfa = DFA(words)

    for w in ["casa", "a", "asa", "saca", "cas", "aa"]:
        print(f"{w}: {dfa.recognize(w)}")
