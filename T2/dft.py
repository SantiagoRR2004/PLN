class DFT:
    def __init__(self, words, transducing_f):
        self.initial = ""  # empty prefix
        self.states = set()
        self.final = set(words)
        self.transitions = {}
        self.transducing_f = transducing_f

        self.reset()

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

    def reset(self):
        self.state = self.initial
        self.transduced = []

    def transduce(self, word):
        nope = ["#", "N", "O", "P", "E"]

        self.reset()
        for ch in word:
            if (self.state, ch) not in self.transitions:
                self.transduced = nope
                return False
            self.state = self.transitions[(self.state, ch)]
            self.transduced.append(self.transducing_f(ch))

        is_accepted = self.state in self.final
        if not is_accepted:
            self.transduced = nope
        return is_accepted


def encrypt_char(ch):
    return chr(ord(ch) + 1)


if __name__ == "__main__":
    words = {"casa", "a", "asa", "saca"}

    dft = DFT(words, encrypt_char)

    for w in ["casa", "a", "asa", "saca", "cas", "aa"]:
        print(f"{w}: {dft.transduce(w)}")
        print(f"{"".join(dft.transduced)}")
