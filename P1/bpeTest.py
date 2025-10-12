from P1 import ByteLevelBPE
import unittest
import pickle
import os


class TestByteLevelBPE(unittest.TestCase):
    def test_to_byte_tokens(self):
        text = "hello"
        tokens = ByteLevelBPE._to_byte_tokens(text)
        self.assertEqual(
            tokens,
            [tuple([104]), tuple([101]), tuple([108]), tuple([108]), tuple([111])],
        )

    def test_count_pairs(self):
        lines = [
            [tuple([104]), tuple([101]), tuple([108]), tuple([108]), tuple([111])],
            [tuple([104]), tuple([101]), tuple([108]), tuple([108, 109]), tuple([111])],
        ]
        pairs = ByteLevelBPE()._count_pairs(lines)
        expectedPairs = {
            (tuple([104]), tuple([101])): 2,
            (tuple([101]), tuple([108])): 2,
            (tuple([108]), tuple([108])): 1,
            (tuple([108]), tuple([111])): 1,
            (tuple([108]), tuple([108, 109])): 1,
            (tuple([108, 109]), tuple([111])): 1,
        }
        self.assertEqual(pairs, expectedPairs)

    def test_merge_in_line1(self):
        line = (tuple([0]), tuple([1]), tuple([0]), tuple([1]))
        bpe = ByteLevelBPE()
        bpe.pairs = {}
        merged = bpe._merge_in_line(line, (tuple([0]), tuple([1])))
        self.assertEqual(merged, [tuple([0, 1]), tuple([0, 1])])

    def test_merge_in_line2(self):
        # Check that it is greedy
        line = (tuple([0]), tuple([0]), tuple([0]))
        bpe = ByteLevelBPE()
        bpe.pairs = {}
        merged = bpe._merge_in_line(line, (tuple([0]), tuple([0])))
        self.assertEqual(merged, [tuple([0, 0]), tuple([0])])

    def test_eval_model(self):
        # Custom unpickler that can handle module path issues
        class CustomUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                # If the class was pickled from __main__, redirect to our import
                if module == "__main__" and name == "ByteLevelBPE":
                    return ByteLevelBPE
                return super().find_class(module, name)

        currentDirectory = os.path.dirname(os.path.abspath(__file__))

        parentDirectory = os.path.dirname(currentDirectory)

        model_path = os.path.join(parentDirectory, "bpe_model.pkl")
        if not os.path.exists(model_path):
            self.skipTest(f"Model file {model_path} not found.")

        with open(model_path, "rb") as f:
            bpe: ByteLevelBPE = CustomUnpickler(f).load()

        input_text = "the and ing tion with that this from they have been"
        tokens = bpe.tokenize(input_text)
        readable_tokens = [bytes(token).decode("utf-8") for token in tokens]
        self.assertEqual(
            readable_tokens,
            [
                "the ",
                "and ",
                "ing ",
                "tion ",
                "with ",
                "that ",
                "this ",
                "from ",
                "they ",
                "have ",
                "be",
                "en",
            ],
        )


if __name__ == "__main__":
    unittest.main()
