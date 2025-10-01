from P1.p1_bpe import ByteLevelBPE
import unittest


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
        pairs = ByteLevelBPE._count_pairs(lines)
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
        merged = ByteLevelBPE._merge_in_line(line, (tuple([0]), tuple([1])))
        self.assertEqual(merged, [tuple([0, 1]), tuple([0, 1])])

    def test_merge_in_line2(self):
        # Check that it is greedy
        line = (tuple([0]), tuple([0]), tuple([0]))
        merged = ByteLevelBPE._merge_in_line(line, (tuple([0]), tuple([0])))
        self.assertEqual(merged, [tuple([0, 0]), tuple([0])])
