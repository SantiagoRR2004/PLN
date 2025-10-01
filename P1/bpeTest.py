from P1.p1_bpe import ByteLevelBPE
import unittest


class TestByteLevelBPE(unittest.TestCase):
    def test_to_byte_tokens(self):
        bpe = ByteLevelBPE()
        text = "hello"
        tokens = bpe._to_byte_tokens(text)
        self.assertEqual(
            tokens,
            [tuple([104]), tuple([101]), tuple([108]), tuple([108]), tuple([111])],
        )
