import unittest

from tfhe import utils


class TestUtils(unittest.TestCase):
    def test_encode_decode(self):
        for i in range(-4, 4):
            self.assertEqual(utils.decode(utils.encode(i)), i)

    def test_decode_max_int(self):
        self.assertEqual(utils.decode(2**31 - 1), -4)

    def test_encode_bool(self):
        self.assertEqual(utils.encode_bool(False), utils.encode(0))
        self.assertEqual(utils.encode_bool(True), utils.encode(2))

    def test_decode_bool(self):
        self.assertFalse(utils.decode_bool(utils.encode(0)))
        self.assertTrue(utils.decode_bool(utils.encode(2)))


if __name__ == "__main__":
    unittest.main()
