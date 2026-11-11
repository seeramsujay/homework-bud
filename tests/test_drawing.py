import unittest
import numpy as np
from drawing import alphabet, alpha_to_num, num_to_alpha, skew, stretch, align

class TestDrawing(unittest.TestCase):
    def test_alphabet_mapping(self):
        self.assertIn("a", alphabet)
        self.assertIn("z", alphabet)
        self.assertEqual(alphabet[alpha_to_num["a"]], "a")
        self.assertEqual(chr(num_to_alpha[alpha_to_num["a"]]), "a")

    def test_stretch(self):
        coords = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
        stretched = stretch(coords, 2.0, 3.0)
        np.testing.assert_allclose(stretched[:, 0], [2.0, 6.0])
        np.testing.assert_allclose(stretched[:, 1], [6.0, 12.0])
        np.testing.assert_allclose(stretched[:, 2], [0.0, 1.0])

    def test_skew(self):
        coords = np.array([[1.0, 2.0, 0.0], [3.0, 4.0, 1.0]])
        skewed_zero = skew(coords, 0.0)
        np.testing.assert_allclose(skewed_zero[:, :2], coords[:, :2], atol=1e-6)

    def test_align(self):
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 2.0, 1.0]])
        aligned = align(coords)
        self.assertEqual(aligned.shape, coords.shape)

if __name__ == "__main__":
    unittest.main()
