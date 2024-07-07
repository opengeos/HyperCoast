import unittest
import hypercoast
import os


class TestHypercoastNeon(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.url = "https://github.com/opengeos/datasets/releases/download/hypercoast/NEON_D02_SERC_DP3_368000_4306000_reflectance.h5"
        cls.filepath = "test_data/neon.h5"
        os.makedirs("test_data", exist_ok=True)
        hypercoast.download_file(cls.url, cls.filepath)
        cls.dataset = hypercoast.read_neon(cls.filepath)

    def test_download_file(self):
        self.assertTrue(os.path.exists(self.filepath))

    def test_read_neon(self):
        dataset = hypercoast.read_neon(self.filepath)
        self.assertIsNotNone(dataset)

    def test_map_set_center(self):
        m = hypercoast.Map()
        m.set_center(-76.5134, 38.8973, 16)
        self.assertEqual(m.center, [38.8973, -76.5134])
        self.assertEqual(m.zoom, 16)


if __name__ == "__main__":
    unittest.main()
