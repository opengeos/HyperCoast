import unittest
import hypercoast
import os


class TestHypercoastEmit(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.url = "https://github.com/opengeos/datasets/releases/download/netcdf/EMIT_L2A_RFL_001_20240404T161230_2409511_009.nc"
        cls.filepath = "test_data/EMIT_L2A_RFL_001_20240404T161230_2409511_009.nc"
        os.makedirs("test_data", exist_ok=True)
        hypercoast.download_file(cls.url, cls.filepath)
        cls.dataset = hypercoast.read_emit(cls.filepath)

    def test_download_file(self):
        self.assertTrue(os.path.exists(self.filepath))

    def test_read_emit(self):
        dataset = hypercoast.read_emit(self.filepath)
        self.assertIsNotNone(dataset)

    def test_map(self):
        m = hypercoast.Map()
        m.add_basemap("SATELLITE")
        wavelengths = [1000, 600, 500]
        m.add_emit(self.dataset, wavelengths, vmin=0, vmax=0.3, layer_name="EMIT")
        m.add("spectral")
        html = m.to_html()
        self.assertIsNotNone(m)
        assert "EMIT" in html


if __name__ == "__main__":
    unittest.main()
