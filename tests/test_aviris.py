import unittest
import hypercoast
import os
import matplotlib

matplotlib.use("Agg")  # Use the Agg backend to suppress plots


class TestHypercoastDesis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.url = "https://github.com/opengeos/datasets/releases/download/hypercoast/desis.tif"
        cls.filepath = "test_data/desis.tif"
        os.makedirs("test_data", exist_ok=True)
        hypercoast.download_file(cls.url, cls.filepath)

    def test_download_file(self):
        self.assertTrue(os.path.exists(self.filepath))

    def test_read_desis(self):
        dataset = hypercoast.read_desis(self.filepath)
        self.assertIsNotNone(dataset)

    def test_map_add_desis_single_band(self):
        m = hypercoast.Map()
        self.assertIsNotNone(m)
        m.add_basemap("Hybrid")
        m.add_desis(
            self.filepath,
            wavelengths=[1000],
            vmin=0,
            vmax=5000,
            nodata=0,
            colormap="jet",
        )
        m.add_colormap(cmap="jet", vmin=0, vmax=0.5, label="Reflectance")
        html = m.to_html()
        assert "DESIS" in html

    def test_map_add_desis_rgb(self):
        m = hypercoast.Map()
        self.assertIsNotNone(m)
        m.add_basemap("Hybrid")
        m.add_desis(
            self.filepath, wavelengths=[900, 600, 525], vmin=0, vmax=1000, nodata=0
        )
        m.add("spectral")
        html = m.to_html()
        assert "DESIS" in html


if __name__ == "__main__":
    unittest.main()
