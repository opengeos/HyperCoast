import unittest
import hypercoast
import os
import matplotlib
import pytest

matplotlib.use("Agg")  # Use the Agg backend to suppress plots


class TestHypercoast(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.url = "https://github.com/opengeos/datasets/releases/download/netcdf/PACE_OCI.20240423T184658.L2.OC_AOP.V1_0_0.NRT.nc"
        cls.filepath = "test_data/PACE_OCI.20240423T184658.L2.OC_AOP.V1_0_0.NRT.nc"
        os.makedirs("test_data", exist_ok=True)
        hypercoast.download_file(cls.url, cls.filepath)
        cls.dataset = hypercoast.read_pace(cls.filepath)

    # uncomment marker if testing download files
    @pytest.mark.skip(reason="download takes long")
    def test_download_file(self):
        self.assertTrue(os.path.exists(self.filepath))

    def test_view_pace_pixel_locations(self):
        plot = hypercoast.view_pace_pixel_locations(self.filepath, step=20)
        self.assertIsNotNone(plot)

    def test_read_pace(self):
        dataset = hypercoast.read_pace(self.filepath)
        self.assertIsNotNone(dataset)

    def test_map(self):
        m = hypercoast.Map()
        m.add_basemap("Hybrid")
        wavelengths = [450]
        m.add_pace(
            self.dataset,
            wavelengths,
            colormap="jet",
            vmin=0,
            vmax=0.02,
            layer_name="PACE",
        )
        m.add_colormap(cmap="jet", vmin=0, vmax=0.02, label="Reflectance")
        m.add("spectral")
        html = m.to_html()
        self.assertIsNotNone(m)
        assert "PACE" in html

        m = hypercoast.Map()
        m.add_basemap("Hybrid")
        wavelengths = [450, 550, 650]
        m.add_pace(
            self.dataset,
            wavelengths,
            indexes=[3, 2, 1],
            vmin=0,
            vmax=0.02,
            layer_name="PACE",
        )
        m.add("spectral")
        html = m.to_html()
        self.assertIsNotNone(m)
        assert "PACE" in html


if __name__ == "__main__":
    unittest.main()
