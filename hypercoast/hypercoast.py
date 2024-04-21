"""Main module."""

import leafmap

from .emit import read_emit, plot_emit, viz_emit, emit_to_netcdf, emit_to_image
from .ui import SpectralWidget


class Map(leafmap.Map):
    """
    A class that extends leafmap.Map to provide additional functionality for hypercoast.

    Attributes:
        Any attributes inherited from leafmap.Map.

    Methods:
        Any methods inherited from leafmap.Map.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Map class.

        Args:
            **kwargs: Arbitrary keyword arguments that are passed to the parent class's constructor.
        """
        super().__init__(**kwargs)

    def search_emit(self, default_datset="EMITL2ARFL"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for EMIT.

        Args:
            default_dataset (str, optional): The default dataset to search for. Defaults to "EMITL2ARFL".
        """
        self.add("nasa_earth_data", default_dataset=default_datset)

    def search_pace(self, default_datset="PACE_OCI_L2_AOP_NRT"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for PACE.

        Args:
            default_dataset (str, optional): The default dataset to search for. Defaults to "PACE_OCI_L2_AOP_NRT".
        """
        self.add("nasa_earth_data", default_dataset=default_datset)

    def add_raster(
        self,
        source,
        indexes=None,
        colormap=None,
        vmin=None,
        vmax=None,
        nodata=None,
        attribution=None,
        layer_name="Raster",
        zoom_to_layer=True,
        visible=True,
        array_args={},
        **kwargs,
    ):
        """Add a local raster dataset to the map.
            If you are using this function in JupyterHub on a remote server (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib` to use when plotting a single band. See https://matplotlib.org/stable/gallery/color/colormap_reference.html. Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'Raster'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        import numpy as np

        if nodata is None:
            nodata = np.nan
        super().add_raster(
            source,
            indexes=indexes,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            attribution=attribution,
            layer_name=layer_name,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            array_args=array_args,
            **kwargs,
        )

    def add_emit(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=None,
        vmax=None,
        nodata=None,
        attribution=None,
        layer_name="Raster",
        zoom_to_layer=True,
        visible=True,
        array_args={},
        **kwargs,
    ):
        """Add a local raster dataset to the map.
            If you are using this function in JupyterHub on a remote server (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib` to use when plotting a single band. See https://matplotlib.org/stable/gallery/color/colormap_reference.html. Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'Raster'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        if isinstance(source, str):

            ds = read_emit(source, wavelengths=wavelengths)
            source = emit_to_image(ds, wavelengths=wavelengths)

        self.add_raster(
            source,
            indexes=indexes,
            colormap=colormap,
            vmin=vmin,
            vmax=vmax,
            nodata=nodata,
            attribution=attribution,
            layer_name=layer_name,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            array_args=array_args,
            **kwargs,
        )
