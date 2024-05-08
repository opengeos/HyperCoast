"""Main module."""

import ipyleaflet
import leafmap
import xarray as xr
import numpy as np
from .common import download_file, netcdf_groups
from .emit import read_emit, plot_emit, viz_emit, emit_to_netcdf, emit_to_image
from .pace import *


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

    def add(self, obj, position="topright", **kwargs):
        """Add a layer to the map.

        Args:
            **kwargs: Arbitrary keyword arguments that are passed to the parent class's add_layer method.
        """

        if isinstance(obj, str):
            if obj == "spectral":
                from .ui import SpectralWidget

                SpectralWidget(self, position=position)
                self.set_plot_options(add_marker_cluster=True)
            else:
                super().add(obj, **kwargs)

        else:
            super().add(obj, **kwargs)

    def search_emit(self, default_dataset="EMITL2ARFL"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for EMIT.

        Args:
            default_dataset (str, optional): The default dataset to search for. Defaults to "EMITL2ARFL".
        """
        self.add("nasa_earth_data", default_dataset=default_dataset)

    def search_pace(self, default_dataset="PACE_OCI_L2_AOP_NRT"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for PACE.

        Args:
            default_dataset (str, optional): The default dataset to search for. Defaults to "PACE_OCI_L2_AOP_NRT".
        """
        self.add("nasa_earth_data", default_dataset=default_dataset)

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
        nodata=np.nan,
        attribution=None,
        layer_name="EMIT",
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
            layer_name (str, optional): The layer name to use. Defaults to 'EMIT'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        xds = None
        if isinstance(source, str):

            xds = read_emit(source)
            source = emit_to_image(xds, wavelengths=wavelengths)
        elif isinstance(source, xr.Dataset):
            xds = source
            source = emit_to_image(xds, wavelengths=wavelengths)

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

        self.cog_layer_dict[layer_name]["xds"] = xds
        self.cog_layer_dict[layer_name]["type"] = "EMIT"

    def add_pace(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap="jet",
        vmin=None,
        vmax=None,
        nodata=np.nan,
        attribution=None,
        layer_name="PACE",
        zoom_to_layer=True,
        visible=True,
        method="nearest",
        gridded=False,
        array_args={},
        **kwargs,
    ):
        """Add a PACE dataset to the map.
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
            layer_name (str, optional): The layer name to use. Defaults to 'EMIT'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        if isinstance(source, str):

            source = read_pace(source)

        image = pace_to_image(
            source, wavelengths=wavelengths, method=method, gridded=gridded
        )

        if isinstance(wavelengths, list) and len(wavelengths) > 1:
            colormap = None

        self.add_raster(
            image,
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

        self.cog_layer_dict[layer_name]["xds"] = source
        self.cog_layer_dict[layer_name]["type"] = "PACE"

    def set_plot_options(
        self,
        add_marker_cluster=False,
        plot_type=None,
        overlay=False,
        position="bottomright",
        min_width=None,
        max_width=None,
        min_height=None,
        max_height=None,
        **kwargs,
    ):
        """Sets plotting options.

        Args:
            add_marker_cluster (bool, optional): Whether to add a marker cluster. Defaults to False.
            sample_scale (float, optional):  A nominal scale in meters of the projection to sample in . Defaults to None.
            plot_type (str, optional): The plot type can be one of "None", "bar", "scatter" or "hist". Defaults to None.
            overlay (bool, optional): Whether to overlay plotted lines on the figure. Defaults to False.
            position (str, optional): Position of the control, can be ‘bottomleft’, ‘bottomright’, ‘topleft’, or ‘topright’. Defaults to 'bottomright'.
            min_width (int, optional): Min width of the widget (in pixels), if None it will respect the content size. Defaults to None.
            max_width (int, optional): Max width of the widget (in pixels), if None it will respect the content size. Defaults to None.
            min_height (int, optional): Min height of the widget (in pixels), if None it will respect the content size. Defaults to None.
            max_height (int, optional): Max height of the widget (in pixels), if None it will respect the content size. Defaults to None.

        """
        plot_options_dict = {}
        plot_options_dict["add_marker_cluster"] = add_marker_cluster
        plot_options_dict["plot_type"] = plot_type
        plot_options_dict["overlay"] = overlay
        plot_options_dict["position"] = position
        plot_options_dict["min_width"] = min_width
        plot_options_dict["max_width"] = max_width
        plot_options_dict["min_height"] = min_height
        plot_options_dict["max_height"] = max_height

        for key in kwargs:
            plot_options_dict[key] = kwargs[key]

        self._plot_options = plot_options_dict

        if not hasattr(self, "_plot_marker_cluster"):
            self._plot_marker_cluster = ipyleaflet.MarkerCluster(name="Marker Cluster")

        if add_marker_cluster and (self._plot_marker_cluster not in self.layers):
            self.add(self._plot_marker_cluster)

    def spectral_to_df(self, **kwargs):
        """Converts the spectral data to a pandas DataFrame.

        Returns:
            pd.DataFrame: The spectral data as a pandas DataFrame.
        """
        import pandas as pd

        df = pd.DataFrame(self._spectral_data, **kwargs)
        return df

    def spectral_to_csv(self, filename, index=True, **kwargs):
        """Saves the spectral data to a CSV file.

        Args:
            filename (str): The output CSV file.
            index (bool, optional): Whether to write the index. Defaults to True.
        """
        df = self.spectral_to_df()
        df = df.rename_axis("band")
        df.to_csv(filename, index=index, **kwargs)
