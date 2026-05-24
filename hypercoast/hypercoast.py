# SPDX-FileCopyrightText: 2024 Qiusheng Wu <giswqs@gmail.com>
#
# SPDX-License-Identifier: MIT

"""Main module."""

import ipyleaflet
import leafmap
import xarray as xr
import numpy as np
import tempfile
import os
from typing import Union
from .aviris import (
    aviris_to_image,
    read_aviris,
    extract_aviris,
    get_aviris_asset_url,
    get_aviris_bounds,
    get_aviris_collection_concept_id,
    get_aviris_granule_ur,
)
from .desis import desis_to_image, read_desis, extract_desis, filter_desis
from .prisma import read_prisma, prisma_to_image, extract_prisma
from .enmap import read_enmap, enmap_to_image, extract_enmap
from .emit import (
    emit_to_image,
    read_emit,
    plot_emit,
    viz_emit,
    emit_to_netcdf,
)
from .neon import neon_to_image, read_neon
from .pace import (
    pace_to_image,
    read_pace,
    read_pace_aop,
    read_pace_bgc,
    read_pace_chla,
    view_pace_pixel_locations,
    viz_pace,
    viz_pace_chla,
    filter_pace,
    extract_pace,
    grid_pace,
    grid_pace_bgc,
    pace_to_image,
    pace_chla_to_image,
)
from .tanager import (
    read_tanager,
    read_tanager_stac,
    search_tanager,
    tanager_footprints,
    download_tanager,
    get_tanager_asset_url,
    tanager_to_image,
    extract_tanager,
    grid_tanager,
)
from .wyvern import read_wyvern, wyvern_to_image, extract_wyvern, filter_wyvern
from .cesl import (
    search_cesl,
    get_cesl_metadata,
    get_cesl_spectrum,
    plot_cesl_spectrum,
    get_cesl_sites,
    cesl_to_gdf,
    cesl_to_geojson,
)
from .appeears import (
    AppEEARSClient,
    appeears_area_task,
    appeears_download,
    appeears_emit_layers,
    appeears_layers,
    appeears_login,
    appeears_point_task,
    appeears_products,
    appeears_submit_task,
    appeears_wait,
    read_appeears,
)
from .ui import SpectralWidget
from .common import (
    download_file,
    search_datasets,
    search_nasa_data,
    download_nasa_data,
    search_pace,
    search_pace_chla,
    search_emit,
    search_aviris,
    search_ecostress,
    download_pace,
    download_emit,
    download_aviris,
    download_ecostress,
    nasa_earth_login,
    image_cube,
    open_dataset,
    download_acolite,
    run_acolite,
    pca,
    show_field_data,
)


class Map(leafmap.Map):
    """
    A class that extends leafmap.Map to provide additional functionality for
        hypercoast.

    Attributes:
        Any attributes inherited from leafmap.Map.

    Methods:
        Any methods inherited from leafmap.Map.
    """

    def __init__(self, **kwargs):
        """
        Initializes a new instance of the Map class.

        Args:
            **kwargs: Arbitrary keyword arguments that are passed to the parent
                class's constructor.
        """
        super().__init__(**kwargs)
        self._spectral_data = {}
        self._plot_options = None
        self._plot_marker_cluster = ipyleaflet.MarkerCluster(name="Marker Cluster")

    def add(self, obj, position="topright", xlim=None, ylim=None, **kwargs):
        """Add a layer to the map.

        Args:
            obj (str or object): The name of the layer or a layer object.
            position (str, optional): The position of the layer widget. Can be
                'topright', 'topleft', 'bottomright', or 'bottomleft'. Defaults
                to 'topright'.
            xlim (tuple, optional): The x-axis limits of the plot. Defaults to None.
            ylim (tuple, optional): The y-axis limits of the plot. Defaults to None.
            **kwargs: Arbitrary keyword arguments that are passed to the parent
                class's add_layer method.
        """

        if isinstance(obj, str):
            if obj == "spectral":

                SpectralWidget(self, position=position, xlim=xlim, ylim=ylim, **kwargs)
                self.set_plot_options(add_marker_cluster=True)
            else:
                super().add(obj, **kwargs)

        else:
            super().add(obj, **kwargs)

    def search_emit(self, default_dataset="EMITL2ARFL"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for
            EMIT.

        Args:
            default_dataset (str, optional): The default dataset to search for.
                Defaults to "EMITL2ARFL".
        """
        self.add("nasa_earth_data", default_dataset=default_dataset)

    def search_aviris(self, default_dataset="AV3_L2A_RFL_2357"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for
            AVIRIS.

        Args:
            default_dataset (str, optional): The default dataset to search for.
                Defaults to "AV3_L2A_RFL_2357".
        """
        self.add("nasa_earth_data", default_dataset=default_dataset)

    def search_pace(self, default_dataset="PACE_OCI_L2_AOP_NRT"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for
            PACE.

        Args:
            default_dataset (str, optional): The default dataset to search for.
                Defaults to "PACE_OCI_L2_AOP_NRT".
        """
        self.add("nasa_earth_data", default_dataset=default_dataset)

    def search_ecostress(self, default_dataset="ECO_L2T_LSTE"):
        """
        Adds a NASA Earth Data search tool to the map with a default dataset for
            ECOSTRESS.

        Args:
            default_dataset (str, optional): The default dataset to search for.
                Defaults to "ECO_L2T_LSTE".
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
        layer_index=None,
        zoom_to_layer=True,
        visible=True,
        opacity=1.0,
        array_args=None,
        client_args={"cors_all": False},
        open_args=None,
        **kwargs,
    ):
        """Add a local raster dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'Raster'.
            layer_index (int, optional): The index of the layer. Defaults to None.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to
                True.
            opacity (float, optional): The opacity of the layer. Defaults to 1.0.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            client_args (dict, optional): Additional arguments to pass to
                localtileserver.TileClient. Defaults to { "cors_all": False }.
            open_args (dict, optional): Additional arguments to pass to
                rioxarray.open_rasterio.

        """

        import rioxarray as rxr

        if array_args is None:
            array_args = {}
        if open_args is None:
            open_args = {}

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
            layer_index=layer_index,
            zoom_to_layer=zoom_to_layer,
            visible=visible,
            opacity=opacity,
            array_args=array_args,
            client_args=client_args,
            **kwargs,
        )

        if isinstance(source, str):
            da = rxr.open_rasterio(source, **open_args)
            dims = da.dims
            da = da.transpose(dims[1], dims[2], dims[0])

            xds = da.to_dataset(name="data")
            self.cog_layer_dict[layer_name]["xds"] = xds
            # if self.cog_layer_dict[layer_name].get("hyper") is None:
            #     self.cog_layer_dict[layer_name]["hyper"] = "COG"

        self.cog_layer_dict[layer_name]["type"] = "LOCAL"

    def add_dataset(
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
        array_args=None,
        open_args=None,
        **kwargs,
    ):
        import rioxarray as rxr
        from leafmap import array_to_image

        if array_args is None:
            array_args = {}

        if open_args is None:
            open_args = {}

        if isinstance(source, str):
            da = rxr.open_rasterio(source, **open_args)
            dims = da.dims
            da = da.transpose(dims[1], dims[2], dims[0])
            xds = da.to_dataset(name="data")

        elif not isinstance(source, xr.Dataset):
            raise ValueError(
                "source must be a path to a raster file or an xarray.Dataset object."
            )
        else:
            xds = source

        if indexes is None:
            if xds.sizes[dims[2]] < 3:
                indexes = [1]
            elif xds.sizes[dims[2]] < 4:
                indexes = [1, 2, 3]
            else:
                indexes = [3, 2, 1]

        bands = [i - 1 for i in indexes]
        da = xds.isel(band=bands)["data"]
        image = array_to_image(da, transpose=False)

        self.add_raster(
            image,
            indexes=None,
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
        self.cog_layer_dict[layer_name]["type"] = "XARRAY"
        self.cog_layer_dict[layer_name]["hyper"] = "XARRAY"
        self.cog_layer_dict[layer_name]["band_names"] = [
            "b" + str(i) for i in xds.coords["band"].values.tolist()
        ]
        self.cog_layer_dict[layer_name]["indexes"] = indexes
        self.cog_layer_dict[layer_name]["vis_bands"] = ["b" + str(i) for i in indexes]

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
        array_args=None,
        **kwargs,
    ):
        """Add an EMIT dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band.
                    See https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                    Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to
                interpret as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'EMIT'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to
                True.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        if array_args is None:
            array_args = {}

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
        self.cog_layer_dict[layer_name]["hyper"] = "EMIT"
        self._update_band_names(layer_name, wavelengths)

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
        array_args=None,
        **kwargs,
    ):
        """Add a PACE dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                    https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                    Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'EMIT'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        if array_args is None:
            array_args = {}

        if isinstance(source, str):

            source = read_pace(source)

        try:
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
            self.cog_layer_dict[layer_name]["hyper"] = "PACE"
            self._update_band_names(layer_name, wavelengths)
        except Exception as e:
            print(e)

    def add_desis(
        self,
        source,
        wavelengths=[900, 650, 525],
        indexes=None,
        colormap="jet",
        vmin=None,
        vmax=None,
        nodata=np.nan,
        attribution=None,
        layer_name="DESIS",
        zoom_to_layer=True,
        visible=True,
        method="nearest",
        array_args=None,
        **kwargs,
    ):
        """Add a DESIS dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is 'jet'.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'EMIT'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
        """
        if array_args is None:
            array_args = {}

        if isinstance(source, str):

            source = read_desis(source)

        image = desis_to_image(source, wavelengths=wavelengths, method=method)

        if isinstance(wavelengths, list) and len(wavelengths) > 1:
            colormap = None

        if isinstance(wavelengths, int):
            wavelengths = [wavelengths]

        if indexes is None:
            if isinstance(wavelengths, list) and len(wavelengths) == 1:
                indexes = [1]
            else:
                indexes = [1, 2, 3]

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
        self.cog_layer_dict[layer_name]["hyper"] = "DESIS"
        self._update_band_names(layer_name, wavelengths)

    def add_wyvern(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap="jet",
        vmin=None,
        vmax=None,
        nodata=np.nan,
        attribution=None,
        layer_name="WYVERN",
        zoom_to_layer=True,
        visible=True,
        method="nearest",
        wavelength_list=None,
        array_args=None,
        **kwargs,
    ):
        """Add a WYVERN dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the GeoTIFF file or the URL of the Cloud
                Optimized GeoTIFF.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is 'jet'.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'WYVERN'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            method (str, optional): The method to use for data interpolation.
                Defaults to "nearest".
            wavelength_list (list, optional): The list of wavelengths to use. If
                None, the wavelengths will be inferred from the dataset attributes.
                Defaults to None.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
        """
        if array_args is None:
            array_args = {}

        if isinstance(source, str):

            source = read_wyvern(source, wavelength_list=wavelength_list)

        image = wyvern_to_image(
            source,
            wavelengths=wavelengths,
            method=method,
            nodata=nodata,
            vmin=vmin,
            vmax=vmax,
        )

        if isinstance(wavelengths, list) and len(wavelengths) > 1:
            colormap = None

        if isinstance(wavelengths, int):
            wavelengths = [wavelengths]

        if indexes is None:
            if isinstance(wavelengths, list) and len(wavelengths) == 1:
                indexes = [1]
            else:
                indexes = [1, 2, 3]

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
        self.cog_layer_dict[layer_name]["hyper"] = "WYVERN"
        self._update_band_names(layer_name, wavelengths)

    def add_neon(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=0,
        vmax=0.5,
        nodata=np.nan,
        attribution=None,
        layer_name="NEON",
        zoom_to_layer=True,
        visible=True,
        array_args=None,
        method="nearest",
        **kwargs,
    ):
        """Add an NEON AOP dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str): The path to the NEON AOP HDF5 file.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                    https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                    Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to 0.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to 0.5.
            nodata (float, optional): The value from the band to use to
                interpret as not valid data. Defaults to np.nan.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'NEON'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults
                to True.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            method (str, optional): The method to use for data interpolation.
                Defaults to "nearest".
        """

        if array_args is None:
            array_args = {}
        xds = None
        if isinstance(source, str):

            xds = read_neon(source)
            source = neon_to_image(xds, wavelengths=wavelengths, method=method)
        elif isinstance(source, xr.Dataset):
            xds = source
            source = neon_to_image(xds, wavelengths=wavelengths, method=method)

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
        self.cog_layer_dict[layer_name]["hyper"] = "NEON"
        self._update_band_names(layer_name, wavelengths)

    def add_aviris(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=0,
        vmax=0.5,
        nodata=np.nan,
        attribution=None,
        layer_name="AVIRIS",
        zoom_to_layer=True,
        visible=True,
        array_args=None,
        method="nearest",
        asset="RFL_ORT",
        visual_asset="RFL_ORT_QL",
        prefer_s3=False,
        open_args=None,
        dataset=None,
        use_cmr=True,
        titiler_cmr_endpoint=None,
        **kwargs,
    ):
        """Add an AVIRIS dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str, dict, or xarray.Dataset): The path or URL to an AVIRIS
                file, a CMR/earthaccess granule, or an xarray Dataset.
            wavelengths (list, optional): Wavelength values to render and use for
                band metadata. Defaults to None.
            indexes (int, optional): The band(s) to use. Band indexing starts
                at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                    https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                    Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to 0.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to 0.5.
            nodata (float, optional): The value from the band to use to
                interpret as not valid data. Defaults to np.nan.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'NEON'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults
                to True.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            method (str, optional): The method to use for data interpolation.
                Defaults to "nearest".
            asset (str, optional): AVIRIS granule asset used for the spectral
                dataset when ``source`` is a CMR/earthaccess granule. Defaults
                to ``"RFL_ORT"``.
            visual_asset (str, optional): AVIRIS granule COG asset used for the
                map layer. Set to None to render the map from the spectral
                dataset. Defaults to ``"RFL_ORT_QL"``.
            prefer_s3 (bool, optional): Prefer S3 asset links from granules when
                available. Defaults to False.
            open_args (dict, optional): Extra keyword arguments passed to
                ``read_aviris`` for path, URL, and granule inputs.
            dataset (xarray.Dataset, optional): Pre-opened AVIRIS dataset to
                attach for spectral extraction when ``source`` is a CMR or
                earthaccess granule. Defaults to None.
            use_cmr (bool, optional): Use NASA TiTiler CMR to stream the
                quicklook COG when ``source`` is a CMR or earthaccess granule.
                Defaults to True.
            titiler_cmr_endpoint (str, optional): TiTiler CMR endpoint. Defaults
                to the leafmap default.
        """
        if array_args is None:
            array_args = {}
        if open_args is None:
            open_args = {}

        xds = dataset
        granule_source = None
        cmr_layer = False
        if not isinstance(source, (str, os.PathLike, xr.Dataset)):
            granule_source = source
            if visual_asset is not None and use_cmr:
                concept_id = get_aviris_collection_concept_id(granule_source)
                granule_ur = get_aviris_granule_ur(granule_source)
                if concept_id is None or granule_ur is None:
                    raise ValueError(
                        "The AVIRIS granule is missing CMR collection or granule IDs."
                    )
                self.add_cmr_layer(
                    concept_id=concept_id,
                    granule_ur=granule_ur,
                    backend="rasterio",
                    bands=visual_asset,
                    bands_regex=visual_asset,
                    name=layer_name,
                    attribution=attribution or "NASA Earthdata",
                    opacity=kwargs.pop("opacity", 1.0),
                    shown=visible,
                    titiler_cmr_endpoint=titiler_cmr_endpoint,
                    zoom_to_layer=zoom_to_layer,
                    **kwargs,
                )
                if zoom_to_layer:
                    bounds = get_aviris_bounds(granule_source)
                    if bounds is not None:
                        lon_span = bounds[2] - bounds[0]
                        lat_span = bounds[3] - bounds[1]
                        max_span = max(lon_span, lat_span)
                        if max_span > 0:
                            self.zoom = int(
                                np.clip(np.floor(np.log2(360.0 / max_span)), 1, 18)
                            )
                        self.center = [
                            (bounds[1] + bounds[3]) / 2,
                            (bounds[0] + bounds[2]) / 2,
                        ]
                        self.fit_bounds(
                            [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
                        )
                cmr_layer = True
            else:
                source = get_aviris_asset_url(
                    granule_source, asset=asset, prefer_s3=prefer_s3
                )

        if isinstance(source, (str, os.PathLike)):
            source = os.fspath(source)

            if xds is None:
                xds = read_aviris(source, method=method, **open_args)
            if granule_source is not None and visual_asset is not None:
                source = get_aviris_asset_url(
                    granule_source, asset=visual_asset, prefer_s3=prefer_s3
                )
            else:
                source = aviris_to_image(xds, wavelengths=wavelengths, method=method)
        elif isinstance(source, xr.Dataset):
            xds = source
            source = aviris_to_image(xds, wavelengths=wavelengths, method=method)

        if cmr_layer:
            if not hasattr(self, "cog_layer_dict"):
                self.cog_layer_dict = {}
            if layer_name not in self.cog_layer_dict:
                self.cog_layer_dict[layer_name] = {}
        elif (
            isinstance(source, str)
            and source.split("?", 1)[0].lower().endswith((".tif", ".tiff"))
            and source.startswith(("http://", "https://", "s3://"))
        ):
            cog_kwargs = kwargs.copy()
            if indexes is not None:
                cog_kwargs["bidx"] = indexes
            self.add_cog_layer(
                source,
                name=layer_name,
                attribution=attribution or "",
                shown=visible,
                zoom_to_layer=zoom_to_layer,
                **cog_kwargs,
            )
        else:
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
        self.cog_layer_dict[layer_name]["hyper"] = "AVIRIS"
        self._update_band_names(layer_name, wavelengths)

    def add_prisma(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=0,
        vmax=0.5,
        nodata=np.nan,
        attribution=None,
        layer_name="PRISMA",
        zoom_to_layer=True,
        visible=True,
        array_args=None,
        method="nearest",
        **kwargs,
    ):
        """Add a PRISMA dataset to the map.

        This function reads a PRISMA hyperspectral dataset, optionally selects
        specific wavelengths, converts the data to an image, and adds it as a
        raster layer to the map. The dataset can be provided as a file path or
        as an xarray Dataset.

        Args:
            source (str or xarray.Dataset): The path to the PRISMA file or an
                in-memory xarray Dataset containing PRISMA data.
            wavelengths (list or np.ndarray, optional): Specific wavelengths to
                select from the dataset. If None, all wavelengths are used.
                Defaults to None.
            indexes (int or list, optional): The band(s) to display. Band
                indexing starts at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See:
                https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                Default is greyscale.
            vmin (float, optional): The minimum value for color mapping when
                plotting a single band. Defaults to 0.
            vmax (float, optional): The maximum value for color mapping when
                plotting a single band. Defaults to 0.5.
            nodata (float, optional): Value in the raster to interpret as
                no-data. Defaults to np.nan.
            attribution (str, optional): Attribution for the source raster.
                Defaults to None.
            layer_name (str, optional): The name to assign to the map layer.
                Defaults to "PRISMA".
            zoom_to_layer (bool, optional): Whether to zoom the map to the
                extent of the layer after adding it. Defaults to True.
            visible (bool, optional): Whether the layer should be visible when
                first added. Defaults to True.
            array_args (dict, optional): Additional keyword arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            method (str, optional): Method to use for wavelength interpolation
                when selecting bands. Options may include "nearest", "linear",
                etc. Defaults to "nearest".
            **kwargs: Additional keyword arguments passed to `add_raster`.
        """
        if array_args is None:
            array_args = {}

        if isinstance(source, str):
            xds = read_prisma(source, wavelengths=wavelengths, method=method)
        else:
            xds = source

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_path = tmp.name

        prisma_to_image(xds, wavelengths=wavelengths, method=method, output=temp_path)

        self.add_raster(
            temp_path,
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
        self.cog_layer_dict[layer_name]["hyper"] = "PRISMA"

    def add_enmap(
        self,
        source,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=0,
        vmax=0.5,
        nodata=np.nan,
        attribution=None,
        layer_name="EnMAP",
        zoom_to_layer=True,
        visible=True,
        array_args=None,
        method="nearest",
        **kwargs,
    ):
        """Add an EnMAP dataset to the map.

        This function reads an EnMAP hyperspectral dataset, optionally selects
        specific wavelengths, converts the data to an image, and adds it as a
        raster layer to the map. The dataset can be provided as a file path or as
        an xarray Dataset.

        Args:
            source (str or xarray.Dataset): The path to the EnMAP file or an
                in-memory xarray Dataset containing EnMAP data.
            wavelengths (list or np.ndarray, optional): Specific wavelengths to
                select from the dataset. If None, all wavelengths are used.
                Defaults to None.
            indexes (int or list, optional): The band(s) to display. Band
                indexing starts at 1. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band.
            vmin (float, optional): The minimum value for color mapping when
                plotting a single band. Defaults to 0.
            vmax (float, optional): The maximum value for color mapping when
                plotting a single band. Defaults to 0.5.
            nodata (float, optional): Value in the raster to interpret as
                no-data. Defaults to np.nan.
            attribution (str, optional): Attribution for the source raster.
                Defaults to None.
            layer_name (str, optional): The name to assign to the map layer.
                Defaults to "EnMAP".
            zoom_to_layer (bool, optional): Whether to zoom the map to the
                extent of the layer after adding it. Defaults to True.
            visible (bool, optional): Whether the layer should be visible when
                first added. Defaults to True.
            array_args (dict, optional): Additional keyword arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
            method (str, optional): Method to use for wavelength interpolation
                when selecting bands. Defaults to "nearest".
            **kwargs: Additional keyword arguments passed to `add_raster`.
        """
        if array_args is None:
            array_args = {}

        if isinstance(source, str):
            xds = read_enmap(source, wavelengths=wavelengths, method=method)
        else:
            xds = source

        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            temp_path = tmp.name

        enmap_to_image(xds, wavelengths=wavelengths, method=method, output=temp_path)

        self.add_raster(
            temp_path,
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
        self.cog_layer_dict[layer_name]["hyper"] = "EnMAP"

    def add_tanager(
        self,
        source,
        bands=None,
        wavelengths=None,
        indexes=None,
        colormap=None,
        vmin=0,
        vmax=120,
        nodata=np.nan,
        attribution=None,
        layer_name="Tanager",
        zoom_to_layer=True,
        visible=True,
        method="nearest",
        asset="ortho_radiance_hdf5",
        visual_asset="ortho_visual",
        array_args=None,
        **kwargs,
    ):
        """Add a Tanager dataset to the map.
            If you are using this function in JupyterHub on a remote server
                (e.g., Binder, Microsoft Planetary Computer) and
            if the raster does not render properly, try installing
                jupyter-server-proxy using `pip install jupyter-server-proxy`,
            then running the following code before calling this function. For
                more info, see https://bit.ly/3JbmF93.

            import os
            os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = 'proxy/{port}'

        Args:
            source (str or dict): The path to a Tanager HDF5 file, a Tanager
                STAC item URL, a Planet STAC browser item URL, a STAC item
                dictionary, or an xarray.Dataset.
            asset (str, optional): HDF5 STAC asset key to download when
                ``source`` is a STAC item. Defaults to
                ``"ortho_radiance_hdf5"``.
            visual_asset (str, optional): STAC COG asset key used for the map
                layer when ``source`` is a STAC item. Defaults to
                ``"ortho_visual"``. Set to None to render from the HDF5 data.
            bands (list, optional): The band indices to select. Defaults to None.
            wavelengths (list, optional): The wavelength values to select. Takes priority over bands. Defaults to None.
            colormap (str, optional): The name of the colormap from `matplotlib`
                to use when plotting a single band. See
                    https://matplotlib.org/stable/gallery/color/colormap_reference.html.
                    Default is greyscale.
            vmin (float, optional): The minimum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            vmax (float, optional): The maximum value to use when colormapping
                the palette when plotting a single band. Defaults to None.
            nodata (float, optional): The value from the band to use to interpret
                as not valid data. Defaults to None.
            attribution (str, optional): Attribution for the source raster. This
                defaults to a message about it being a local file.. Defaults to None.
            layer_name (str, optional): The layer name to use. Defaults to 'EMIT'.
            zoom_to_layer (bool, optional): Whether to zoom to the extent of the
                layer. Defaults to True.
            visible (bool, optional): Whether the layer is visible. Defaults to True.
            array_args (dict, optional): Additional arguments to pass to
                `array_to_memory_file` when reading the raster. Defaults to {}.
        """

        if array_args is None:
            array_args = {}

        stac_source = None
        if isinstance(source, dict) or (
            isinstance(source, str)
            and source.startswith("http")
            and "/data/stac/" in source
            and ".json" in source
        ):
            stac_source = source
            source = read_tanager_stac(source, asset=asset)
        elif isinstance(source, str):
            source = read_tanager(source)

        selected_wavelengths = []
        if wavelengths is not None:
            selected_wavelengths = wavelengths
        elif bands is not None:
            for band in bands:
                if isinstance(band, (int, np.integer)) or (
                    isinstance(band, float) and band < 500
                ):
                    # Treat as band index
                    selected_wavelengths.append(
                        source.coords["wavelength"].values[int(band)]
                    )
                else:
                    # Treat as wavelength value
                    selected_wavelengths.append(band)

        else:
            selected_wavelengths = [876.3, 675.88, 625.83]

        if isinstance(selected_wavelengths, list) and len(selected_wavelengths) > 1:
            colormap = None

        try:
            if stac_source is not None and visual_asset is not None:
                image = get_tanager_asset_url(stac_source, asset=visual_asset)
                cog_kwargs = kwargs.copy()
                if indexes is not None:
                    cog_kwargs["bidx"] = indexes
                self.add_cog_layer(
                    image,
                    name=layer_name,
                    attribution=attribution or "",
                    shown=visible,
                    zoom_to_layer=zoom_to_layer,
                    **cog_kwargs,
                )
            else:
                image = tanager_to_image(
                    source, wavelengths=selected_wavelengths, method=method
                )
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
            self.cog_layer_dict[layer_name]["vmax"] = vmax
            self.cog_layer_dict[layer_name]["vmin"] = vmin
            self.cog_layer_dict[layer_name]["hyper"] = "TANAGER"
            self._update_band_names(layer_name, selected_wavelengths)
        except Exception as e:
            print(e)

    def add_tanager_footprints(
        self,
        bbox=None,
        temporal=None,
        collections=None,
        count=-1,
        query=None,
        cloud_percent=None,
        catalog_url=None,
        output=None,
        crs="EPSG:4326",
        unique=True,
        timeout=30,
        layer_name="Tanager footprints",
        style=None,
        hover_style=None,
        style_callback=None,
        fill_colors=None,
        info_mode="on_hover",
        zoom_to_layer=True,
        **kwargs,
    ):
        """Add Tanager STAC item footprints to the map.

        Args:
            bbox (list, optional): Bounding box ``[xmin, ymin, xmax, ymax]`` in
                EPSG:4326.
            temporal (str or tuple, optional): Date/time range as
                ``"start/end"``, ``"start,end"``, or ``(start, end)``.
            collections (str or list, optional): Tanager collection ids or
                titles. Defaults to all collections.
            count (int, optional): Maximum number of matching STAC item records
                to inspect. ``-1`` means all. Defaults to ``-1``.
            query (str, optional): Case-insensitive text search against item id,
                title, description, and location description.
            cloud_percent (float, optional): Maximum item ``cloud_percent``.
            catalog_url (str, optional): Tanager STAC catalog URL.
            output (str, optional): File path to save the footprint
                GeoDataFrame.
            crs (str, optional): CRS for the returned GeoDataFrame. Defaults to
                ``"EPSG:4326"``.
            unique (bool, optional): Deduplicate scenes that appear in multiple
                thematic collections. Defaults to True.
            timeout (int, optional): HTTP request timeout in seconds. Defaults
                to 30.
            layer_name (str, optional): Map layer name. Defaults to
                ``"Tanager footprints"``.
            style (dict, optional): Polygon style dictionary.
            hover_style (dict, optional): Polygon hover style dictionary.
            style_callback (callable, optional): Per-feature style callback.
            fill_colors (list, optional): Random fill colors passed to leafmap.
            info_mode (str, optional): ``"on_hover"`` or ``"on_click"``.
                Defaults to ``"on_hover"``.
            zoom_to_layer (bool, optional): Whether to zoom to the footprints.
                Defaults to True.
            **kwargs: Additional exact-match STAC property filters and keyword
                arguments passed to ``leafmap.Map.add_gdf``.

        Returns:
            geopandas.GeoDataFrame: The footprint GeoDataFrame added to the map.
        """
        if style is None:
            style = {
                "color": "#00A6D6",
                "weight": 2,
                "fillColor": "#00A6D6",
                "fillOpacity": 0.08,
            }
        if hover_style is None:
            hover_style = {"weight": 4, "fillOpacity": 0.18}

        search_kwargs = {}
        add_gdf_kwargs = {}
        add_gdf_keys = {"encoding", "marker", "marker_cluster"}
        for key, value in kwargs.items():
            if key in add_gdf_keys:
                add_gdf_kwargs[key] = value
            else:
                search_kwargs[key] = value

        if catalog_url is None:
            gdf = tanager_footprints(
                bbox=bbox,
                temporal=temporal,
                collections=collections,
                count=count,
                query=query,
                cloud_percent=cloud_percent,
                output=output,
                crs=crs,
                unique=unique,
                timeout=timeout,
                **search_kwargs,
            )
        else:
            gdf = tanager_footprints(
                bbox=bbox,
                temporal=temporal,
                collections=collections,
                count=count,
                query=query,
                cloud_percent=cloud_percent,
                catalog_url=catalog_url,
                output=output,
                crs=crs,
                unique=unique,
                timeout=timeout,
                **search_kwargs,
            )

        self.add_gdf(
            gdf,
            layer_name=layer_name,
            style=style,
            hover_style=hover_style,
            style_callback=style_callback,
            fill_colors=fill_colors,
            info_mode=info_mode,
            zoom_to_layer=zoom_to_layer,
            **add_gdf_kwargs,
        )
        return gdf

    def add_hyper(self, xds, dtype, wvl_indexes=None, **kwargs):
        """Add a hyperspectral dataset to the map.

        Args:
            xds (str): The Xarray dataset containing the hyperspectral data.
            dtype (str): The type of the hyperspectral dataset. Can be one of
                "EMIT", "PACE", "DESIS", "NEON", "AVIRIS".
            **kwargs: Additional keyword arguments to pass to the corresponding
                add function.
        """

        if wvl_indexes is not None:
            if dtype == "XARRAY":
                kwargs["indexes"] = [i + 1 for i in wvl_indexes]
            else:
                if "wavelength" in xds.dims:
                    kwargs["wavelengths"] = (
                        xds.isel(wavelength=wvl_indexes)
                        .coords["wavelength"]
                        .values.tolist()
                    )
                else:
                    kwargs["bands"] = wvl_indexes

        if dtype == "EMIT":
            self.add_emit(xds, **kwargs)
        elif dtype == "PACE":
            self.add_pace(xds, **kwargs)
        elif dtype == "DESIS":
            self.add_desis(xds, **kwargs)
        elif dtype == "NEON":
            self.add_neon(xds, **kwargs)
        elif dtype == "AVIRIS":
            self.add_aviris(xds, **kwargs)
        elif dtype == "TANAGER":
            self.add_tanager(xds, **kwargs)
        elif dtype == "XARRAY":
            kwargs.pop("wavelengths", None)
            self.add_dataset(xds, **kwargs)

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
            add_marker_cluster (bool, optional): Whether to add a marker cluster.
                Defaults to False.
            sample_scale (float, optional):  A nominal scale in meters of the
                projection to sample in . Defaults to None.
            plot_type (str, optional): The plot type can be one of "None", "bar",
                "scatter" or "hist". Defaults to None.
            overlay (bool, optional): Whether to overlay plotted lines on the
                figure. Defaults to False.
            position (str, optional): Position of the control, can be
                ‘bottomleft’, ‘bottomright’, ‘topleft’, or ‘topright’. Defaults
                to 'bottomright'.
            min_width (int, optional): Min width of the widget (in pixels), if
                None it will respect the content size. Defaults to None.
            max_width (int, optional): Max width of the widget (in pixels), if
                None it will respect the content size. Defaults to None.
            min_height (int, optional): Min height of the widget (in pixels), if
                None it will respect the content size. Defaults to None.
            max_height (int, optional): Max height of the widget (in pixels), if
                None it will respect the content size. Defaults to None.

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

    def spectral_to_gdf(self, **kwargs):
        """Converts the spectral data to a GeoPandas GeoDataFrame.

        Returns:
            gpd.DataFrame: The spectral data as a pandas DataFrame.
        """
        import geopandas as gpd
        from shapely.geometry import Point

        df = self.spectral_to_df()

        if len(df) == 0:
            return df

        # Step 1: Extract the coordinates from the columns
        if "wavelength" in df.columns:
            df = df.rename(columns={"wavelength": "latlon"})
        elif "wavelengths" in df.columns:
            df = df.rename(columns={"wavelengths": "latlon"})
        coordinates = [col.strip("()").split() for col in df.columns[1:]]
        coords = [(float(lat), float(lon)) for lat, lon in coordinates]

        # Step 2: Create Point geometries for each coordinate
        points = [Point(lon, lat) for lat, lon in coords]

        # Step 3: Create a GeoDataFrame
        df_transposed = df.set_index("latlon").T

        # Convert the column names to strings to ensure compatibility with GeoJSON
        df_transposed.columns = df_transposed.columns.astype(str)

        # Create the GeoDataFrame
        gdf = gpd.GeoDataFrame(df_transposed, geometry=points, **kwargs)

        # Set the coordinate reference system (CRS)
        gdf = gdf.set_geometry("geometry").set_crs("EPSG:4326")

        return gdf

    def spectral_to_csv(self, filename, index=True, **kwargs):
        """Saves the spectral data to a CSV file.

        Args:
            filename (str): The output CSV file.
            index (bool, optional): Whether to write the index. Defaults to True.
        """
        df = self.spectral_to_df()
        df = df.rename_axis("band")
        df.to_csv(filename, index=index, **kwargs)

    def _update_band_names(self, layer_name, wavelengths):

        # Function to find the nearest indices
        def find_nearest_indices(
            dataarray, selected_wavelengths, dim_name="wavelength"
        ):
            indices = []
            for wavelength in selected_wavelengths:
                if dim_name == "band":
                    nearest_wavelength = dataarray.sel(
                        band=wavelength, method="nearest"
                    )
                else:
                    nearest_wavelength = dataarray.sel(
                        wavelength=wavelength, method="nearest"
                    )
                nearest_wavelength_index = nearest_wavelength[dim_name].item()
                nearest_index = (
                    dataarray[dim_name].values.tolist().index(nearest_wavelength_index)
                )
                indices.append(nearest_index + 1)
            return indices

        if "xds" in self.cog_layer_dict[layer_name]:
            xds = self.cog_layer_dict[layer_name]["xds"]
            if xds is None:
                return
            dim_name = "wavelength"

            if "band" in xds:
                dim_name = "band"

            band_count = xds.sizes[dim_name]
            band_names = ["b" + str(band) for band in range(1, band_count + 1)]
            self.cog_layer_dict[layer_name]["band_names"] = band_names

            if wavelengths is None:
                return

            try:
                indexes = find_nearest_indices(xds, wavelengths, dim_name=dim_name)
                vis_bands = ["b" + str(index) for index in indexes]
                self.cog_layer_dict[layer_name]["indexes"] = indexes
                self.cog_layer_dict[layer_name]["vis_bands"] = vis_bands
            except Exception as e:
                print(e)

    def add_field_data(
        self,
        data: Union[str],
        x_col: str = "wavelength",
        y_col_prefix: str = "(",
        x_label: str = "Wavelengths (nm)",
        y_label: str = "Reflectance",
        use_marker_cluster: bool = True,
        min_width: int = 400,
        max_width: int = 600,
        min_height: int = 200,
        max_height: int = 250,
        layer_name: str = "Marker Cluster",
        **kwargs,
    ):
        """
        Displays field data on a map with interactive markers and popups showing time series data.

        Args:
            data (Union[str, pd.DataFrame]): Path to the CSV file or a pandas DataFrame containing the data.
            x_col (str): Column name to use for the x-axis of the charts. Default is "wavelength".
            y_col_prefix (str): Prefix to identify the columns that contain the location-specific data. Default is "(".
            x_label (str): Label for the x-axis of the charts. Default is "Wavelengths (nm)".
            y_label (str): Label for the y-axis of the charts. Default is "Reflectance".
            use_marker_cluster (bool): Whether to use marker clustering. Default is True.
            min_width (int): Minimum width of the popup. Default is 400.
            max_width (int): Maximum width of the popup. Default is 600.
            min_height (int): Minimum height of the popup. Default is 200.
            max_height (int): Maximum height of the popup. Default is 250.
            layer_name (str): Name of the marker cluster layer. Default is "Marker Cluster".

        Returns:
            Map: An ipyleaflet Map with the added markers and popups.
        """
        show_field_data(
            data,
            x_col,
            y_col_prefix,
            x_label=x_label,
            y_label=y_label,
            use_marker_cluster=use_marker_cluster,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            layer_name=layer_name,
            m=self,
            **kwargs,
        )
