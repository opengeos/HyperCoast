"""This module contains the user interface for the hypercoast package.
"""

import os
import ipyleaflet
import bqplot
import ipywidgets as widgets
import numpy as np
import xarray as xr
from bqplot import pyplot as plt
from IPython.core.display import display
from ipyfilechooser import FileChooser
from .pace import extract_pace
from .desis import extract_desis
from .neon import extract_neon
from .aviris import extract_aviris


class SpectralWidget(widgets.HBox):
    """
    A widget for spectral data visualization on a map.

    Attributes:
        _host_map (Map): The map to host the widget.
        on_close (function): Function to be called when the widget is closed.
        _output_widget (widgets.Output): The output widget to display results.
        _output_control (ipyleaflet.WidgetControl): The control for the output widget.
        _on_map_interaction (function): Function to handle map interactions.
        _spectral_widget (SpectralWidget): The spectral widget itself.
        _spectral_control (ipyleaflet.WidgetControl): The control for the spectral widget.
    """

    def __init__(self, host_map, stack=True, position="topright"):
        """
        Initializes a new instance of the SpectralWidget class.

        Args:
            host_map (Map): The map to host the widget.
            position (str, optional): The position of the widget on the map. Defaults to "topright".
        """
        self._host_map = host_map
        self.on_close = None
        self._stack = stack
        self._show_plot = False

        fig_margin = {"top": 20, "bottom": 35, "left": 50, "right": 20}
        fig = plt.figure(
            # title=None,
            fig_margin=fig_margin,
            layout={"width": "500px", "height": "300px"},
        )

        self._fig = fig
        self._host_map._fig = fig

        layer_names = list(host_map.cog_layer_dict.keys())
        layers_widget = widgets.Dropdown(options=layer_names)
        layers_widget.layout.width = "18ex"

        close_btn = widgets.Button(
            icon="times",
            tooltip="Close the widget",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        reset_btn = widgets.Button(
            icon="trash",
            tooltip="Remove all markers",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        settings_btn = widgets.Button(
            icon="gear",
            tooltip="Change layer settings",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        stack_btn = widgets.ToggleButton(
            value=stack,
            icon="area-chart",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        def settings_btn_click(_):
            self._host_map._add_layer_editor(
                position="topright",
                layer_dict=self._host_map.cog_layer_dict[layers_widget.value],
            )

        settings_btn.on_click(settings_btn_click)

        def reset_btn_click(_):
            if hasattr(self._host_map, "_plot_marker_cluster"):
                self._host_map._plot_marker_cluster.markers = []
                self._host_map._plot_markers = []

            if hasattr(self._host_map, "_spectral_data"):
                self._host_map._spectral_data = {}

            self._output_widget.clear_output()
            self._show_plot = False
            plt.clear()

        reset_btn.on_click(reset_btn_click)

        save_btn = widgets.Button(
            icon="floppy-o",
            tooltip="Save the data to a CSV",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        def chooser_callback(chooser):
            if chooser.selected:
                file_path = chooser.selected
                self._host_map.spectral_to_csv(file_path)
                if (
                    hasattr(self._host_map, "_file_chooser_control")
                    and self._host_map._file_chooser_control in self._host_map.controls
                ):
                    self._host_map.remove_control(self._host_map._file_chooser_control)
                    self._host_map._file_chooser.close()

        def save_btn_click(_):
            if not hasattr(self._host_map, "_spectral_data"):
                return

            self._output_widget.clear_output()
            file_chooser = FileChooser(
                os.getcwd(), layout=widgets.Layout(width="454px")
            )
            file_chooser.filter_pattern = "*.csv"
            file_chooser.use_dir_icons = True
            file_chooser.title = "Save spectral data to a CSV file"
            file_chooser.default_filename = "spectral_data.csv"
            file_chooser.show_hidden = False
            file_chooser.register_callback(chooser_callback)
            file_chooser_control = ipyleaflet.WidgetControl(
                widget=file_chooser, position="topright"
            )
            self._host_map.add(file_chooser_control)
            setattr(self._host_map, "_file_chooser", file_chooser)
            setattr(self._host_map, "_file_chooser_control", file_chooser_control)

        save_btn.on_click(save_btn_click)

        def close_widget(_):
            self.cleanup()

        close_btn.on_click(close_widget)

        super().__init__(
            [layers_widget, settings_btn, stack_btn, reset_btn, save_btn, close_btn]
        )

        output = widgets.Output()
        output_control = ipyleaflet.WidgetControl(widget=output, position="bottomright")
        self._output_widget = output
        self._output_control = output_control
        self._host_map.add(output_control)

        if not hasattr(self._host_map, "_spectral_data"):
            self._host_map._spectral_data = {}

        def handle_interaction(**kwargs):

            latlon = kwargs.get("coordinates")
            lat = latlon[0]
            lon = latlon[1]
            if kwargs.get("type") == "click" and self._host_map._layer_editor is None:
                layer_name = layers_widget.value

                if not hasattr(self._host_map, "_plot_markers"):
                    self._host_map._plot_markers = []
                markers = self._host_map._plot_markers
                marker_cluster = self._host_map._plot_marker_cluster
                markers.append(ipyleaflet.Marker(location=latlon, draggable=False))
                marker_cluster.markers = markers
                self._host_map._plot_marker_cluster = marker_cluster

                ds = self._host_map.cog_layer_dict[layer_name]["xds"]
                if self._host_map.cog_layer_dict[layer_name]["hyper"] == "EMIT":
                    da = ds.sel(latitude=lat, longitude=lon, method="nearest")[
                        "reflectance"
                    ]

                    if "wavelength" not in self._host_map._spectral_data:
                        self._host_map._spectral_data["wavelength"] = ds[
                            "wavelength"
                        ].values
                elif self._host_map.cog_layer_dict[layer_name]["hyper"] == "PACE":
                    try:
                        da = extract_pace(ds, lat, lon)
                    except:
                        da = xr.DataArray(
                            np.full(len(ds["wavelength"]), np.nan),
                            dims=["wavelength"],
                            coords={"wavelength": ds["wavelength"]},
                        )
                    if "wavelengths" not in self._host_map._spectral_data:
                        self._host_map._spectral_data["wavelengths"] = ds[
                            "wavelength"
                        ].values

                elif self._host_map.cog_layer_dict[layer_name]["hyper"] == "DESIS":
                    da = extract_desis(ds, lat, lon)

                elif self._host_map.cog_layer_dict[layer_name]["hyper"] == "NEON":
                    da = extract_neon(ds, lat, lon)

                elif self._host_map.cog_layer_dict[layer_name]["hyper"] == "AVIRIS":
                    da = extract_aviris(ds, lat, lon)

                self._host_map._spectral_data[f"({lat:.4f} {lon:.4f})"] = da.values

                da[da < 0] = np.nan
                axes_options = {
                    "x": {"label_offset": "30px"},
                    "y": {"label_offset": "35px"},
                }

                if not stack_btn.value:
                    plt.clear()
                    plt.plot(
                        da.coords[da.dims[0]].values,
                        da.values,
                        axes_options=axes_options,
                    )
                else:
                    color = np.random.rand(
                        3,
                    )
                    plt.plot(
                        da.coords[da.dims[0]].values,
                        da.values,
                        color=color,
                        axes_options=axes_options,
                    )
                    try:
                        if isinstance(self._fig.axes[0], bqplot.ColorAxis):
                            self._fig.axes = self._fig.axes[1:]
                        elif isinstance(self._fig.axes[-1], bqplot.ColorAxis):
                            self._fig.axes = self._fig.axes[:-1]
                    except:
                        pass
                plt.xlabel("Wavelength (nm)")
                plt.ylabel("Reflectance")

                if not self._show_plot:
                    with self._output_widget:
                        plt.show()
                        self._show_plot = True

                self._host_map.default_style = {"cursor": "crosshair"}

        self._host_map.on_interaction(handle_interaction)
        self._on_map_interaction = handle_interaction

        self._spectral_widget = self
        self._spectral_control = ipyleaflet.WidgetControl(
            widget=self, position=position
        )
        self._host_map.add(self._spectral_control)

    def cleanup(self):
        """Removes the widget from the map and performs cleanup."""
        if self._host_map:
            self._host_map.default_style = {"cursor": "default"}
            self._host_map.on_interaction(self._on_map_interaction, remove=True)

            if self._output_control:
                self._host_map.remove_control(self._output_control)

                if self._output_widget:
                    self._output_widget.close()
                    self._output_widget = None

            if self._spectral_control:
                self._host_map.remove_control(self._spectral_control)
                self._spectral_control = None

                if self._spectral_widget:
                    self._spectral_widget.close()
                    self._spectral_widget = None

            if hasattr(self._host_map, "_plot_marker_cluster"):
                self._host_map._plot_marker_cluster.markers = []
                self._host_map._plot_markers = []

            if hasattr(self._host_map, "_spectral_data"):
                self._host_map._spectral_data = {}

            if hasattr(self, "_output_widget") and self._output_widget is not None:
                self._output_widget.clear_output()

        if self.on_close is not None:
            self.on_close()
