"""This module contains the user interface for the hypercoast package.
"""

import ipyleaflet
import ipywidgets as widgets


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

    def __init__(self, host_map, position="topright"):
        """
        Initializes a new instance of the SpectralWidget class.

        Args:
            host_map (Map): The map to host the widget.
            position (str, optional): The position of the widget on the map. Defaults to "topright".
        """
        self._host_map = host_map
        self.on_close = None

        close_btn = widgets.Button(
            icon="times",
            tooltip="Close the widget",
            button_style="primary",
            layout=widgets.Layout(width="32px"),
        )

        def close_widget(_):
            self.cleanup()

        close_btn.on_click(close_widget)

        layer_names = list(host_map.cog_layer_dict.keys())
        layers_widget = widgets.Dropdown(options=layer_names)
        layers_widget.layout.width = "18ex"
        super().__init__([layers_widget, close_btn])

        output = widgets.Output()
        output_control = ipyleaflet.WidgetControl(widget=output, position="bottomright")
        self._output_widget = output
        self._output_control = output_control
        self._host_map.add(output_control)

        def handle_interaction(**kwargs):

            latlon = kwargs.get("coordinates")
            if kwargs.get("type") == "click":
                with self._output_widget:
                    self._output_widget.clear_output()
                    print(latlon)

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

        if self.on_close is not None:
            self.on_close()
