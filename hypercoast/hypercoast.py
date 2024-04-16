"""Main module."""

import leafmap


class Map(leafmap.Map):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def search_emit(self, default_datset="EMITL2ARFL"):
        """Search for Earth Engine datasets."""
        self.add("nasa_earth_data", default_dataset=default_datset)

    def search_pace(self, default_datset="PACE_OCI_L2_AOP_NRT"):
        """Search for Earth Engine datasets."""
        self.add("nasa_earth_data", default_dataset=default_datset)
