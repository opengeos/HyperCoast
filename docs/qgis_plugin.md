# QGIS Plugin

[![QGIS](https://img.shields.io/badge/QGIS-plugin-orange.svg)](https://plugins.qgis.org/plugins/hypercoast)


HyperCoast provides a QGIS plugin for visualizing and analyzing hyperspectral data, including EMIT, PACE, DESIS, NEON, AVIRIS, PRISMA, EnMAP, Tanager, and Wyvern datasets.

## Video Tutorials

Check out this [short video demo](https://youtu.be/EEUAC5BxqtM) and [full video tutorial](https://youtu.be/RxDUcfv-vBc) on how to use the HyperCoast plugin in QGIS.

[![youtube-video](https://github.com/user-attachments/assets/d7bb977a-c523-485f-8ffb-3c99cb4e89d3)](https://youtu.be/RxDUcfv-vBc)

## Features

-   Load NASA EMIT, PACE, DESIS, NEON AOP, AVIRIS, PRISMA, EnMAP, Planet Tanager, Wyvern, and generic GeoTIFF or NetCDF hyperspectral data.
-   Search Planet Tanager STAC scenes from QGIS, add footprint layers, open visual imagery, and download HDF5 assets for hyperspectral analysis.
-   Change RGB band combinations by wavelength and use presets such as true color, color infrared, agriculture, vegetation, water, geology, and chlorophyll-a.
-   Extract and compare spectral signatures by clicking hyperspectral layers on the map.
-   Use dockable QGIS panels and batch-friendly Processing tools.

## Installation

The plugin is available from the official QGIS plugin repository:

1. Open QGIS.
2. Go to **Plugins** > **Manage and Install Plugins**.
3. Search for **HyperCoast**.
4. Click **Install Plugin**.

For development installation, packaging, and troubleshooting details, see the [QGIS plugin README](https://github.com/opengeos/HyperCoast/tree/main/qgis_plugin).

## Usage

### Searching Tanager Data

![](https://github.com/user-attachments/assets/f59327cd-3ea4-43ee-8a15-2af32b85c6fb)

1. Click the **Search Tanager Data** button.
2. Use the current map extent, draw a bounding box, or enter coordinates manually.
3. Set optional date, collection, query, cloud, and count filters.
4. Click **Search**.
5. Add footprints, open the visual asset, or download an HDF5 asset for analysis.

### Loading Hyperspectral Data

![](https://github.com/user-attachments/assets/b9f388f9-8cbc-4f83-b905-4b7c0ba78bef)

1. Click the **Load Hyperspectral Data** button in the toolbar.
2. Browse to your hyperspectral data file.
3. Select the data type, or use auto-detect.
4. Configure RGB wavelengths for visualization.
5. Click **Load Data**.

### Changing Band Combinations

![](https://github.com/user-attachments/assets/ae452a32-eafb-4487-ac50-861b84ffb7fb)

1. Click the **Band Combination** button.
2. Select the hyperspectral layer.
3. Adjust R, G, B wavelengths using spinboxes or sliders.
4. Use presets for quick common combinations.
5. Click **Apply**.

### Inspecting Spectral Signatures

![](https://github.com/user-attachments/assets/a0f0582a-2973-4761-b0ec-e4345461e836)

1. Click the **Spectral Inspector** button.
2. Click a location within a hyperspectral layer.
3. Compare spectra in the spectral plot panel.
4. Export data to CSV or save the plot image.

### Running Processing Tools

Open **Processing** > **Toolbox** > **HyperCoast** to run batch tools for RGB composites, single-band export, spectral index rasters, and PCA components.

## Supported File Formats

Sensor | Extensions | Description
--- | --- | ---
EMIT | .nc, .nc4 | NASA EMIT L2A Reflectance
PACE | .nc, .nc4 | NASA PACE OCI L2 AOP
DESIS | .nc, .tif | DESIS Hyperspectral
NEON | .h5 | NEON AOP Hyperspectral
AVIRIS | .nc, .img, .bil | AVIRIS/AVIRIS-NG
PRISMA | .he5, .nc | ASI PRISMA
EnMAP | .nc, .tif | DLR EnMAP
Tanager | .h5 | Planet Tanager
Wyvern | .tif, .tiff | Wyvern Hyperspectral
Generic | .tif, .nc | Multi-band GeoTIFF/NetCDF
