# HyperCoast QGIS Plugin

A QGIS plugin for visualizing and analyzing hyperspectral data, including EMIT, PACE, DESIS, NEON, AVIRIS, PRISMA, EnMAP, Tanager, and Wyvern datasets.

## Video Tutorials

Check out this [short video demo](https://youtu.be/EEUAC5BxqtM) and [full video tutorial](https://youtu.be/RxDUcfv-vBc) on how to use the HyperCoast plugin in QGIS.

[![youtube-video](https://github.com/user-attachments/assets/d7bb977a-c523-485f-8ffb-3c99cb4e89d3)](https://youtu.be/RxDUcfv-vBc)

## Features

-   **Load Multiple Hyperspectral Formats**: Support for NASA EMIT, PACE, DESIS, NEON AOP, AVIRIS, PRISMA, EnMAP, Planet Tanager, Wyvern, and generic GeoTIFF/NetCDF hyperspectral data.

-   **Band Combination Visualization**: Easily change RGB band combinations using wavelength values. Includes presets for:

    -   True Color (RGB)
    -   Color Infrared (CIR)
    -   False Color (Urban)
    -   Agriculture
    -   Vegetation Analysis
    -   Water Bodies
    -   Geology
    -   Chlorophyll-a

Before using the plugin, please create a new conda environment to install QGIS and HyperCoast:

```bash
conda create -n geo python=3.12
conda activate geo
conda install -c conda-forge qgis hypercoast
```

After the installation, you can start QGIS by running the following command:

```bash
conda run qgis
```

### QGIS Version

-   QGIS 3.22 or later

## Installation

### From ZIP File

1. Download the plugin ZIP file
2. In QGIS, go to **Plugins** → **Manage and Install Plugins**
3. Click **Install from ZIP**
4. Select the downloaded ZIP file
5. Click **Install Plugin**

### Manual Installation

1. Copy the `hypercoast_qgis` folder to your QGIS plugins directory:

    - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
    - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
    - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`

2. Restart QGIS

3. Enable the plugin in **Plugins** → **Manage and Install Plugins** → **Installed**

## Usage

### Loading Hyperspectral Data

1. Click the **Load Hyperspectral Data** button in the toolbar
2. Browse to your hyperspectral data file
3. Select the data type (or use auto-detect)
4. Configure RGB wavelengths for visualization
5. Click **Load Data**

### Changing Band Combinations

1. Click the **Band Combination** button
2. Select the hyperspectral layer
3. Adjust R, G, B wavelengths using spinboxes or sliders
4. Use presets for quick common combinations
5. Click **Apply**

### Inspecting Spectral Signatures

1. Click the **Spectral Inspector** button to activate the tool
2. Click on any location within a hyperspectral layer
3. The spectral plot window will show the extracted spectrum
4. Click multiple locations to compare spectra
5. Export data to CSV or save the plot image

## Supported File Formats

| Sensor  | Extensions      | Description               |
| ------- | --------------- | ------------------------- |
| EMIT    | .nc, .nc4       | NASA EMIT L2A Reflectance |
| PACE    | .nc, .nc4       | NASA PACE OCI L2 AOP      |
| DESIS   | .nc, .tif       | DESIS Hyperspectral       |
| NEON    | .h5             | NEON AOP Hyperspectral    |
| AVIRIS  | .nc, .img, .bil | AVIRIS/AVIRIS-NG          |
| PRISMA  | .he5, .nc       | ASI PRISMA                |
| EnMAP   | .nc, .tif       | DLR EnMAP                 |
| Tanager | .h5             | Planet Tanager            |
| Wyvern  | .tif, .tiff     | Wyvern Hyperspectral      |
| Generic | .tif, .nc       | Multi-band GeoTIFF/NetCDF |

## Screenshots

_Coming soon_

## Development

### Building and Installing with Python Script (Cross-Platform)

The recommended way to package and install the plugin:

```bash
cd qgis_plugin

# Package the plugin (creates dist/hypercoast_qgis_0.1.0.zip)
python install_plugin.py

# Package and install directly to QGIS
python install_plugin.py --install

# Force reinstall (overwrites existing)
python install_plugin.py --install --force

# Show plugin and system information
python install_plugin.py --info

# Uninstall from QGIS
python install_plugin.py --uninstall
```

### Building with Shell Script (Linux/macOS)

Alternative using the shell script:

```bash
cd qgis_plugin
./install_plugin.sh
```

### Manual Packaging

To create a distributable ZIP file manually:

```bash
cd qgis_plugin
zip -r hypercoast_qgis.zip hypercoast_qgis -x "*.pyc" -x "__pycache__/*"
```

### Running Tests

```bash
cd qgis_plugin
python -m pytest tests/
```

## License

This plugin is part of the HyperCoast project and is licensed under the MIT License.
