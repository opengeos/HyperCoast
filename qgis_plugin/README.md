# HyperCoast QGIS Plugin

A QGIS plugin for visualizing and analyzing hyperspectral data, including EMIT, PACE, DESIS, NEON, AVIRIS, PRISMA, EnMAP, Tanager, and Wyvern datasets.

## Features

- **Load Multiple Hyperspectral Formats**: Support for NASA EMIT, PACE, DESIS, NEON AOP, AVIRIS, PRISMA, EnMAP, Planet Tanager, Wyvern, and generic GeoTIFF/NetCDF hyperspectral data.

- **Band Combination Visualization**: Easily change RGB band combinations using wavelength values. Includes presets for:
  - True Color (RGB)
  - Color Infrared (CIR)
  - False Color (Urban)
  - Agriculture
  - Vegetation Analysis
  - Water Bodies
  - Geology
  - Chlorophyll-a

- **Interactive Spectral Inspector**: Click anywhere on the map to extract and display spectral signatures. Features include:
  - Stack multiple spectra for comparison
  - Auto-scaling and manual axis range control
  - Export to CSV
  - Export plots as PNG/PDF/SVG

- **Wavelength-based Selection**: Select bands by wavelength (nm) rather than band index for intuitive visualization.

## Requirements

### Python Dependencies

**Recommended**: Install the hypercoast library for best compatibility with all data formats:

```bash
pip install hypercoast
```

This will install all required dependencies automatically.

Alternatively, you can install dependencies manually:
```bash
pip install xarray h5netcdf rasterio h5py numpy matplotlib leafmap
```

Optional dependencies for specific data formats:
```bash
pip install rioxarray scipy pandas
```

> **Note**: The plugin works best when hypercoast is installed, as it provides properly tested data loaders for all supported hyperspectral formats (EMIT, PACE, DESIS, NEON, etc.). Without hypercoast, the plugin will use simplified fallback loaders that may not handle all edge cases.

### QGIS Version

- QGIS 3.22 or later

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

| Sensor | Extensions | Description |
|--------|------------|-------------|
| EMIT | .nc, .nc4 | NASA EMIT L2A Reflectance |
| PACE | .nc, .nc4 | NASA PACE OCI L2 AOP |
| DESIS | .nc, .tif | DESIS Hyperspectral |
| NEON | .h5 | NEON AOP Hyperspectral |
| AVIRIS | .nc, .img, .bil | AVIRIS/AVIRIS-NG |
| PRISMA | .he5, .nc | ASI PRISMA |
| EnMAP | .nc, .tif | DLR EnMAP |
| Tanager | .h5 | Planet Tanager |
| Wyvern | .tif, .tiff | Wyvern Hyperspectral |
| Generic | .tif, .nc | Multi-band GeoTIFF/NetCDF |

## Screenshots

*Coming soon*

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
