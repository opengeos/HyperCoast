# HyperCoast

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->

[![All Contributors](https://img.shields.io/badge/all_contributors-7-orange.svg?style=flat-square)](#contributors-)

<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/HyperCoast/blob/main)
[![image](https://img.shields.io/pypi/v/HyperCoast.svg)](https://pypi.python.org/pypi/HyperCoast)
[![image](https://static.pepy.tech/badge/hypercoast)](https://pepy.tech/project/hypercoast)
[![image](https://img.shields.io/conda/vn/conda-forge/hypercoast.svg)](https://anaconda.org/conda-forge/hypercoast)
[![Conda Recipe](https://img.shields.io/badge/recipe-hypercoast-green.svg)](https://github.com/conda-forge/hypercoast-feedstock)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/hypercoast.svg)](https://anaconda.org/conda-forge/hypercoast)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.07025/status.svg)](https://doi.org/10.21105/joss.07025)
[![QGIS Plugin](https://img.shields.io/badge/QGIS-Plugin-green.svg)](https://plugins.qgis.org/plugins/hypercoast)

![](https://assets.gishub.org/images/hypercoast_logo_600.png)

**A Python Package for Visualizing and Analyzing Hyperspectral Data in Coastal Environments**

-   Free software: MIT License
-   Documentation: <https://hypercoast.org>

## Introduction

HyperCoast is a Python package designed to provide an accessible and comprehensive set of tools for visualizing and analyzing hyperspectral data in coastal environments. Hyperspectral data refers to the information collected by sensors that capture light across a wide range of wavelengths, beyond what the human eye can see. This data allows scientists to detect and analyze various materials and conditions on the Earth's surface with great detail. Unlike multispectral data, which captures light in a limited number of broad wavelength bands (typically 3 to 10), hyperspectral data captures light in many narrow, contiguous wavelength bands, often numbering in the hundreds. This provides much more detailed spectral information. Leveraging the capabilities of popular packages like [Leafmap](https://leafmap.org) and [PyVista](https://pyvista.org), HyperCoast streamlines the exploration and interpretation of complex hyperspectral remote sensing data from existing spaceborne and airborne missions. It is also poised to support future hyperspectral missions, such as NASA's SBG and GLIMR. It enables researchers and environmental managers to gain deeper insights into the dynamic processes occurring in aquatic environments.

HyperCoast supports the reading and visualization of hyperspectral data from various missions, including [AVIRIS](https://aviris.jpl.nasa.gov), [NEON](https://data.neonscience.org/data-products/DP3.30006.001), [PACE](https://pace.gsfc.nasa.gov), [EMIT](https://earth.jpl.nasa.gov/emit), [DESIS](https://www.earthdata.nasa.gov/s3fs-public/imported/DESIS_TCloud_Mar0421.pdf), [PRISMA](https://www.asi.it/en/earth-science/prisma/) and [ENMAP](https://www.enmap.org/) along with other datasets like [ECOSTRESS](https://ecostress.jpl.nasa.gov). Users can interactively explore hyperspectral data, extract spectral signatures, change band combinations and colormaps, visualize data in 3D, and perform interactive slicing and thresholding operations (see Figure 1). Additionally, by leveraging the [earthaccess](https://github.com/nsidc/earthaccess) package, HyperCoast provides tools for interactively searching NASA's hyperspectral data. This makes HyperCoast a versatile and powerful tool for working with hyperspectral data globally, with a particular focus on coastal regions.

![EMIT](https://assets.gishub.org/images/EMIT-demo.png)
**Figure 1.** An example of visualizing NASA EMIT hyperspectral data using HyperCoast.

## Citations

If you find HyperCoast useful in your research, please consider citing the following papers to support us. Thank you!

-   Liu, B., & Wu, Q. (2024). HyperCoast: A Python Package for Visualizing and Analyzing Hyperspectral Data in Coastal Environments. _Journal of Open Source Software_, 9(100), 7025. <https://doi.org/10.21105/joss.07025>.

## Features

-   Searching for NASA hyperspectral data interactively
-   Performing atmospheric correction using [Acolite](https://github.com/acolite/acolite)
-   Interactive visualization and analysis of hyperspectral data, such as [AVIRIS](https://aviris.jpl.nasa.gov), [DESIS](https://www.earthdata.nasa.gov/s3fs-public/imported/DESIS_TCloud_Mar0421.pdf), [EMIT](https://earth.jpl.nasa.gov/emit), [PACE](https://pace.gsfc.nasa.gov), [NEON AOP](https://data.neonscience.org/data-products/DP3.30006.001), [Tanager](https://www.planet.com/data/stac/browser/tanager-core-imagery/catalog.json), [PRISMA](https://www.asi.it/en/earth-science/prisma/) and [ENMAP](https://www.enmap.org/)
-   Interactive visualization of NASA [ECOSTRESS](https://ecostress.jpl.nasa.gov) data
-   Interactive visualization of [PACE](https://pace.gsfc.nasa.gov) chlorophyll-a data
-   Interactive extraction and visualization of spectral signatures
-   Changing band combinations and colormaps interactively
-   Visualizing hyperspectral data in 3D
-   Visualizing ERA5 temperature data in 3D
-   Interactive slicing and thresholding of hyperspectral data in 3D
-   Saving spectral signatures as CSV files

## QGIS Plugin

HyperCoast also provides a QGIS plugin for visualizing and analyzing hyperspectral data, including EMIT, PACE, DESIS, NEON, AVIRIS, PRISMA, EnMAP, Tanager, and Wyvern datasets.

To install the QGIS plugin, please follow the instructions in the [QGIS Plugin README](https://github.com/opengeos/HyperCoast/tree/main/qgis_plugin).

Check out this [short video demo](https://youtu.be/EEUAC5BxqtM) and [full video tutorial](https://youtu.be/RxDUcfv-vBc) on how to use the HyperCoast plugin in QGIS.

[![youtube-video](https://github.com/user-attachments/assets/d7bb977a-c523-485f-8ffb-3c99cb4e89d3)](https://youtu.be/RxDUcfv-vBc)

## Demos

-   Visualizing hyperspectral data in 3D ([notebook](https://hypercoast.org/examples/image_cube))

![Cube](https://i.imgur.com/NNId1Zz.gif)

-   Interactive slicing of hyperspectral data in 3D ([notebook](https://hypercoast.org/examples/image_slicing))

![Slicing](https://i.imgur.com/msK1liO.gif)

-   Interactive thresholding of hyperspectral data in 3D ([notebook](https://hypercoast.org/examples/image_slicing))

![Slicing](https://i.imgur.com/TPd20Tn.gif)

-   Visualizing ERA5 temperature data in 3D ([notebook](https://hypercoast.org/examples/temperature))

![ERA5](https://i.imgur.com/qaKkmKX.gif)

-   Changing band combinations and colormaps interactively ([notebook](https://hypercoast.org/examples/neon))

![colormap](https://i.imgur.com/jYItN4D.gif)

-   Visualizing NASA [AVIRIS](https://aviris.jpl.nasa.gov) hyperspectral data interactively ([notebook](https://hypercoast.org/examples/aviris))

![AVIRIS](https://i.imgur.com/RdegGqx.gif)

-   Visualizing [DESIS](https://www.earthdata.nasa.gov/s3fs-public/imported/DESIS_TCloud_Mar0421.pdf) hyperspectral data interactively ([notebook](https://hypercoast.org/examples/desis))

![DESIS](https://i.imgur.com/PkwOPN5.gif)

-   Visualizing NASA [EMIT](https://earth.jpl.nasa.gov/emit) hyperspectral data interactively ([notebook](https://hypercoast.org/examples/emit))

![EMIT](https://i.imgur.com/zeyABMq.gif)

-   Visualizing NASA [PACE](https://pace.gsfc.nasa.gov) hyperspectral data interactively ([notebook](https://hypercoast.org/examples/pace))

![PACE](https://i.imgur.com/HBMjW6o.gif)

-   Visualizing [NEON AOP](https://data.neonscience.org/data-products/DP3.30006.001) hyperspectral data interactively ([notebook](https://hypercoast.org/examples/neon))

![NEON](https://i.imgur.com/CNP8E3y.gif)

-   Interactive visualization of [PACE](https://pace.gsfc.nasa.gov) chlorophyll-a data ([notebook](https://hypercoast.org/examples/chlorophyll_a))

![Chla](https://i.imgur.com/6hP6OFD.png)

## Acknowledgement

The HyperCoast project draws inspiration from the [nasa/EMIT-Data-Resources](https://github.com/nasa/EMIT-Data-Resources) repository. Credits to the original authors. We also acknowledge the NASA EMIT program support through grant no. 80NSSC24K0865.

## License

HyperCoast is released under the MIT License. However, some of the modules in HyperCoast adapt code from other open-source projects, which may have different licenses. Please refer to the license notice in each module for more information. Credits to the original authors.

-   [emit.py](https://github.com/opengeos/HyperCoast/blob/main/hypercoast/emit.py): Part of the code is adapted from the [nasa/EMIT-Data-Resources](https://github.com/nasa/EMIT-Data-Resources) repository, which is released under the Apache License 2.0.
-   [aviris.py](https://github.com/opengeos/HyperCoast/blob/main/hypercoast/aviris.py): Part of the code is adapted from the [jjmcnelis/aviris-ng-notebooks](https://github.com/jjmcnelis/aviris-ng-notebooks), which is released under the MIT License.

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/bingqing-liu"><img src="https://avatars.githubusercontent.com/u/123585527?v=4?s=100" width="100px;" alt="Bingqing Liu"/><br /><sub><b>Bingqing Liu</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/commits?author=bingqing-liu" title="Code">💻</a> <a href="#design-bingqing-liu" title="Design">🎨</a> <a href="#ideas-bingqing-liu" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://gishub.org"><img src="https://avatars.githubusercontent.com/u/5016453?v=4?s=100" width="100px;" alt="Qiusheng Wu"/><br /><sub><b>Qiusheng Wu</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/commits?author=giswqs" title="Code">💻</a> <a href="#design-giswqs" title="Design">🎨</a> <a href="#maintenance-giswqs" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://auspatious.com"><img src="https://avatars.githubusercontent.com/u/3445853?v=4?s=100" width="100px;" alt="Alex Leith"/><br /><sub><b>Alex Leith</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/commits?author=alexgleith" title="Code">💻</a> <a href="https://github.com/opengeos/HyperCoast/pulls?q=is%3Apr+reviewed-by%3Aalexgleith" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://slowy-portofolio-website.vercel.app/"><img src="https://avatars.githubusercontent.com/u/40540262?v=4?s=100" width="100px;" alt="arfy slowy"/><br /><sub><b>arfy slowy</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/commits?author=slowy07" title="Code">💻</a> <a href="#maintenance-slowy07" title="Maintenance">🚧</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://tucson.ars.ag.gov"><img src="https://avatars.githubusercontent.com/u/20215136?v=4?s=100" width="100px;" alt="Guillermo E. Ponce-Campos"/><br /><sub><b>Guillermo E. Ponce-Campos</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/commits?author=gponce-ars" title="Code">💻</a> <a href="https://github.com/opengeos/HyperCoast/issues?q=author%3Agponce-ars" title="Bug reports">🐛</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://www.hereon.de"><img src="https://avatars.githubusercontent.com/u/2676409?v=4?s=100" width="100px;" alt="Carsten Lemmen"/><br /><sub><b>Carsten Lemmen</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/pulls?q=is%3Apr+reviewed-by%3Aplatipodium" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://advait-0.github.io"><img src="https://avatars.githubusercontent.com/u/99654265?v=4?s=100" width="100px;" alt="Advait Dhamorikar"/><br /><sub><b>Advait Dhamorikar</b></sub></a><br /><a href="https://github.com/opengeos/HyperCoast/commits?author=advait-0" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
