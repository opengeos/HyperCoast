# HyperCoast

[![image](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/opengeos/HyperCoast/blob/main)
[![image](https://img.shields.io/pypi/v/HyperCoast.svg)](https://pypi.python.org/pypi/HyperCoast)
[![image](https://static.pepy.tech/badge/hypercoast)](https://pepy.tech/project/hypercoast)
[![image](https://img.shields.io/conda/vn/conda-forge/hypercoast.svg)](https://anaconda.org/conda-forge/hypercoast)
[![Conda Recipe](https://img.shields.io/badge/recipe-hypercoast-green.svg)](https://github.com/conda-forge/hypercoast-feedstock)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/hypercoast.svg)](https://anaconda.org/conda-forge/hypercoast)

**A Python package for visualizing and analyzing hyperspectral data in coastal regions**

-   Free software: MIT License
-   Documentation: <https://hypercoast.org>
-

## Introduction

HyperCoast is a Python package designed to provide an accessible and comprehensive set of tools for visualizing and analyzing hyperspectral data in coastal regions. Building on the capabilities of popular packages like [Leafmap](https://leafmap.org) and [PyVista](https://pyvista.org), HyperCoast streamlines the process of exploring and interpreting complex remote sensing data. This enables researchers and environmental managers to gain deeper insights into the dynamic processes occurring in coastal environments.

HyperCoast supports the reading and visualization of hyperspectral data from various NASA airborne and satellite missions, such as [AVIRIS](https://aviris.jpl.nasa.gov), [NEON](https://data.neonscience.org/data-products/DP3.30006.001), [PACE](https://pace.gsfc.nasa.gov), and [EMIT](https://earth.jpl.nasa.gov/emit), along with other datasets like [DESIS](https://www.earthdata.nasa.gov/s3fs-public/imported/DESIS_TCloud_Mar0421.pdf) and [ECOSTRESS](https://ecostress.jpl.nasa.gov). Users can interactively explore hyperspectral data, extract spectral signatures, change band combinations and colormaps, visualize data in 3D, and perform interactive slicing and thresholding operations (see Figure 1). Additionally, leveraging the earthaccess package, HyperCoast offers tools for searching NASA hyperspectral data interactively. This makes it a versatile and powerful tool for working with hyperspectral data globally, with a particular focus on coastal regions.

![EMIT](https://assets.gishub.org/images/EMIT-demo.png)
**Figure 1.** An example of visualizing NASA EMIT hyperspectral data using HyperCoast.

## Features

-   Searching for NASA hyperspectral data interactively
-   Interactive visualization and analysis of hyperspectral data, such as [AVIRIS](https://aviris.jpl.nasa.gov), [DESIS](https://www.earthdata.nasa.gov/s3fs-public/imported/DESIS_TCloud_Mar0421.pdf), [EMIT](https://earth.jpl.nasa.gov/emit), [PACE](https://pace.gsfc.nasa.gov), [NEON AOP](https://data.neonscience.org/data-products/DP3.30006.001)
-   Interactive visualization of NASA [ECOSTRESS](https://ecostress.jpl.nasa.gov) data
-   Interactive visualization of [PACE](https://pace.gsfc.nasa.gov) chlorophyll-a data
-   Interactive extraction and visualization of spectral signatures
-   Changing band combinations and colormaps interactively
-   Visualizing hyperspectral data in 3D
-   Visualizing ERA5 temperature data in 3D
-   Interactive slicing and thresholding of hyperspectral data in 3D
-   Saving spectral signatures as CSV files

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

This project draws inspiration and adapts source code from the [nasa/EMIT-Data-Resources](https://github.com/nasa/EMIT-Data-Resources) repository. Credit goes to the original authors.
