---
title: "HyperCoast: A Python package for visualizing and analyzing hyperspectral data in coastal regions"
tags:
    - Python
    - geospatial
    - hyperspectral
    - mapping
    - Jupyter
    - visualization
    - pyvista
authors:
    - name: Bingqing Liu
      orcid: 0000-0003-4651-6996
      affiliation: "1"
    - name: Qiusheng Wu
      orcid: 0000-0001-5437-4073
      affiliation: "2"
affiliations:
    - name: School of Geosciences, University of Louisiana at Lafayette, Lafayette, LA 70504, United States
      index: 1
    - name: Department of Geography & Sustainability, University of Tennessee, Knoxville, TN 37996, United States
      index: 2
date: 10 July 2024
bibliography: paper.bib
---

# Summary

HyperCoast is a Python package designed to provide an accessible and comprehensive set of tools for visualizing and analyzing hyperspectral data in coastal regions. Building on the capabilities of popular packages like Leafmap [@Wu2021] and PyVista [@Sullivan2019], HyperCoast streamlines the process of exploring and interpreting complex remote sensing data. This enables researchers and environmental managers to gain deeper insights into the dynamic processes occurring in coastal environments [@Liu2021; @Liu2023].

HyperCoast supports the reading and visualization of hyperspectral data from various NASA airborne and satellite missions, such as AVIRIS [@Green1998], NEON [@Kampe2010], PACE [@Gorman2019], and EMIT [@Green2021], along with other datasets like DESIS [@Alonso2019] and ECOSTRESS [@Fisher2020]. Users can interactively explore hyperspectral data, extract spectral signatures, change band combinations and colormaps, visualize data in 3D, and perform interactive slicing and thresholding operations. Additionally, leveraging the earthaccess [@barrett2024] package, HyperCoast offers tools for searching NASA hyperspectral data interactively. This makes it a versatile and powerful tool for working with hyperspectral data globally, with a particular focus on coastal regions.

# Statement of Need

Coastal regions are dynamic and complex environments that require advanced remote sensing techniques to monitor and understand their various geophysical, biological, and biogeochemical processes [@Klemas2010; @McCarthy2017]. Hyperspectral remote sensing data, with its high spectral resolution, offers valuable insights into the composition and characteristics of coastal environments [@Brando2003; @Bioucas-Dias2013].

The launch of new hyperspectral sensors, such as NASA's Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission [@Gorman2019] and Earth Surface Mineral Dust Source Investigation (EMIT) [@Green2021], provides unprecedented opportunities to study coastal regions with high spatial and spectral resolutions. These sensors capture detailed information about water quality, benthic habitats, and biogeochemical processes in coastal waters, which are essential for monitoring and managing these sensitive ecosystems.

However, effectively working with and visualizing hyperspectral data poses significant challenges, especially for non-expert users. HyperCoast addresses this gap by offering a user-friendly and powerful Python package that simplifies the exploration and analysis of hyperspectral data in coastal regions.

Currently, very few Python packages are available for hyperspectral data visualization and analysis. One such package, HyperSpy [@De_La_Pena2017], is widely used for hyperspectral data analysis but is not specifically designed for new hyperspectral sensors or coastal applications. Additionally, it does not leverage the latest advances in the Jupyter ecosystem and 3D visualization. Existing packages are often limited in functionality or lack a focus on coastal applications.

HyperCoast fills this gap by providing a comprehensive set of tools tailored to the unique needs of researchers and environmental managers working in coastal regions. By integrating advanced visualization techniques and interactive tools, HyperCoast enables users to effectively analyze hyperspectral data, facilitating better understanding and management of coastal ecosystems.

# Acknowledgements

TBA

# References
