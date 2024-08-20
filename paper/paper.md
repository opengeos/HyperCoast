---
title: "HyperCoast: A Python Package for Visualizing and Analyzing Hyperspectral Data in Coastal Environments"
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

HyperCoast is a Python package designed to provide an accessible and comprehensive set of tools for visualizing and analyzing hyperspectral data in coastal environments. Hyperspectral data refers to the information collected by sensors that capture light across a wide range of wavelengths, beyond what the human eye can see. This data allows scientists to detect and analyze various materials and conditions on the Earth's surface with great detail. Unlike multispectral data, which captures light in a limited number of broad wavelength bands (typically 3 to 10), hyperspectral data captures light in many narrow, contiguous wavelength bands, often numbering in the hundreds. This provides much more detailed spectral information. Leveraging the capabilities of popular packages like Leafmap [@Wu2021] and PyVista [@Sullivan2019], HyperCoast streamlines the exploration and interpretation of complex hyperspectral remote sensing data from existing spaceborne and airborne missions. It is also poised to support future hyperspectral missions, such as NASA's SBG and GLIMR [@Dierssen2021].

HyperCoast supports the reading and visualization of hyperspectral data from various missions, including the Airborne Visible/Infrared Imaging Spectrometer (AVIRIS) [@Green1998], the National Ecological Observatory Network (NEON) Airborne Observation Platform (AOP) [@Kampe2010], the Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission [@Gorman2019], the Earth Surface Mineral Dust Source Investigation (EMIT) [@Green2021], and the DLR Earth Sensing Imaging Spectrometer (DESIS) [@Alonso2019], along with other datasets like the ECOsystem Spaceborne Thermal Radiometer Experiment on Space Station (ECOSTRESS) [@Fisher2020]. Users can interactively explore hyperspectral data, extract spectral signatures, change band combinations and colormaps, visualize data in 3D, and perform interactive slicing and thresholding operations (see Figure 1). Additionally, by leveraging the earthaccess [@barrett2024] package, HyperCoast provides tools for interactively searching NASA's hyperspectral data. This makes HyperCoast a versatile and powerful tool for working with hyperspectral data globally, with a particular focus on coastal regions.

![EMIT](https://assets.gishub.org/images/EMIT-demo.png)
**Figure 1.** An example of visualizing NASA EMIT hyperspectral data using HyperCoast.

# Statement of Need

Coastal systems, characterized by complex physical, chemical, and bio-optical processes [@Liu2019], play a crucial role in connecting terrestrial landscapes with marine ecosystems [@Pringle2001]. These systems have undergone significant anthropogenic modifications [@Elliott2007] and are particularly vulnerable to the impacts of climate change [@Junk2013]. This diverse array of stressors underscores the increasing need to enhance monitoring techniques and capabilities. Hyperspectral views of coastal systems provide significantly greater spectral details for characterizing biodiversity, habitats, water quality, and both natural and anthropogenic hazards, such as oil spills and harmful algal blooms (HABs).

The launch of new hyperspectral sensors, such as NASA's Ocean Color Instrument (OCI) aboard the Plankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission [@Gorman2019], marks a transformative era in global hyperspectral data acquisition. These sensors feature narrow spectral bands ranging from ultraviolet to near-infrared and offer 2-day global coverage. Focusing on inland-coastal applications, the Earth Surface Mineral Dust Source Investigation (EMIT) instrument serves as a precursor to the Surface Biology Geology (SBG) mission, combining high spectral resolution (380-2500 nm with a spectral resolution of 7.4 nm) and spatial resolution (60 m). EMIT provides significant hyperspectral advantages for monitoring water quality and biodiversity across diverse habitats (see Figure 1) [@Green2021; @Thompson2020].

However, effectively working with and visualizing diverse hyperspectral data, such as PACE's swath data, poses significant challenges, especially for non-expert users. Currently, there are few Python packages dedicated to hyperspectral data visualization and analysis. HyperSpy [@De_La_Pena2017], for example, is widely used for such analysis but is not tailored for new hyperspectral data (e.g., PACE, EMIT) or coastal applications. Additionally, it does not leverage the latest advances in the Jupyter ecosystem and 3D visualization. Thus, existing packages are often limited in functionality or lack a focus on coastal applications.

HyperCoast fills this gap by providing a comprehensive set of tools tailored to the unique needs of researchers and environmental managers working in coastal regions. By integrating advanced visualization techniques and interactive tools, HyperCoast enables users to effectively analyze hyperspectral data, facilitating a better understanding and management of coastal ecosystems (see Figure 2).

![PACE](https://assets.gishub.org/images/PACE-demo.png)
**Figure 2.** An example of mapping chlorophyll-a concentration using NASA PACE hyperspectral data with HyperCoast.

# Acknowledgements

The HyperCoast project draws inspiration from the [nasa/EMIT-Data-Resources](https://github.com/nasa/EMIT-Data-Resources) repository. Credit goes to the original authors. We also acknowledge the NASA EMIT program support through grant no. 80NSSC24K0865. We also acknowledge contributions from community contributors, including [Arfy Slowy](https://github.com/slowy07), [Guillermo E. Ponce-Campos](https://github.com/gponce-ars), and [Alex Leith](https://github.com/alexgleith).

# References
