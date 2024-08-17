# Open-Source-Petrophysics
A comprehensive Open Source repository for Petrophysics, providing tools, scripts, and resources for analyzing subsurface data. Ideal for geoscientists, engineers, and researchers working with porosity, permeability, and other petrophysical properties. Join us in advancing the science of subsurface exploration and reservoir characterization.

We have created this GitHub repository titled **Open-Source-Petrophysics** where we will be sharing specialized petrophysical tools, primarily written in Jupyter Notebooks, to document and streamline various processes. This repository will also include example datasets, and each notebook will feature links to run the code directly in Google Colab, allowing users to execute the notebooks without needing Python installed locally. 

We recently conducted a course using Colab, and it performed admirably. You can explore some of the Colab examples from that course on this [GitHub repository](https://github.com/Philliec459/Launchpad-for-STS-Processing-of-STELLA-Spectrometer-Landsat-and-PACE-Ocean-Data), where the notebooks can be launched directly from the README.md page. While Colab is a great tool, there are other options like Binder, Kaggle, SageMaker, and Docker that might be even better for certain use cases.

The goal of the Open-Source-Petrophysics repository is to consolidate various petrophysical tools and scripts into a focused, well-documented resource. Initially, the repository will serve as a launchpad for tools designed with one or two specific objectives, making them easy for others to try out and build upon. Over time, we’ll also develop more comprehensive workflows, like those we have shared in this [repository](https://github.com/Philliec459/Jupyter-Notebooks_for-Characterization-of-a-New-Open-Source-Carbonate-Reservoir-Benchmarking-Case-St).

Here are some ideas for the tools we'll be developing:

**Petrophysical Analysis**
- Neutron-Density Chartbook Porosity calculation using KNN
- Saturation determinations with interactive Pickett Plots using Panel
- Clay Bound Water and Qv calculation for Waxman-Smits saturation analysis with Panel interactive widgets
- Python-based optimization for lithology estimation

**Permeability and Core Data Interrogation**
- Conversion of routine core analysis Kair to Klinkenberg permeability
- Creation of Relative Permeability curves using rock/fluid properties with Panel
- Permeability estimation using KNN, etc.
- Visualization of thin section photomicrographs from Porosity-Permeability cross plots using Altair
- Extraction of Thomeer parameters from High Pressure Mercury Injection (HPMI) data using optimization
- Evaluation of various unsupervised Cluster analysis techniques
- Rock Typing and the creation of comprehensive pore system **Petrophysical Rock Types**

**Field Studies**
- Use of Thomeer parameters to estimate well log and 3D model saturations
- Free Water Level (**FWL**) search using Capillary Pressure data
- Free Water Level search on individual wells used to create a FWL surface in 3D models
- Calculation of Reservoir Fluid properties and IFT*cos(theta) for carbonate and clastic reservoirs
- Field maps with interactive links to well-by-well dynamic data for improved field productivity analysis

**Workflows**
- Comprehensive workflow for clastic reservoir characterization
- Comprehensive workflow for carbonate reservoir characterization
- Interactive workflows for Waxman-Smits saturation analysis

**Miscellaneous**
- Creation of NMR logs from echo trains using Python
