# Open-Source-Petrophysics
A comprehensive Open Source repository for Petrophysics, providing tools, scripts, and resources for analyzing subsurface data. Ideal for geoscientists, engineers, and researchers working with porosity, permeability, and other petrophysical properties. Join us in advancing the science of subsurface exploration and reservoir characterization.

We have created this GitHub repository titled **Open-Source-Petrophysics** where we will be sharing specialized petrophysical tools, primarily written in Jupyter Notebooks, to document and streamline various processes. This repository will also include example datasets, and each notebook will feature links to run the code directly in Google Colab, allowing users to execute the notebooks without needing Python installed locally. 

We recently conducted a course using Colab, and it performed admirably. You can explore some of the Colab examples from that course on this [GitHub repository](https://github.com/Philliec459/Launchpad-for-STS-Processing-of-STELLA-Spectrometer-Landsat-and-PACE-Ocean-Data), where the notebooks can be launched directly from the README.md page. While Colab is a great tool, there are other options like Binder, Kaggle, SageMaker, and Docker that might be even better for certain use cases.

The goal of the Open-Source-Petrophysics repository is to consolidate various petrophysical tools and scripts into a focused, well-documented resource. Initially, the repository will serve as a launchpad for tools designed with one or two specific objectives, making them easy for others to try out and build upon. Over time, we’ll also develop more comprehensive workflows, like those we have shared in this [repository](https://github.com/Philliec459/Jupyter-Notebooks_for-Characterization-of-a-New-Open-Source-Carbonate-Reservoir-Benchmarking-Case-St).

Please find below some ideas for the tools we'll be developing, Any topic in blue have a CoLab-ready Jupyter Notebook available from that link. The number of blue links will increase with time as we develop more notebooks. 

**Petrophysical Analysis**
- [Read in las file, explore data](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/1_Read_LAS_Explore_Data.ipynb)
- [Read in las file and create a Depth Plot](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/2_Read_LAS_Depth_Plot.ipynb)
- [Neutron-Density Chartbook Porosity calculation using Knn](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/3_Read_LAS_Chartbook_Porosity.ipynb)
- Saturation determinations with interactive Pickett Plots using Panel
- Clay Bound Water and Qv calculation for Waxman-Smits saturation analysis with Panel interactive widgets
- [Ruben's Python-based optimization for lithology estimation](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/rubens_optimization_methods.ipynb)

**Permeability and Core Data Interrogation**
- Conversion of routine core analysis Kair to Klinkenberg permeability
- Creation of Relative Permeability curves using rock/fluid properties with Panel
- Permeability estimation using KNN, ...
- [View Clastic Thin Section photomicrographs from selected Porosity-Permeability samples using Altair](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Clastic_poro_perm_thinsections_Altair.ipynb)
- [View Carbonate Capillary Pressure curves from selected Porosity-Permeability samples using Altair](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Altair_Interrogation_RosettaStone.ipynb)
- Determine Thomeer parameters from High Pressure Mercury Injection (HPMI) data using interactive optimization
- Evaluation of various unsupervised Cluster analysis techniques with different elbow type methods being employed
- Rock Typing and the creation of full pore system **Petrophysical Rock Types**

**Field Studies**
- Use of Thomeer parameters to estimate well log and 3D model saturations
- Free Water Level (**FWL**) search using Capillary Pressure data
- Free Water Level search results from individual wells are then used to create a FWL surface for our 3D models
- Calculate Reservoir Fluid properties and IFT*cos(theta) for carbonate and clastic reservoirs characterization projects. 
- Field maps with interactive links to well-by-well dynamic data (reservoir pressure, production) for improved field productivity analysis

**Workflows**
- Comprehensive workflow for clastic reservoir characterization
- Comprehensive workflow for carbonate reservoir characterization
- Interactive workflows for Waxman-Smits saturation analysis
- Interactive Passey methods used in unconventional well log analysis

**Miscellaneous**
- [Creation of NMR logs from echo trains using Python](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_echo_train_processing.ipynb)

---
---
## **HOW TO RUN STS JUPYTER NOTEBOOKS IN COLAB:**
1) If you click on the Jupyter Notebook links that we have provided, you will see the following type of image while you are still in GitHub. Click on the banner **"Open in CoLab"** at the top, and this will open this notebook in Google CoLab. Look to the far upper right corner of the web page to make sure that you are logged into your google account. If not, then login before trying to run CoLab. 

![Image](GitHub_link.png)

2) On the top title bar of CoLab there is a label called **"Runtime"**(second image below). Click on this and then click on **"Run all"**. That is it. The notebook will get all of the data files and run python. It could take a few minutes, but I have tested this in my environment, and it works fine.

![Image](CoLab_link.png)

3) When finished, then I would suggest in the **"Runtime"** column, click on **"Disconnect and delete runtime"** to end your CoLab session. 

