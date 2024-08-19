# Open-Source-Petrophysics
A comprehensive Open Source repository for Petrophysics, providing tools, scripts, and resources for analyzing subsurface data. Ideal for geoscientists, engineers, and researchers working with porosity, permeability, and other petrophysical properties. Join us in advancing the science of subsurface exploration and reservoir characterization.

We have created this GitHub repository titled **Open-Source-Petrophysics** where we will be sharing specialized petrophysical tools, primarily written in Jupyter Notebooks, to document and streamline various processes. This repository will also include example datasets, and each notebook will feature links to run the code directly in Google Colab, allowing users to execute the notebooks without needing Python installed locally. 

We recently conducted a course using Colab, and it performed admirably. You can explore some of the Colab examples from that course on this [GitHub repository](https://github.com/Philliec459/Launchpad-for-STS-Processing-of-STELLA-Spectrometer-Landsat-and-PACE-Ocean-Data), where the notebooks can be launched directly from the README.md page. While Colab is a great tool, there are other options like Binder, Kaggle, SageMaker, and Docker that might be even better for certain use cases.

The goal of the Open-Source-Petrophysics repository is to consolidate various petrophysical tools and scripts into a focused, well-documented resource. Initially, the repository will serve as a launchpad for tools designed with one or two specific objectives, making them easy for others to try out and build upon. Over time, we’ll also develop more comprehensive workflows, like those we have shared in this [repository](https://github.com/Philliec459/Jupyter-Notebooks_for-Characterization-of-a-New-Open-Source-Carbonate-Reservoir-Benchmarking-Case-St).

Please find below some topics for the petrophysial tools that we will be developing. Any topic in blue will have a CoLab-ready Jupyter Notebook available from that link. The number of blue links will increase with time as we develop more notebooks. 

By pressing the link, the available notebooks will first open in GitHub. Press the upper left "Open in CoLab" button, and this will direct you to this Jupyter Notebook now oepn in CoLab. While in CoLab, each notebook will load the necessary data files and any required python libraries for this environment. **Look to the far upper right corner of the web page to make sure that you are logged into your google account.** If not, then log into your Google account before trying to run CoLab. On the top title bar of CoLab there is a tab called **"Runtime"**. Click on this tab and then click on **"Run all"**. The notebook will upload all of the necessary data files and run python. The process could take a few minutes, but we have tested this in our environment, and CoLab usually works fine. When finished, we suggest that in the **"Runtime"** column, click on **"Disconnect and delete runtime"** to end your CoLab session. That is it. 

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
- [Create an NMR logs from echo trains created from bin porosities and then process these echo trains with added noise for the T2 distribution.](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_echo_train_processing.ipynb) You can add any amount of random noise, stack the echo trains to reduce some of this noise and then process the echo trains to create an NMR log. This method will give you a flavor of Time domain to T2 domain T2 inversion. 

---
---
## **HOW TO RUN STS JUPYTER NOTEBOOKS IN COLAB:**
1) If you click on the Jupyter Notebook links that we have provided, then you will see the following type of image while you are still in GitHub. Click on the banner **"Open in CoLab"** at the top. This will then open the notebook in Google CoLab. Look to the far upper right corner of the web page to make sure that you are logged into your Google account. If not, then login before trying to run CoLab. 

![Image](GitHub_link.png)

2) On the top title bar of CoLab there is a label called **"Runtime"**(second image below). Click on this and then click on **"Run all"**. That is it. The notebook should upload all of the data files, load the necessary python libraries and run the python notebook. It could take a few minutes, but we have tested each notebook in this environment, and CoLab works well most of the time. If you have any issues, then click on **Run all** again. That should take care of the problem unless the CoLab network is extremely busy. 

![Image](CoLab_link.png)

3) When finished, then we suggest in the **"Runtime"** tab, click on **"Disconnect and delete runtime"** to end your CoLab session. This will delete all of the data and terminate the runtime. 

