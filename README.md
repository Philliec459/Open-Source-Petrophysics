# Open-Source-Petrophysics
A comprehensive Open Source repository for Petrophysics, providing tools, scripts, and resources for analyzing subsurface data. Ideal for geoscientists, engineers, and researchers working with porosity, permeability, and other petrophysical properties. Join us in advancing the science of subsurface exploration and reservoir characterization.

![image](Interactive_Petrophysics.gif)

We have created this GitHub repository titled **Open-Source-Petrophysics** where we will be sharing specialized petrophysical tools, primarily written in Jupyter Notebooks, to document and streamline various processes. This repository will also include example datasets, and each notebook will feature links to run the code directly in Google Colab, allowing users to execute the notebooks without needing Python installed locally. 

The goal of the Open-Source-Petrophysics repository is to consolidate various petrophysical tools and scripts into a focused, well-documented resource. Initially, the repository will serve as a launchpad for tools designed with one or two specific objectives, making them easy for others to try out and build upon. Over time, we’ll also develop more comprehensive workflows, like those we have shared in this [repository](https://github.com/Philliec459/Jupyter-Notebooks_for-Characterization-of-a-New-Open-Source-Carbonate-Reservoir-Benchmarking-Case-St).

---
Below are some topics for the petrophysical tools we are developing. **Any topic highlighted in blue is a hyperlink to a python Colab-ready Jupyter Notebook accessible from that link.** These notebooks can be run directly in the Colab environment using an ordinary web browser. As we develop more notebooks, the number of blue links will increase.

Clicking a link will first open the Jupyter Notebook in GitHub. Press the 'Open in Colab' button in the upper left corner to launch the notebook in Colab. While in Colab, the notebook will automatically load the necessary data files and any required python libraries. Ensure that you are logged into your Google account by checking the upper right corner of the webpage. If not, log in before running Colab. To execute the notebook, go to the top title bar of Colab, click on the 'Runtime' tab, and select 'Run all.' The notebook will upload all necessary data files and run the Python code. The process may take a few minutes, but we've tested it in our environment, and Colab usually works fine. When finished, we suggest that in the **"Runtime"** column, click on **"Disconnect and delete runtime"** to end your CoLab session. That is it. 

## **Traditional Petrophysical Analysis**
- [Read in las file using Lasio and explore data](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Simple_Petrpphysical_Workflow/1_Read_LAS_Explore_Data.ipynb)
- [Read in las file and create a Depth Plot using matplotlib](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Simple_Petrpphysical_Workflow/2_Read_LAS_Depth_Plot.ipynb)
- [Neutron-Density Chartbook Porosity calculation using Knn](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Simple_Petrpphysical_Workflow/3_Read_LAS_Chartbook_Porosity.ipynb)
- [Lithology from Optimization](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Simple_Petrpphysical_Workflow/4_Read_LAS_Optimized_Lithology.ipynb)
- [Refine saturation calculations using interactive Pickett plots with Panel interactive widgets](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Simple_Petrpphysical_Workflow/5_Read_LAS_Log_Saturations_Pickett_Plot.ipynb) 
- [Python-based optimization for lithology estimation using the original work of Ruben Charles](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Simple_Petrpphysical_Workflow/rubens_optimization_methods.ipynb) 

## **Shaley Sand Log Analysis**
- [Shaley Sand Workflow](https://github.com/Philliec459/Geolog-Project-Notebook-for-a-Clastic-Comprehensive-Interactive-Petrophysical-Analysis-Workflow/blob/main/Conventional_NMR_Logs-las-Panel_ver2.ipynb)
    - Hodges-Lehmann Multiple Shale Indicator method used to model Clay Bound Water, PHIE and Qv. 
    - Calculate Clay Bound Water, PHIE and Qv for Waxman-Smits saturation analysis using Panel interactive widgets.

## **Permeability and Core Data Interrogation**
- [Convert routine core analysis Kair to Klinkenberg permeability](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Klinkenberg/Klinkenberg_Perm_GitHub.ipynb)
- [Create Relative Permeability curves using rock/fluid properties with Panel](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Relative_Permeability/RelPerm.ipynb)
- [View Clastic Thin Section photomicrographs from selected Porosity-Permeability samples using Altair](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Clastic_Poro-perm_Thin_Sections/Clastic_poro_perm_thinsections_Altair.ipynb)
- [View Carbonate Capillary Pressure curves from selected Porosity-Permeability samples using Altair](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Carbonate_Rosetta-Stone_Data_Interrogation/Altair_Create_Pc_Interrogation_RosettaStone.ipynb). This notebook creates the HPMI Pc curves from the Thomeer parameters prior to the interactice visualization. 
- [Interactive Thomeer parameter analysis from High Pressure Mercury Injection data](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Thomeer_from_Clerke_spreadsheet/Thomeer_from_Pc_curve_fit_auto-use-picks-Auto-Put_on_GitHub_read_Edspreadsheet_ver2.ipynb) and a Geolog Project with [Interactive Python Loglans](https://github.com/Philliec459/Geolog-Used-to-Model-Thomeer-Parameters-from-High-Pressure-Mercury-Injection-Data)
- [Chicheng Xu's normal Gaussian fit of HMPI data - Needs_Revision_to_CDF](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Gaussian_Clerke_Spreadsheet/Panel_Bvocc_ver6_GitHub_colab.ipynb)
- [Buiting Upscaled Capillary Pressure and Buiting-Clerke Upscaled Permeability](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Upscaled_BVocc_BC_Perm/SatHg_LinkedIn.ipynb)
- Evaluation of various unsupervised Cluster analysis techniques with different elbow type methods.
- Rock Typing and the creation of full pore system **Petrophysical Rock Types**.
- Permeability estimation using KNN, ...

## **Field Studies**
- [Use of Thomeer parameters to estimate well log and 3D model saturations](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Carbonate_Workflow_Costa_Field/CO3_full_workflow.ipynb)
- [Free Water Level (**FWL**) search using Capillary Pressure data](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Carbonate_Workflow_Costa_Field/CO3_full_workflow.ipynb)
- [Free Water Level search results from individual wells used to create a FWL surface for 3D modeling of saturations](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Carbonate_Workflow_Costa_Field/CO3_full_workflow.ipynb)
- Calculate Reservoir Fluid properties and IFT*cos(theta) for carbonate and clastic reservoirs characterization projects. IFT*cos(theta) for carbonates is much lower than for clastic reservoirs. 
- [Create field production maps with interactive links to well-by-well dynamic data (pressure and production)](https://github.com/Philliec459/Altair-used-to-Visualize-and-Interrogate-well-by-well-Production-Data-from-Volve-Field/blob/master/Volve_GitHub_brief.ipynb) for better understanding of the productive characteristics of your reservoir. 

## **Carbonate and Clastic Workflows**
- [Comprehensive Carbonate workflow for reservoir characterization](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/Carbonate_Workflow_Costa_Field/CO3_full_workflow.ipynb) using a new, comprehensive carbonate reservoir characterization database from Costa, Geiger and Arnold(1). This repository has 17 wells from Costa Field, and a single Jupyter Notebook that performs the following tasks:
    - Load las file using lasio
    - Visualize Data Extent using Andy McDonald's methods
    - Calculate Total Porosity from digitized Chart Book data using Knn
    - Calculate Water Saturation using Interactive Pickett Plot tool from Panel
    - Optimization of log response functions to model lithology
    - Model Permeability using kNN
    - Estimate Thomeer Capillary Pressure parameters from PHIT and Knn Permeability using Knn again
    - Perform Free Water Level (FWL) search on subject well
    - Estimate Capillary Pressure based saturations from Thomeer parameters and and height above FWL
    - Export results to Excel and las files
    - [Workflow in a Geolog Project with python Loglans](https://github.com/Philliec459/Geolog-Project-with-Interactive-Pickett-Plot-and-Lithology-Optimization-Using-Python-Loglans/tree/main/HW_Interactive)
- Interactive workflows for Waxman-Smits saturation analysis [Workflow in a Geolog Project with python Loglans](https://github.com/Philliec459/Geolog-Project-Notebook-for-a-Comprehensive-Interactive-Petrophysical-Analysis-Workflow)
- Interactive Passey methods used in unconventional well log analysis

## **NMR**
- [**Create an NMR logs from echo trains created from bin porosities and then process these echo trains with added noise for the T2 distribution**](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/NMR_echo_train_processing.ipynb) This approach provides an introductory understanding of converting time-domain echo train data into the T2 domain through T2 inversion. While the process is somewhat circular, real echo trains could be used instead. During this process, the user can introduce random noise to the echo trains, stack multiple echo trains to average out some of the noise, and then process the stacked echo trains to generate an NMR log. This method serves as an effective educational tool for NMR logging to better understand the process.
- [**Tutorial on Phase Rotation of NMR data**](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/Tutorial-curve_fit_echo_train_using_phase_rotation.ipynb)

>![image](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/Create_Echo_Phase_Correct.png)

- [**Phase Correct entire NMR logging section and Perform T2 Inversion**](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/curve_fit_echo_train_using_phase_rotation.ipynb)

>![image](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/phase_rotation.gif)

- [**T2 Inversion from actual CMR data**] which is in progress, but the dlis file is too large for GitHub.
- [**T2 Inversion from actual MRIL-Prime data**](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/curve_fit_ECHOA_MRIL_All_from_Excel-GitHub.ipynb)

>![image](https://github.com/Philliec459/Open-Source-Petrophysics/blob/main/NMR_Echo_Train_Simple_T2_Inversion_Example/nmr_animation.gif)
><img src="https://github.com/Philliec459/Open-Source-Petrophysics/raw/main/NMR_Echo_Train_Simple_T2_Inversion_Example/MRIL_customer_results.png" width=30% height=30%>

## **Miscellaneous**  
- [Interactively Interrogate Well Log data using Python's Altair](https://github.com/Philliec459/Geolog-Python-Loglan-use-of-Altair-to-Interrogate-Log-Analysis-data/blob/main/LogAnalysis_GitHub_read_las-best3_test.ipynb) with Geolog python Loglan files.

## **Additional Open Source Petrophysical Sites:**
- [SPWLA-PDDA Open Petro Data and Utilities (OPDU) Website](https://github.com/PDDA-OPDU) with their first Repository on [MICP-Analytics](https://github.com/PDDA-OPDU/MICP-Analytics)
- [Equinor DLISIO](https://github.com/equinor/dlisio) - Andy McDonald has a very good [Jupyter Notebook](https://github.com/andymcdgeo/Petrophysics-Python-Series/blob/master/17%20-%20Loading%20DLIS%20Data.ipynb) on using dlisio.
- [LASIO](https://github.com/kinverarity1/lasio)
- [Utilities – SPWLA PDDA ML Contests](https://github.com/pddasig)
- [Utilities – UTFE](https://github.com/TCDAG/AutoPetrophysics-GOM-CCS)
- [SEG](https://github.com/seg/)
- [Andy McDonald Petrophysical Website](https://github.com/andymcdgeo)

---
---
## **HOW TO RUN JUPYTER NOTEBOOKS IN COLAB:**
1) If you click on the Jupyter Notebook links that we have provided, then you will see the following type of image while you are still in GitHub. Click on the banner **"Open in CoLab"** at the top. This will then open the notebook in Google CoLab. Look to the far upper right corner of the web page to make sure that you are logged into your Google account. If not, then login before trying to run CoLab. 

![Image](GitHub_link.png)

2) On the top title bar of CoLab there is a label called **"Runtime"**(second image below). Click on this and then click on **"Run all"**. That is it. The notebook should upload all of the data files, load the necessary python libraries and run the python notebook. It could take a few minutes, but we have tested each notebook in this environment, and CoLab works well most of the time. If you have any issues, then click on **Run all** again. That should take care of the problem unless the CoLab network is extremely busy. 

![Image](CoLab_link.png)

3) When finished, then we suggest in the **"Runtime"** tab, click on **"Disconnect and delete runtime"** to end your CoLab session. This will delete all of the data and terminate the runtime. 


---
---
## REFERENCES:
1.  Xu, C., Torres-Verdín, C. Pore System Characterization and Petrophysical Rock Classification Using a Bimodal Gaussian Density Function. Math Geoscience 45, 753–771 (2013). https://doi.org/10.1007/s11004-013-9473-2
2.  Costa Gomes J, Geiger S, Arnold D. The Design of an Open-Source Carbonate Reservoir Model. Petroleum Geoscience, 
    https://doi.org/10.1144/petgeo2021-067
3.  Phillips, E. C., Buiting, J. M., Clerke, E. A, “Full Pore System Petrophysical Characterization Technology for Complex Carbonate Reservoirs – Results from Saudi Arabia”, AAPG, 2009 Extended Abstract.
4.  Clerke, E. A., Mueller III, H. W., Phillips, E. C., Eyvazzadeh, R. Y., Jones, D. H., Ramamoorthy, R., Srivastava, A., (2008) “Application of Thomeer Hyperbolas to decode the pore systems, facies and reservoir properties of the Upper Jurassic Arab D Limestone, Ghawar field, Saudi Arabia: A Rosetta Stone approach”, GeoArabia, Vol. 13, No. 4, p. 113-160, October 2008.
5.  J.M. Buiting, Fully Upscaled Saturation-Height Functions for Reservoir Modeling Based on Thomeer's Method for Analyzing Capillary Pressure Measurements, SPE Paper 105139, 2007.
6.  J.J.M. Buiting, E.A. Clerke, Permeability from Porosimetry Measurements: Derivation for a Tortuous and Fractal Tubular Bundle, Journal of Petroleum Science and Engineering, 108 (2013), 267–278.
7. Raheem, O., Morales, M., Saputra, W., Torres-Verdín, C., Phillips, C., Xu, C., “Universal Data-Driven Permeability Modeling by Connecting MICP Analytics with Big Data,” SPWLA 2025.
