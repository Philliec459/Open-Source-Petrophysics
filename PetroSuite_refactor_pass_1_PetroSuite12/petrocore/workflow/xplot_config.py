


# petrocore/workflow/xplots/xplot_config.py

# ---------------------------------------------------------
# Chart files
# ---------------------------------------------------------
CHART_FILES = {
    1: "cnl_chart_1pt1.xlsx",
    2: "neutron-sonic.xlsx",
    3: "sonic-density.xlsx",
    4: "pef_rhob.xlsx",
    5: "umma_rhomaa.xlsx",
    6: "pota_thor.xlsx",
    7: "pota_pef.xlsx",
    8: "pickett.xlsx",
    9: "buckles.xlsx",
    10: "steiber.xlsx",
    11: "xy.xlsx",
    12: "gxy.xlsx",
    13: "pef_rhob.xlsx",
    14: "pef_rhob.xlsx",
    15: "pef_rhob.xlsx",
      
}

POINTS_UMAA = [
    ("Dolomite", 9.0, 2.87),
    ("Quartz", 4.79, 2.65),
    ("Calcite", 13.79, 2.71),
    ("Kaolinite", 7.0, 3.2),
    ("Illite", 12.5, 3.02),
    ("Anhydrite", 15.5, 2.96),
    ("Orthoclase", 9.0, 2.62),
    ("Coal", 0.0, 2.28),
]


POINTS_PEF_K = [
    ("Muscovite", 0.09, 2.2),
    ("Illite", 0.065, 3.7),
    ("Biotite", 0.082, 6.3),
    ("Kaolinite", 0.0110, 1.8),
    ("Smectite", 0.016, 2.1),
    ("Glauconite", 0.042, 6.2),
    ("Chlorite", 0.001, 6.2),
]


POINTS_POTA_THOR = [
    ("Heavy", 0.003,20),
    ("Kaolinite", 0.01,17),
    ("Smectite", 0.008,7.5),	
    ("Mixed Layer", 0.02,8),	
    ("Illite", 0.04,10),	
    ("Mica", 0.044,7),	
    ("Glauconite", 0.04,4),	
    ("Feldspar", 0.04,2),		
    ("K-evaporate", 0.04,1),

]

# ---------------------------------------------------------
# Chart definitions
# These describe the redigitized chart overlay files
# ---------------------------------------------------------
CHART_DEFS = {
    1: {
        "name": "Neutron-Density",
        "file": "cnl_chart_1pt1.xlsx",
        "y_col": "RHOB",
        "x_col": "Neutron",
        "phi_col": "Porosity",
        "extra_cols": ["Rho_Matrix"],
        "y_label": "RHOB",
        "x_label": "Neutron",
        "ylim": (1.9, 3),
        "xlim": (-0.05, 0.45),
        "reverse_y": True,
        "reverse_x": False,
    },
    2: {
        "name": "Neutron-Sonic",
        "file": "neutron-sonic.xlsx",
        "x_col": "Neutron",
        "y_col": "DT",
        "phi_col": "Porosity",
        "x_label": "Neutron",
        "y_label": "DT",
        "xlim": (-0.05, 0.45),
        "ylim": (40, 110),
        "reverse_x": False,
        "reverse_y": False,
    },
    3: {
        "name": "Sonic-Density",
        "file": "sonic-density.xlsx",
        "x_col": "DT",
        "y_col": "RHOB",
        "phi_col": "Porosity",
        "extra_cols": [],
        "x_label": "DT",
        "y_label": "RHOB",
        "xlim": (40, 130),
        "ylim": (1.8, 3),
        "reverse_x": False,
        "reverse_y": True,
    },
    4: {
        "name": "PEF-Density",
        "file": "pef_rhob.xlsx",
        "x_col": "PEF",
        "y_col": "RHOB",
        "phi_col": "Porosity",
        "extra_cols": [],
        "x_label": "PEF",
        "y_label": "RHOB",
        "xlim": (0, 10),
        "ylim": (1.9, 3),
        "reverse_x": False,
        "reverse_y": True,
    },
    5: {
        "name": "UMAA-RHOMAA",
        "file": "umma_rhomaa.xlsx",
        "x_col": "UMAA",
        "y_col": "RHOMAA",
	"points": POINTS_UMAA ,
        "phi_col": None,
        "extra_cols": [],
        "x_label": "UMAA",
        "y_label": "RHOMAA",
        "xlim": (0, 20),
        "ylim": (2.0, 3.2),
        "reverse_x": False,
        "reverse_y": True,
    },
    6:{
        "name": "POTA-THOR" ,
        "file": "pota_thor.xlsx",
        "x_col": "POTA",
        "y_col": "THOR",
        "points": POINTS_POTA_THOR ,
        "phi_col": None,
        "extra_cols": [],
        "x_label": "POTA",
        "y_label": "THOR",
        "xlim": (0, 0.05),
        "ylim": (0, 25),
        "reverse_x": False,
        "reverse_y": False,
    },
    7:{
        "name": "POTA-PEF",
        "file": "pota_pef.xlsx",
        "x_col": "HFK",
        "y_col": "PEF",
        "points": POINTS_PEF_K ,
        "phi_col": None,
        "extra_cols": [],
        "x_label": "POTA",
        "y_label": "PEF",
        "xlim": (0, 0.1),
        "ylim": (0, 10.0),
        "reverse_x": False,
        "reverse_y": False,
    },
    8:{
        "name": "Pickett Plot",
        "file": "pickett.xlsx",
        "x_col": "RT",
        "y_col": "PHIT",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "RT",
        "y_label": "PHIT",
        "xlim": (0.01, 1000),
        "ylim": (0.01, 1),
        "reverse_x": False,
        "reverse_y": False,
        "log_x": True,
        "log_y": True,
    },
    9:{
 
        "name": "Buckles Plot",
        "file": "buckles.xlsx",
        "x_col": "SW",
        "y_col": "PHIT",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "SW",
        "y_label": "PHIT",
        "xlim": (0, 1),
        "ylim": (0.0, 0.5),
        "reverse_x": False,
        "reverse_y": False,
    },
    10:{
 
        "name": "Steiber Plot",
        "file": "steiber.xlsx",
        "x_col": "VSH_HL",
        "y_col": "PHIT",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "VSH_HL",
        "y_label": "PHIT",
        "xlim": (0, 1),
        "ylim": (0, 0.5),
        "reverse_x": False,
        "reverse_y": False,
    },
    11:{
 
        "name": "Generic XY",
        "file": "xy.xlsx",
        "x_col": "X",
        "y_col": "Y",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "X",
        "y_label": "Y",
        "xlim": (0, 1),
        "ylim": (.0, 1),
        "reverse_x": False,
        "reverse_y": False,
    },
    12:{
 
        "name": "Generic Regression XY",
        "file": "gxy.xlsx",
        "x_col": "X",
        "y_col": "Y",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "X",
        "y_label": "Y",
        "xlim": (0, 1),
        "ylim": (0, 1),
        "reverse_x": False,
        "reverse_y": False,
    },
    13: {
        "name": "Histogram",
        "plot_type": "histogram",
        "title": "Histogram",
        "x_curve_candidates": ["GR"],
        "y_curve_candidates": [],
        "z_curve_candidates": [],
        "bins": 30,
    },
   14:{
        "name": "log_log",
        "file": "umma_rhomaa.xlsx",
        "x_col": "UMAA",
        "y_col": "RHOMAA",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "UMAA",
        "y_label": "RHOMAA",
        "xlim": (0, 20),
        "ylim": (2.0, 3.2),
        "reverse_x": False,
        "reverse_y": False,
        "log_x": True,
        "log_y": True,

    },
    15:{
        "name": "semilog",
        "file": "umma_rhomaa.xlsx",
        "x_col": "UMAA",
        "y_col": "RHOMAA",
        "phi_col": None,
        "extra_cols": [],
        "x_label": "UMAA",
        "y_label": "RHOMAA",
        "xlim": (0, 20),
        "ylim": (2.0, 3.2),
        "reverse_x": False,
        "reverse_y": False,
        "log_x": False,
        "log_y": True,

    },

}

# ---------------------------------------------------------
# Z-axis defaults
# Easy place for user to edit curve color scaling
# ---------------------------------------------------------
Z_AXIS_DEFAULTS = {
    "GR":   {"mode": "percentile", "pmin": 2, "pmax": 98, "scale": "linear"},
    "PHIT": {"mode": "fixed", "zmin": 0.0, "zmax": 0.35, "scale": "linear"},
    "VSH_HL":  {"mode": "fixed", "zmin": 0.0, "zmax": 1.0, "scale": "linear"},
    "SW":   {"mode": "fixed", "zmin": 0.0, "zmax": 1.0, "scale": "linear"},
    "RT":   {"mode": "percentile", "pmin": 5, "pmax": 95, "scale": "log"},
    "PERM": {"mode": "percentile", "pmin": 5, "pmax": 95, "scale": "log"},
}

Z_AXIS_FALLBACK = {
    "mode": "percentile",
    "pmin": 2,
    "pmax": 98,
    "scale": "linear",
}

# ---------------------------------------------------------
# Xplot presets
# These describe the well-log crossplot choices
# ---------------------------------------------------------
XPLOT_PRESETS = {
    "neutron_density": {
        "title": "Neutron-Density",
        "chart_number": 1,
        "y_curve_candidates": ["RHOZ", "RHOB", "DEN", "ZDEN"],
        "x_curve_candidates": ["TNPH", "NPHI", "NPOR", "CNL"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT", "SW"],
        "default_z_curve": "GR",
    },
    "neutron_sonic": {
        "title": "Neutron-Sonic",
        "chart_number": 2,
        "y_curve_candidates": ["DTCO", "DTCO3" ,"DTC", "DT"],
        "x_curve_candidates": ["TNPH", "NPHI", "NPOR", "CNL"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT", "SW"],
        "default_z_curve": "GR",
    },
    "sonic_density": {
        "title": "Sonic-Density",
        "chart_number": 3,
        "x_curve_candidates": ["DTCO", "DTCO3", "DT", "DTC"],
        "y_curve_candidates": ["RHOZ", "RHOB", "DEN", "ZDEN"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT", "SW"],
        "default_z_curve": "GR",
    },
    "pef_density": {
        "title": "PEF-Density",
        "chart_number": 4,
        "x_curve_candidates": ["PEF", "PEFZ","PEF8"],
        "y_curve_candidates": ["RHOZ", "RHOB", "DEN", "ZDEN"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "umaa_rhomaa": {
        "title": "UMAA-RHOMAA",
        "chart_number": 5,
        "x_curve_candidates": ["UMAA"],
        "y_curve_candidates": ["RHOMAA"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },

    "pota_thor": {
        "title": "POTA-THOR",
        "chart_number": 6,
        "x_curve_candidates": ["POTA", "HFK"],
        "y_curve_candidates": ["THOR", "HTHO"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR",   "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "pota_pef": {
        "title": "POTA-PEF",
        "chart_number": 7,
        "x_curve_candidates": [ "HFK", "POTA"],
        "y_curve_candidates": ["PEF", "PEFZ"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "pickett_plot": {
        "title": "Pickett Plot",
        "chart_number": 8,
        "x_curve_candidates": ["RT", "AT90", "AF90", "ILD" ,"LLD","RD"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL","GR_EDTC",  "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "buckles_plot": {
        "title": "Buckles Plot",
        "chart_number": 9,
        "x_curve_candidates": ["SW"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "steiber_plot": {
        "title": "Steiber Plot",
        "chart_number": 10,
        "x_curve_candidates": ["VSH_HL","VSH"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "generic_xy": {
        "title": "Generic Cross Plot",
        "chart_number": 11,
        "x_curve_candidates": ["VSH_HL","VSH"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "generic_regression_xy": {
        "title": "Generic Cross Plot",
        "chart_number": 12,
        "x_curve_candidates": ["VSH_HL","VSH"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "histogram": {
        "title": "Histogram",
        "plot_type": "histogram",
        "x_curve_candidates": ["GR", "NPHI", "RHOB", "DT", "PHIT", "VSH"],
        "y_curve_candidates": [],
        "z_curve_candidates": [],
        "bins": 30,
    },
    "log_log": {
        "title": "Log-Log Plot",
        "chart_number": 14,
        "x_curve_candidates": ["VSH_HL","VSH"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },
    "semilog": {
        "title": "SemiLog Plot",
        "chart_number": 15,
        "x_curve_candidates": ["VSH_HL","VSH"],
        "y_curve_candidates": ["PHIT"],
        "z_curve_candidates": ["VSH_HL", "GR_EDTC", "GR", "VSH", "PHIT"],
        "default_z_curve": "GR",
    },

}


'''
6: "POTA-THOR",
7: "POTA-PEF",
8: "Pickett Plot",
9: "Buckles Plot",
10: "Steiber Plot",
11: "Generic XY",
12: "Generic Regression XY",
13: "Histogram",
14: "log-log",
15: "semilog",
'''
