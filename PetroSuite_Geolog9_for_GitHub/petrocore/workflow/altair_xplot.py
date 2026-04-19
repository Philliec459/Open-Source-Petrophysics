from pathlib import Path

import pandas as pd
import altair as alt

alt.data_transformers.disable_max_rows()

CHART_DIR = Path(__file__).resolve().parent / "charts_altair"

COLOR_DOMAIN   = (0, 150)
COLOR_SCHEME   = "rainbow"
INACTIVE_COLOR = "lightgray"
LITH_COLORS    = "magenta"
POR_COLORS     = "magenta"

LITH_COLORS2 = [
    "#4FC3F7", "#FFD54F", "#81C784", "#FF8A65", "#BA68C8",
    "#4DB6AC", "#F06292", "#AED581", "#9575CD", "#FFD54F"
]

POR_COLORS2 = [
    "#90CAF9", "#FFCC80", "#A5D6A7", "#F48FB1", "#CE93D8",
    "#80CBC4", "#FFAB91", "#B39DDB", "#FFF59D", "#80DEEA"
]

CAT_COLORS = [
    "#4FC3F7", "#FFD54F", "#81C784", "#FF8A65", "#BA68C8",
    "#4DB6AC", "#F06292", "#AED581", "#9575CD", "#FFB74D",
    "#64B5F6", "#4DD0E1", "#FFF176", "#E57373", "#A1887F",
    "#7986CB", "#DCE775", "#FF8A65", "#81D4FA", "#C5E1A5"
]


def _first_existing(obj, *candidate_lists):
    if obj is None:
        return None

    if hasattr(obj, "columns"):
        available = set(obj.columns)
    else:
        available = set(obj)

    for candidates in candidate_lists:
        if candidates is None:
            continue

        if isinstance(candidates, str):
            if candidates in available:
                return candidates
            continue

        for c in candidates:
            if c in available:
                return c

    return None


def _unique_cols(*cols):
    out = []
    seen = set()

    for c in cols:
        if c is None:
            continue
        if c not in seen:
            out.append(c)
            seen.add(c)

    return out


def _load_chart_df(chart_source):
    if chart_source is None:
        return None

    if isinstance(chart_source, pd.DataFrame):
        df_loaded = chart_source.copy()
        df_loaded.columns = df_loaded.columns.astype(str).str.strip()
        return df_loaded

    if isinstance(chart_source, (str, Path)):
        path = Path(chart_source)

        if not path.is_absolute():
            candidate_paths = [
                path,
                CHART_DIR / path.name,
                Path(__file__).resolve().parent / path,
            ]
            path = next((p for p in candidate_paths if p.exists()), CHART_DIR / path.name)

        if not path.exists():
            print(f"[altair_xplot] chart file not found: {path}")
            return None

        suffix = path.suffix.lower()

        try:
            if suffix in [".xlsx", ".xls"]:
                df_loaded = pd.read_excel(path)
            elif suffix == ".csv":
                df_loaded = pd.read_csv(path)
            else:
                print(f"[altair_xplot] unsupported chart file type: {path}")
                return None

            df_loaded.columns = df_loaded.columns.astype(str).str.strip()
            print(f"[altair_xplot] loaded chart file: {path}")
            print(f"[altair_xplot] columns: {list(df_loaded.columns)}")
            return df_loaded

        except Exception as e:
            print(f"[altair_xplot] failed to load chart file {path}: {e}")
            return None

    print(f"[altair_xplot] unsupported chart source type: {type(chart_source)}")
    return None


def _apply_dark_theme(chart):
    return (
        chart
        .configure_view(
            stroke=None,
            fill="#1e1e1e",
        )
        .configure(
            background="#1e1e1e",
        )
        .configure_axis(
            domainColor="#aaaaaa",
            gridColor="#444444",
            tickColor="#aaaaaa",
            labelColor="#e6e6e6",
            titleColor="#ffffff",
        )
        .configure_axisX(
            domainColor="#aaaaaa",
            gridColor="#444444",
            tickColor="#aaaaaa",
            labelColor="#e6e6e6",
            titleColor="#ffffff",
        )
        .configure_axisY(
            domainColor="#aaaaaa",
            gridColor="#444444",
            tickColor="#aaaaaa",
            labelColor="#e6e6e6",
            titleColor="#ffffff",
        )
        .configure_title(
            color="#ffffff",
            anchor="start",
        )
        .configure_legend(
            labelColor="#e6e6e6",
            titleColor="#ffffff",
            fillColor="#1e1e1e",
            strokeColor="#444444",
        )
        .configure_header(
            labelColor="#e6e6e6",
            titleColor="#ffffff",
        )
        .configure_point(
            filled=True,
        )
        .configure_range(
            category=CAT_COLORS,
            ordinal=CAT_COLORS,
        )
    )


def build_altair_dashboard(
    df,
    df_chart="cnl_chart_1pt1.xlsx",
    df_pef="PEF_Rhob_chart.xlsx",
    df_pickett="Pickett_Ro_chart.xlsx",
    top=None,
    bottom=None,
):
    print("build_altair_dashboard from: petrocore.workflow.altair_xplot")
    print("CHART_DIR:", CHART_DIR)

    if df is None or df.empty:
        empty = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_point()
        return _apply_dark_theme(empty)

    df_chart = _load_chart_df(df_chart)
    df_pef = _load_chart_df(df_pef)
    df_pickett = _load_chart_df(df_pickett)

    print("df_chart loaded:", df_chart is not None)
    print("df_pef loaded:", df_pef is not None)
    print("df_pickett loaded:", df_pickett is not None)

    depth_col = _first_existing(df, ["DEPTH", "DEPT", "Depth", "MD", "TVD", "Z"])
    gr_col    = _first_existing(df, ["GR", "GR_EDTC", "SGR"])
    rhob_col  = _first_existing(df, ["RHOB", "RHOZ", "RHO"])
    nphi_col  = _first_existing(df, ["TNPH", "NPHI", "NPOR", "CNL", "CNCF", "CNC"])
    dt_col    = _first_existing(df, ["DTCO", "DTC", "DT", "AC"])
    rt_col    = _first_existing(df, ["RT", "AF90", "AT90", "ILD", "LLD", "RD"])
    phi_col   = _first_existing(df, ["PHIT", "PHIE", "PHI"])
    pef_col   = _first_existing(df, ["PEF", "PE", "PEFZ"])

    if depth_col is None:
        raise ValueError("No depth column found in dataframe")

    work_df = df.copy()

    for col in [depth_col, gr_col, rhob_col, nphi_col, dt_col, rt_col, phi_col, pef_col]:
        if col is not None and col in work_df.columns:
            work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    work_df = work_df[work_df[depth_col].notna()].copy()


    #interval = alt.selection_interval(name="depth_interval")
    interval = alt.selection_interval()

    def _sample_color_expr(fallback_col):
        color_col = gr_col if gr_col is not None else fallback_col
        return alt.condition(
            interval,
            alt.Color(
                f"{color_col}:Q",
                scale=alt.Scale(domain=COLOR_DOMAIN, scheme=COLOR_SCHEME),
                legend=None,
            ),
            alt.value(INACTIVE_COLOR),
        )








    #------------------------------------------------
    #      Start of Altair
    #------------------------------------------------
    
    interval = alt.selection_interval()
    
    if top is None:
        top = float(work_df[depth_col].min())
    if bottom is None:
        bottom = float(work_df[depth_col].max())
    
    #------------------------------------------------
    #       Depth of Depth Track
    #------------------------------------------------
    
    base=alt.Chart(work_df).mark_point(clip=True).encode(
        alt.Y(f"{depth_col}:Q",
            scale=alt.Scale(domain=(bottom, top)))
        
    ).properties(
        width=150,
        height=600,
        #title='GR',
        #selection=interval
    ).add_params(interval)
    
    
    base2=alt.Chart(work_df).mark_point(clip=True).encode(
        alt.Y(f"{depth_col}:Q",
            scale=alt.Scale(domain=(bottom, top)), axis=alt.Axis(labels=False),title='')
    ).properties(
        width=150,
        height=600,
        title='',
    ).add_params(interval)
    
    
    
    #------------------------------------------------
    #       Log Curves of Depth Track
    #------------------------------------------------
    gr = base.mark_circle(clip=True, size=24).encode(
        x=f"{gr_col}:Q",  
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme = 'rainbow'),legend=None),
        tooltip=f"{gr_col}:Q", 
    ).properties(
        title='GR',
        #selection=interval
    )
      
    rhob = base2.mark_circle(clip=True , size=24).encode(
        alt.X(f"{rhob_col}:Q",
            scale=alt.Scale(domain=(2, 3))
        ),     
        color=alt.condition(interval, f"{gr_col}:Q" , alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip='RHOB:Q', 
    ).properties(
        title='RHOB',
        #selection=interval
    )
     
    nphi = base2.mark_circle(clip=True, size=24).encode(
        alt.X(f"{nphi_col}:Q",
            scale=alt.Scale(domain=(.45, -0.15))
        ),     
        color=alt.condition(interval, f"{gr_col}:Q" , alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip='NPHI:Q', 
    ).properties(
        title='NPHI',
        #selection=interval
    )
    
    dt = base2.mark_circle(clip=True, size= 24).encode(
        alt.X(f"{dt_col}:Q",
            scale=alt.Scale(domain=(112, 28))
        ),     
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip='DT:Q', 
    ).properties(
        title='DT',
        #selection=interval
    )
    
    rt = base2.mark_circle(clip=True, size= 24).encode(
        alt.X(f"{rt_col}:Q", 
              scale=alt.Scale(type='log', domain=(0.2, 2000.0))
        ),
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip='RT:Q', 
    ).properties(
        title='RT',
        #selection=interval
    )
    
    phit = base2.mark_circle(clip=True, size= 24).encode(
        alt.X(f"{phi_col}:Q",
            scale=alt.Scale(domain=(.45, -0.15))
        ),    
        #color=alt.value('blue'),
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip=f"{phi_col}:Q", 
    ).properties(
        title='PHIE',
        #selection=interval
    )
      
    
    #------------------------------------------------
    #       Neutron-Density Cross Plot
    #------------------------------------------------
    nd_chart = alt.Chart(work_df).mark_line().encode(
        alt.X('CNL_chart:Q',
            scale=alt.Scale(domain=(-0.05, 0.6))
        ),    
        alt.Y('RHOB_chart:Q',
            scale=alt.Scale(domain=(3, 1.9))
        ),    
        color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='rainbow'),legend=None),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    
    nd_chart2 = alt.Chart(work_df).mark_line().encode(
        alt.X('CNL_chart:Q',
            scale=alt.Scale(domain=(-0.05, 0.6))
        ),    
        alt.Y('RHOB_chart:Q',
            scale=alt.Scale(domain=(3, 1.9))
        ),    
        color=alt.condition(interval, 'Por:O', alt.value('black'),scale=alt.Scale(scheme='rainbow'),legend=None),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    
    ndxplot = base.mark_circle(clip=True,size= 24).encode(
        alt.X(f"{nphi_col}:Q",
            scale=alt.Scale(domain=(-0.05, 0.6))
        ),    
        alt.Y(f"{rhob_col}:Q",
            scale=alt.Scale(domain=(3, 1.9))
        ),    
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip=f"{rhob_col}:Q", 
    ).properties(
        title='Neut-Den Xplot (GR on Color Axis)',
        width=250,
        height=250,
        #selection=interval
    )
    
    
     
    #------------------------------------------------
    #       PEF-Density Cross Plot
    #------------------------------------------------
    pef_chart = alt.Chart(work_df).mark_line().encode(
        alt.X('PEF_chart:Q',
            scale=alt.Scale(domain=(0, 10))
        ),    
        alt.Y('RHOB_chart:Q',
            scale=alt.Scale(domain=(3, 2))
        ),    
        color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='rainbow'),legend=None),
        #color=alt.value('black'),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    
    pef_chart2 = alt.Chart(work_df).mark_line().encode(
        alt.X('PEF_chart:Q',
            scale=alt.Scale(domain=(0, 10))
        ),    
        alt.Y('RHOB_chart:Q',
            scale=alt.Scale(domain=(3, 2))
        ),    
        color=alt.condition(interval, 'Por:O', alt.value('black'),scale=alt.Scale(scheme='rainbow'),legend=None),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    
    pefxplot = base.mark_circle(clip=True, size= 24).encode(
        alt.X(f"{pef_col}:Q",
            scale=alt.Scale(domain=(0, 10))
        ),    
        alt.Y(f"{rhob_col}:Q",
            scale=alt.Scale(domain=(3, 2))
        ),    
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip=f"{pef_col}:Q", 
    ).properties(
        title='PEF-RHOB Xplot (GR on Color Axis)',
        width=250,
        height=250,
    )
    
    
    
    #------------------------------------------------
    #       Pickett Plot
    #------------------------------------------------
    pickett_chart = alt.Chart(work_df).mark_line(clip=True, size=2 ,strokeDash=[5,5]  ).encode(
        alt.X('Rt_Pickett:Q',
            scale=alt.Scale(type='log',domain=(.01, 1000))
        ),    
        alt.Y('Por_at_Ro:Q',
            scale=alt.Scale(type='log',domain=(0.01, 1.0))
        ),    
        color=alt.value('blue'),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    
    pickett_chart8 = alt.Chart(work_df).mark_line(clip=True, size=2 ,strokeDash=[5,5]).encode(
        alt.X('Rt_Pickett:Q',
            scale=alt.Scale(type='log',domain=(.01, 1000))
        ),    
        alt.Y('Por_at_0pt75:Q',
            scale=alt.Scale(type='log',domain=(0.01, 1.0))
        ),    
        #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='rainbow')),
        color=alt.value('cyan'),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    pickett_chart6 = alt.Chart(work_df).mark_line(clip=True, size=2 ,strokeDash=[5,5]  ).encode(
        alt.X('Rt_Pickett:Q',
            scale=alt.Scale(type='log',domain=(.01, 1000))
        ),    
        alt.Y('Por_at_0pt5:Q',
            scale=alt.Scale(type='log',domain=(0.01, 1.0))
        ),    
        #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='rainbow')),
        color=alt.value('yellow'),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    pickett_chart4 = alt.Chart(work_df).mark_line(clip=True, size=2 ,strokeDash=[5,5]  ).encode(
        alt.X('Rt_Pickett:Q',
            scale=alt.Scale(type='log',domain=(.01, 1000))
        ),    
        alt.Y('Por_at_0pt25:Q',
            scale=alt.Scale(type='log',domain=(0.01, 1.0))
        ),    
        #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='rainbow')),
        color=alt.value('orange'),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    pickett_chart2 = alt.Chart(work_df).mark_line(clip=True , size=2 ,strokeDash=[5,5] ).encode(
        alt.X('Rt_Pickett:Q',
            scale=alt.Scale(type='log',domain=(.01, 1000))
        ),    
        alt.Y('Por_at_0pt1:Q',
            scale=alt.Scale(type='log',domain=(0.01, 1.0))
        ),    
        color=alt.value('red'),
    ).properties(
        #title='Neut-Den Xplot with GR on Color Axis',
        width=250,
        height=250
        #selection=interval
    )
    pickett = base.mark_circle(clip=True, size= 24).encode(
        alt.X(f"{rt_col}:Q",
            scale=alt.Scale(type='log',domain=(.01, 100))
        ),    
        alt.Y(f"{phi_col}:Q",
            scale=alt.Scale(type='log',domain=(.01, 1))
        ),    
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
        tooltip = f"{rt_col}:Q", 
    ).properties(
        title='Pickett Plot (GR on Color Axis)',
        width=250,
        height=250,
        #selection=interval
    )
    
    
    
    #------------------------------------------------
    #       Histograms
    #------------------------------------------------
    grhist = alt.Chart(work_df).mark_bar(clip=True,size=5).encode(
        #alt.X("GR:Q", bin=alt.Bin(maxbins=75)),
        alt.X(f"{gr_col}:Q",
            bin=alt.Bin(maxbins=75),
            #scale=alt.Scale(domain=(0,600)),
        ),
        y='count():Q',
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),    
    ).properties(
        title='GR Hist',
        width=250,
        height=250,
        #selection=interval
    ).add_params(interval)
    
    rhobhist = alt.Chart(work_df).mark_bar(clip=True,size=5).encode(
        alt.X(f"{rhob_col}:Q",
            #bin=True,
            bin=alt.Bin(maxbins=75),
            scale=alt.Scale(domain=(2.0,3.0)),
        ),       
        y='count():Q',
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    ).properties(
        title='RHOB Hist',
        width=250,
        height=250,
    ).add_params(interval)
    
    nphihist = alt.Chart(work_df).mark_bar(clip=True,size=5).encode(
        alt.X(f"{nphi_col}:Q",
            #bin=True,
            bin=alt.Bin(maxbins=75),
            scale=alt.Scale(domain=(0.45, -0.05)),
        ),
        y='count():Q',
        color=alt.condition(interval, f"{gr_col}:Q", alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    ).properties(
        title='NPHI Hist',
        width=250,
        height=250,
    ).add_params(interval)
    
    
    #------------------------------------------------
    #
    #       Define Plot Regions for Altair
    #
    #------------------------------------------------
    
    #depth = gr | rhob | nphi |  phit | rt 
    depth = gr | rhob | nphi | rt 
    
    xplot = ndxplot+nd_chart+nd_chart2| pefxplot+pef_chart+pef_chart2 |pickett+pickett_chart+pickett_chart8+pickett_chart6+pickett_chart4  
       
    hist =  grhist | rhobhist | nphihist
    



    #dashboard = depth & xplot & hist
    dashboard = depth | xplot & hist

    return _apply_dark_theme(dashboard)