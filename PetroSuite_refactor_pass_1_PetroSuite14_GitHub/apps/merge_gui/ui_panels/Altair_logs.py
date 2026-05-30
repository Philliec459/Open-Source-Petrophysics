#!/usr/bin/env python3
# python loglan

# Imports


'''
  select the proper Neutron-Density Chartbook file
'''

#file = r'./data/cnl_chart_1pt0.xlsx'
file = r'./data/cnl_chart_1pt1.xlsx'
#file = r'./data/tnph_chart_1pt0.xlsx'
#file = r'./data/tnph_chart_1pt19.xlsx'
df_chart = pd.read_excel(file,index_col=False)
#df_chart.head()


'''
  PEF vs. RHOB overlay
'''

#Load PEF vs. RHOB overlay chart
file = r'./data/PEF_Rhob_chart.xlsx'
df_pef = pd.read_excel(file,index_col=False)
df_pef.head()


'''    
  Select the proper Pickett data
  Adjust the Rw and m in the file designated below.
'''

file = r'./data/Pickett_Ro_chart.xlsx'
df_pickett = pd.read_excel(file,index_col=False)
#df_pickett.head()


'''    
  Select log curves for DataFrame called df
'''
#df = pd.DataFrame({'DEPTH':depth, 'RHOB':rhob, 'NPHI':nphi, 'ILD':ild, 'GR':gr, 'PHIT':phit,'PEF':pef, 'DT':dt })

#read the file
file = r'./data/main_well.xlsx'
df_log = pd.read_excel(file,index_col=False)

'''
#------------------------------------------------
#      Start of Altair
#------------------------------------------------
'''


interval = alt.selection_interval()

#bottom = max(df_log['DEPTH'])
#top = min(df_log['DEPTH'])

#------------------------------------------------
#       Depth of Depth Track
#------------------------------------------------

base=alt.Chart(work_df).mark_point(clip=True).encode(
    alt.Y('DEPTH:Q',
        scale=alt.Scale(domain=(bottom, top)))
    
).properties(
    width=150,
    height=600,
    #title='GR',
    #selection=interval
).add_params(interval)


#no depth labels for base2      
base2=alt.Chart(work_df).mark_point(clip=True).encode(
    alt.Y('DEPTH:Q',
        scale=alt.Scale(domain=(bottom, top)), axis=alt.Axis(labels=False),title='')
).properties(
    width=150,
    height=600,
    title='',
).add_params(interval)



#------------------------------------------------
#       Log Curves of Depth Track
#------------------------------------------------
gr = base.mark_circle(clip=True, size=50).encode(
    x='GR:Q',  
    #size=('PHIX:Q'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(domain=(0, 75),scheme='rainbow'),legend=None),
    #color=alt.condition(selector, 'Well_Name:O', alt.value('lightgray'), legend=None),
    tooltip='GR:Q', 
).properties(
    title='GR',
    #selection=interval
)
   
rhob = base2.mark_circle(clip=True , size=50).encode(
    alt.X('RHOB:Q',
        scale=alt.Scale(domain=(2, 3))
    ),     
    #color=alt.value('red'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    #color=alt.condition(selector, 'Well_Name:O', alt.value('lightgray'), legend=None),
    tooltip='RHOB:Q', 
).properties(
    title='RHOB',
    #selection=interval
)
 
nphi = base2.mark_circle(clip=True, size=50).encode(
    alt.X('NPHI:Q',
        scale=alt.Scale(domain=(.45, -0.15))
    ),     
    #y=('DEPTH'),
    #color=alt.value('green'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='NPHI:Q', 
).properties(
    title='NPHI',
    #selection=interval
)

dt = base2.mark_circle(clip=True, size=50).encode(
    alt.X('DT:Q',
        scale=alt.Scale(domain=(112, 28))
    ),     
    #y=('DEPTH'),
    #color=alt.value('green'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='DT:Q', 
).properties(
    title='DT',
    #selection=interval
)

rt = base2.mark_circle(clip=True, size=50).encode(
    alt.X('RT:Q', 
          scale=alt.Scale(type='log', domain=(0.2, 2000.0))
    ),
    #x='LRT:Q',  
    #y=('DEPTH'),
    #color=alt.value('black'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='RT:Q', 
).properties(
    title='RT',
    #selection=interval
)

phit = base2.mark_circle(clip=True, size=50).encode(
    alt.X('PHIE:Q',
        scale=alt.Scale(domain=(.45, -0.15))
    ),    
    #color=alt.value('blue'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='PHIE:Q', 
).properties(
    title='PHIE',
    #selection=interval
)
  

#------------------------------------------------
#       Neutron-Density Cross Plot
#------------------------------------------------
nd_chart = alt.Chart(df_chart).mark_line().encode(
    alt.X('CNL_chart:Q',
        scale=alt.Scale(domain=(-0.05, 0.6))
    ),    
    alt.Y('RHOB_chart:Q',
        scale=alt.Scale(domain=(3, 1.9))
    ),    
    color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow'),legend=None),
    #color=alt.value('black'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)

nd_chart2 = alt.Chart(df_chart).mark_line().encode(
    alt.X('CNL_chart:Q',
        scale=alt.Scale(domain=(-0.05, 0.6))
    ),    
    alt.Y('RHOB_chart:Q',
        scale=alt.Scale(domain=(3, 1.9))
    ),    
    color=alt.condition(interval, 'Por:O', alt.value('black'),scale=alt.Scale(scheme='rainbow'),legend=None),
    #color=alt.value('black'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)

ndxplot = base.mark_circle(clip=True,size=50).encode(
    alt.X('NPHI:Q',
        scale=alt.Scale(domain=(-0.05, 0.6))
    ),    
    alt.Y('RHOB:Q',
        scale=alt.Scale(domain=(3, 1.9))
    ),    
    #x='NPHI:Q',  
    #y=('RHOB'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='RHOB:Q', 
).properties(
    title='Neut-Den Xplot (GR on Color Axis)',
    width=250,
    height=250,
    #selection=interval
)


 
#------------------------------------------------
#       PEF-Density Cross Plot
#------------------------------------------------
pef_chart = alt.Chart(df_pef).mark_line().encode(
    alt.X('PEF_chart:Q',
        scale=alt.Scale(domain=(0, 10))
    ),    
    alt.Y('RHOB_chart:Q',
        scale=alt.Scale(domain=(3, 2))
    ),    
    color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow'),legend=None),
    #color=alt.value('black'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)

pef_chart2 = alt.Chart(df_pef).mark_line().encode(
    alt.X('PEF_chart:Q',
        scale=alt.Scale(domain=(0, 10))
    ),    
    alt.Y('RHOB_chart:Q',
        scale=alt.Scale(domain=(3, 2))
    ),    
    color=alt.condition(interval, 'Por:O', alt.value('black'),scale=alt.Scale(scheme='rainbow'),legend=None),
    #color=alt.value('black'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)

pefxplot = base.mark_circle(clip=True, size=50).encode(
    alt.X('PEF:Q',
        scale=alt.Scale(domain=(0, 10))
    ),    
    alt.Y('RHOB:Q',
        scale=alt.Scale(domain=(3, 2))
    ),    
    #x='NPHI:Q',  
    #y=('RHOB'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='PEF:Q', 
).properties(
    title='PEF-RHOB Xplot (GR on Color Axis)',
    width=250,
    height=250,
    #selection=interval
)



#------------------------------------------------
#       Pickett Plot
#------------------------------------------------
pickett_chart = alt.Chart(df_pickett).mark_line(clip=True, size=2 ,strokeDash=[5,5]  ).encode(
    alt.X('Rt_Pickett:Q',
        scale=alt.Scale(type='log',domain=(.01, 1000))
    ),    
    alt.Y('Por_at_Ro:Q',
        scale=alt.Scale(type='log',domain=(0.01, 1.0))
    ),    
    #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow')),
    color=alt.value('blue'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)

pickett_chart8 = alt.Chart(df_pickett).mark_line(clip=True, size=2 ,strokeDash=[5,5]).encode(
    alt.X('Rt_Pickett:Q',
        scale=alt.Scale(type='log',domain=(.01, 1000))
    ),    
    alt.Y('Por_at_0pt75:Q',
        scale=alt.Scale(type='log',domain=(0.01, 1.0))
    ),    
    #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow')),
    color=alt.value('cyan'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)
pickett_chart6 = alt.Chart(df_pickett).mark_line(clip=True, size=2 ,strokeDash=[5,5]  ).encode(
    alt.X('Rt_Pickett:Q',
        scale=alt.Scale(type='log',domain=(.01, 1000))
    ),    
    alt.Y('Por_at_0pt5:Q',
        scale=alt.Scale(type='log',domain=(0.01, 1.0))
    ),    
    #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow')),
    color=alt.value('yellow'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)
pickett_chart4 = alt.Chart(df_pickett).mark_line(clip=True, size=2 ,strokeDash=[5,5]  ).encode(
    alt.X('Rt_Pickett:Q',
        scale=alt.Scale(type='log',domain=(.01, 1000))
    ),    
    alt.Y('Por_at_0pt25:Q',
        scale=alt.Scale(type='log',domain=(0.01, 1.0))
    ),    
    #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow')),
    color=alt.value('orange'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)
pickett_chart2 = alt.Chart(df_pickett).mark_line(clip=True , size=2 ,strokeDash=[5,5] ).encode(
    alt.X('Rt_Pickett:Q',
        scale=alt.Scale(type='log',domain=(.01, 1000))
    ),    
    alt.Y('Por_at_0pt1:Q',
        scale=alt.Scale(type='log',domain=(0.01, 1.0))
    ),    
    #color=alt.condition(interval, 'Lith:O', alt.value('black'),scale=alt.Scale(scheme='sinebow')),
    color=alt.value('red'),
).properties(
    #title='Neut-Den Xplot with GR on Color Axis',
    width=250,
    height=250
    #selection=interval
)
pickett = base.mark_circle(clip=True, size=50).encode(
    alt.X('RT:Q',
        scale=alt.Scale(type='log',domain=(.01, 100))
    ),    
    alt.Y('PHIE:Q',
        scale=alt.Scale(type='log',domain=(.01, 1))
    ),    
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
    tooltip='RT:Q', 
).properties(
    title='Pickett Plot (GR on Color Axis)',
    width=250,
    height=250,
    #selection=interval
)



#------------------------------------------------
#       Histograms
#------------------------------------------------
grhist = alt.Chart(df_log).mark_bar(clip=True,size=5).encode(
    #alt.X("GR:Q", bin=alt.Bin(maxbins=75)),
    alt.X('GR:Q',
        bin=alt.Bin(maxbins=75),
        scale=alt.Scale(domain=(0,50)),
    ),
    y='count():Q',
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),    
).properties(
    title='GR Hist',
    width=250,
    height=250,
    #selection=interval
).add_params(interval)

rhobhist = alt.Chart(df_log).mark_bar(clip=True,size=5).encode(
    #alt.X("RHOB:Q", bin=alt.Bin(maxbins=75)),
    alt.X('RHOB:Q',
        #bin=True,
        bin=alt.Bin(maxbins=75),
        scale=alt.Scale(domain=(2.0,3.0)),
    ),       
    y='count():Q',
    #color=alt.value('red'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
).properties(
    title='RHOB Hist',
    width=250,
    height=250,
    #selection=interval
).add_params(interval)

nphihist = alt.Chart(df_log).mark_bar(clip=True,size=5).encode(
    #alt.X("NPHI:Q",  bin=alt.Bin(maxbins=75)),
    alt.X('NPHI:Q',
        #bin=True,
        bin=alt.Bin(maxbins=75),
        scale=alt.Scale(domain=(0.45, -0.05)),
    ),
    y='count():Q',
    #color=alt.value('green'),
    color=alt.condition(interval, 'GR:Q', alt.value('lightgray'),scale=alt.Scale(scheme='rainbow'),legend=None),
).properties(
    title='NPHI Hist',
    width=250,
    height=250,
    #selection=interval
).add_params(interval)


#------------------------------------------------
#
#       Define Plot Regions for Altair
#
#------------------------------------------------

depth = gr | rhob | nphi |  phit | rt 

xplot = ndxplot+nd_chart+nd_chart2| pefxplot+pef_chart+pef_chart2 |pickett+pickett_chart+pickett_chart8+pickett_chart6+pickett_chart4  
   
hist =  grhist | rhobhist | nphihist

#plot = depth & xplot & hist
#depth & xplot & hist

depth & xplot & hist

#plot.show()

#plot.show()
