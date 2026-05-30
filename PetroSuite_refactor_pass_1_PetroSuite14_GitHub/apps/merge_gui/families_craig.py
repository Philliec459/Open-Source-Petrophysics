




# -----------------------------
# 4) Candidate lists (families)
# -----------------------------
dept_cands = ["DEPT","DEPTH"]

gr_cands   = ["GR_EDTC", "SGR", "GR", "HSGR"]
cgr_cands  = ["HCGR", "CGR"]
sp_cands   = ["SP"]

cali_cand  =  ["CALI", "CALI1", "HCAL", "CALS", "CAL", "C1"]
bs_cand    =  ["BS", "BIT", "BITSIZE"]

rhob_cands = ["RHOZ", "RHOB","DEN", "RHO8"]
tnph_cands = ["TNPH", "NPHI", "CNL", "NPOR"]
dtco_cand  = ["DTCO", "DTC", "DT", "AC"]
pef_cands  = ["PEFZ","PEF8","PEF","PE"]

rt_cands   = ["AT90", "AF90", "ILD", "LLD", "RT", "RESD", "RDEP"]


tcmr_cands = ["PHIT_NMR", "TCMR", "MPHIS"]
cmrp_cands = ["PHIE_NMR", "CMRP_3MS", "CMRP3MS", "CMRP", "MPHI"]
bvie_cands = ["BVIE", "BVI_E"]
cbw_cands  = ["CBW"]
ffi_cands  = ["FFI", "CMFF"]



# -----------------------------
# 5) Pick curves (family winners)
# -----------------------------
dept_curve  = first_present(analysis_df.columns, dept_cands)  # should be DEPT after our rename


# WIRE SET
gr_curve    = first_present(analysis_df.columns, gr_cands)
cgr_curve   = first_present(analysis_df.columns, cgr_cands)
sp_curve    = first_present(analysis_df.columns, sp_cands)
cali_curve  = first_present(analysis_df.columns, cali_cands)
bs_curve    = first_present(analysis_df.columns, bs_cands)

rhob_curve  = first_present(analysis_df.columns, rhob_cands)
tnph_curve  = first_present(analysis_df.columns, tnph_cands)
dtco_curve  = first_present(analysis_df.columns, dtco_cands)
pef_curve   = first_present(analysis_df.columns, pef_cands)

rt_curve    = first_present(analysis_df.columns, rt_cands)


#N MR_SET
tcmr_curve  = first_present(analysis_df.columns, tcmr_cands)
cmrp_curve  = first_present(analysis_df.columns, cmrp_cands)
cbw_curve   = first_present(analysis_df.columns, cbw_cands)
bvie_curve  = first_present(analysis_df.columns, bvie_cands)
ffi_curve   = first_present(analysis_df.columns, ffi_cands)

# ANALYSIS SET
phit_curve  = first_present(analysis_df.columns, phit)
phie_curve  = first_present(analysis_df.columns, phir)
bvwe        = first_present(analysis_df.columns, bvwe)























print("\nCurves that could be used in our analysis:")
print()
print(f"  Density : {rhob_curve}")
print(f"  Neutron : {tnph_curve}")
print(f"  Rt      : {rt_curve}")
print(f"  GR      : {gr_curve}")
print(f"  CGR      : {cgr_curve}")
print(f"  TCMR    : {tcmr_curve}")
print(f"  CMRP    : {cmrp_curve}")
print(f"  CBW     : {cbw_curve}")
print(f"  BVIE    : {bvie_curve}")
print(f"  FFI     : {ffi_curve}")
print(f"  PEF     : {pef_curve}")
print(f"  DTCO     : {dtco_curve}")
print(f"  DEPT     : {dept_curve}")








