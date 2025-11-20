#!/usr/bin/env python
# coding: utf-8

# In[13]:


from pathlib import Path
import numpy as np

from DASXCPick import (
    PickerParams,
    pick_all_traces,
    save_picks_csv,
    plot_section_with_picks,
    pick_masters_interactive,
    plot_overview_with_quality,
)
from helpers import  qc_waveform_picks


# # EXAMPLE of semi-automatic picking with DAS data
# 
# `DASXPICK` is a semi-automatic cross-correlation picker. What does this mean? 
# 
# - You input your DAS data (preferably as 2D arrays) of e.g. shots or events.
# - You will pick 'master picks': the pick and the data around your pick will be used as a template
# - This template is cross-correlated with the next few traces to find your pick on those traces
# - The more times you click, the more templates you get
# - So if you have very variable data or poor SNR, then you do many clicks
# - If you have very clean data you can get away with less clicks
# 
# In principle this example is for picking first arrivals / P-waves. However, of course it can be tuned to S-waves too. For that you could either window out your S-phases separately or just pick with this same script. 

# ### User Settings

# In[ ]:





# In[ ]:


# Sampling / geometry
ns = 450
sr = 20000.0     # Hz
dt = 1.0 / sr    # s
dx = 0.2         # m between channels

# Event & paths
evname = "S18"
borehole = "D3"
phase = "P"
base = Path("./example_data")
path_csv = base / f"stacked_shots/denoised_csv/{evname}_{borehole}_denoised.csv"

# Output path for your pick csvs
out_csv = base / f"picks/{evname}_{borehole}_{phase}.csv"
make_plot = True

# Picker parameters (tune as needed)
params = PickerParams(
    temp_length=55,             # Template length (samples), depends on signal wavelength
    window_ms=1.35,             # Correlation window length (ms), this is the time window it uses to look for the pick (set small if the data is complicated)
    quality_thresh=0.45,        # Minimum CC to accept a pick: lower SNR? you have to lower this value
    use_nearest_template=True,  # Use template repository from clicks
    phase_only=False,           # Use phase-only correlation
    band=(100.0, 3000.0),       # Bandpass filter (optional, if not put None)
    detrend=True,               # Signal detrending prior to picking
    zero_phase=True,            # Zero-phase filtering for band (optional, if band is not None)
    smoothing_window=21,        # Smoothing window for the interpolated picks
    smoothing_poly=2,           # Smoothing polynomial order for the interpolated picks
    upsample=4,                 # upsample factor for sub-sample accuracy   
)


# In[ ]:


# ------------------------------- Load data -----------------------------------
print(f"Loading CSV: {path_csv}")
evt_raw = np.loadtxt(path_csv, delimiter=';', skiprows=1)
assert evt_raw.ndim == 2, "Input must be 2D (traces x samples)."

if ns is not None and ns > 0 and ns <= evt_raw.shape[1]:
    evt = evt_raw[:, :ns]
else:
    evt = evt_raw

# Master picks — leave empty to work interactively
xPe_m = []  # e.g., [2.0, 8.0, 14.0]
tPe_s = []  # e.g., [0.010, 0.011, 0.012]

# ------------------------- Interactive master picks --------------------------
if len(xPe_m) == 0 or len(tPe_s) == 0:
    print("\n>>> No masters provided — interactive picking window will pop out…")
    import matplotlib
    matplotlib.use("TkAgg")  # ensure external pop-out window in notebooks
    xPe_m, tPe_s = pick_masters_interactive(evt, dx, dt, n_required=None,
                                            vmin=-0.002, vmax=0.002)

if len(xPe_m) < 2:
    raise RuntimeError("Need at least two master picks to interpolate a moveout.")

# ------------------------------- Run picker ----------------------------------
print("Running picker…")
res = pick_all_traces(
    evt=evt,
    dx=dx,
    dt=dt,
    xPe_m=xPe_m,
    tPe_s=tPe_s,
    params=params,
)

# ------------------------------- Save ----------------------------------------
out_csv.parent.mkdir(parents=True, exist_ok=True)
depth_axis_m = np.arange(evt.shape[0]) * dx
save_picks_csv(str(out_csv), depth_axis_m, res.picks_seconds, ld_seconds=res.ld_P_seconds)
print(f"Saved picks to: {out_csv}")

# ------------------------------- Plot (opt) ----------------------------------
if make_plot:
    try:

        plot_overview_with_quality(
            evt=evt,
            dx=dx,
            dt=dt,
            xPe_m=xPe_m,
            tPe_s=tPe_s,
            picks_seconds=res.picks_seconds,
            ld_P_seconds=res.ld_P_seconds,
            quality=res.quality,
            quality_thresh=params.quality_thresh,

        )
        plt.show()
    except Exception as e:
        print(f"Plotting failed: {e}")


# ### Compute the bias from the interpolated picks between your manual picks (the rougher the manual, the higher the expected bias)

# In[16]:


bias_s = np.median(res.picks_seconds - res.ld_P_seconds)
print('median bias [ms]:', 1e3*bias_s)


# ### Quality control: 
# 
# Here your template (made from your closest manual pick) is overlain and the final pick is shown for QC. Adapt the `n_show` and `window_ms` to your data settings.

# In[21]:


qc_waveform_picks(
    evt=evt, dt=dt, dx=dx,
    picks_samples=res.picks_samples,
    n_show=20, window_ms=8.0,
    ld_seconds=res.ld_P_seconds,
    quality=res.quality,
    template_bank=res.template_bank,            # <- per-channel overlay
    template_positions=res.template_positions,  # <-
    template_scale=1.0,
)

