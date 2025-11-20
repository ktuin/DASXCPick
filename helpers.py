
"""
helpers.py
=================
Utility functions for interactive picking of P-wave arrivals in DAS data.

"""

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton



def qc_waveform_picks(
    evt,                  # array (n_traces, n_samples)
    dt,                   # seconds per sample
    dx,                   # meters per trace (used in title only)
    picks_samples,        # array (n_traces,) picks in samples
    n_show=20,            # how many panels
    window_ms=8.0,        # +/- window size in ms around pick
    channels=None,        # list/array of channels to show; default = evenly spaced
    ld_seconds=None,      # optional predicted moveout (seconds), same length as traces
    quality=None,         # optional quality vector
    template=None,                 # keep for global mode
    template_bank=None,            # (n_masters, L)
    template_positions=None,       # (n_masters,) channel indices of masters
    template_scale=1.0,   # scale for template overlay (visual only)
    suptitle="Waveform QC around picks",
    popout=True           # True: use TkAgg so it pops out from notebooks
):
    if popout:
        try:
            matplotlib.use("TkAgg", force=True)
        except Exception:
            pass

    ntr, ns = evt.shape
    if channels is None:
        # pick evenly spaced valid channels with finite picks
        valid = np.where(np.isfinite(picks_samples))[0]
        if len(valid) == 0:
            raise ValueError("No valid picks to show.")
        # evenly sample up to n_show from valid
        idx = np.linspace(0, len(valid)-1, min(n_show, len(valid))).round().astype(int)
        channels = valid[idx]
    else:
        channels = np.asarray(channels)[:n_show]

    W = int(round(window_ms*1e-3/dt))
    ncols = 5
    nrows = int(np.ceil(len(channels)/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3.4, nrows*2.4), squeeze=False)
    fig.suptitle(f"{suptitle}  —  dx={dx} m, dt={dt*1e6:.1f} µs, window=±{window_ms} ms", y=0.98)

    for k, ch in enumerate(channels):
        r = k // ncols
        c = k % ncols
        ax = axes[r, c]

        pk = int(round(picks_samples[ch]))
        s0 = max(0, pk - W)
        s1 = min(ns, pk + W)
        sig = evt[ch, s0:s1]

        t = (np.arange(s0, s1) - pk) * dt  # time relative to pick [s]
        ax.plot(t*1e3, sig, lw=1.0, color="0.25", label="trace")

        # mark pick at t=0
        ax.scatter([0], [0], s=40, c="red", marker="*", zorder=5, label="pick")
        

        # overlay predicted moveout time if provided (vertical line at t_pred - t_pick)
        if ld_seconds is not None and np.isfinite(ld_seconds[ch]):
            t_pred = ld_seconds[ch]
            t_pick = picks_samples[ch] * dt
            dt_ms = (t_pred - t_pick) * 1e3
            ax.axvline(dt_ms, color="tab:orange", lw=1.2, label="pred moveout")

        # choose overlay template
        tpl_to_use = None
        if template_bank is not None and template_positions is not None and len(template_bank) > 0:
            j = int(np.argmin(np.abs(template_positions - ch)))
            tpl_to_use = template_bank[j]
        elif template is not None:
            tpl_to_use = template

        if tpl_to_use is not None and len(tpl_to_use) > 1:
            L = len(tpl_to_use)
            tc = (np.arange(L) - L//2) * dt
            tpl = tpl_to_use - np.mean(tpl_to_use)
            tpl /= (np.std(tpl) + 1e-12)
            scal = template_scale * (np.std(sig) + 1e-12)
            ax.plot(tc*1e3, tpl*scal, lw=1.0, alpha=0.9, label="template")

        qtxt = f"ch={ch}"
        if quality is not None and np.isfinite(quality[ch]):
            qtxt += f", q={quality[ch]:.2f}"
        ax.set_title(qtxt, fontsize=9)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-window_ms, +window_ms)
        # tidy axes
        if r < nrows-1:
            ax.set_xticklabels([])
        if c > 0:
            ax.set_yticklabels([])

    # hide any unused axes
    total_ax = nrows*ncols
    for k in range(len(channels), total_ax):
        r = k // ncols
        c = k % ncols
        axes[r, c].axis("off")

    # single legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    plt.show()


