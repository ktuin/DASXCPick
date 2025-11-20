"""
DASXCPick.py
=================
Semi-automatic P-wave picker for DAS with interactive master picking (2025 version).

Key entry points
----------------
- :func:`pick_masters_interactive`
- :func:`pick_all_traces`
- :func:`plot_section_with_picks`
- :func:`save_picks_csv`

Notes
-----
- Input section shape is (n_traces, n_samples).
- Times are seconds; samples are index units.
- Distances are meters along fiber.


Author: K. Tuinstra + a little ChatGPT
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Optional, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import resample_poly  


# ----------------------------- Data classes ---------------------------------

@dataclass
class PickerParams:
    """
    Parameters controlling cross-correlation picking.

    Attributes
    ----------
    temp_length : int
        Template length in samples (odd recommended).
    window_ms : float
        Half-width of the search window around predicted time (milliseconds).
    quality_thresh : float
        Minimum normalized correlation at the detected lag to accept a pick.
    use_nearest_template : bool
        If True, use the nearest master’s template per channel (template bank).
        If False, use the mean template from all masters.
    smoothing_window : int
        Window length (traces). Set 0/1 to disable smoothing.
    smoothing_poly : int
        Polynomial order.
    phase_only : bool
        If True, use phase-only correlation.
    band : tuple[float, float] | None
        Optional bandpass (low_Hz, high_Hz). None disables filtering.
    detrend : bool
        If True, linear detrend per trace before picking.
    zero_phase : bool
        If True, use zero-phase filtering (`sosfiltfilt`) for `band`.
    inpaint : bool
        If True, linearly inpaint failed/low-quality picks along depth.
    upsample : int
        Integer upsampling factor for segment/template before x-corr.
        Use 1 to disable; try 4–8 for sub-sample stability.
    """
    temp_length: int = 30
    window_ms: float = 5.0
    quality_thresh: float = 0.3
    use_nearest_template: bool = False
    smoothing_window: int = 21
    smoothing_poly: int = 2
    phase_only: bool = False
    band: Optional[Tuple[float, float]] = None
    detrend: bool = True
    zero_phase: bool = True
    inpaint: bool = True
    upsample: int = 4  # Change this to >1 for upsampling x-corr


@dataclass
class PickerResult:
    """
    Results returned by :func:`pick_all_traces`.

    Attributes
    ----------
    picks_samples : np.ndarray
        Picked arrival indices (samples), length = n_traces.
    picks_seconds : np.ndarray
        Picked arrival times (seconds), length = n_traces.
    quality : np.ndarray
        Normalized correlation value at the detected lag, length = n_traces.
    ld_P_seconds : np.ndarray
        Predicted moveout (seconds) interpolated from master picks.
    template : np.ndarray
        Mean template constructed from masters (z-scored).
    template_bank : np.ndarray | None
        Templates per master (z-scored), shape (n_masters, L) if available.
    template_positions : np.ndarray | None
        Channel indices of masters corresponding to `template_bank`.
    meta : dict
        Metadata (e.g., dt, dx, params, n_traces, n_samples).
    """
    picks_samples: np.ndarray
    picks_seconds: np.ndarray
    quality: np.ndarray
    ld_P_seconds: np.ndarray
    template: np.ndarray
    template_bank: Optional[np.ndarray]
    template_positions: Optional[np.ndarray]
    meta: Dict


# ----------------------------- Utilities ------------------------------------

def _zscore(x: np.ndarray, axis: Optional[int] = None, eps: float = 1e-10) -> np.ndarray:
    """Z-score normalization with numerical safeguard."""
    m = np.mean(x, axis=axis, keepdims=True)
    s = np.std(x, axis=axis, keepdims=True)
    return (x - m) / (s + eps)


def _parabolic_refine(y: np.ndarray, k: int) -> float:
    """
    Quadratic peak refinement.

    Parameters
    ----------
    y : np.ndarray
        1D array to refine on (e.g., correlation).
    k : int
        Index of the discrete peak.

    Returns
    -------
    float
        Sub-sample offset in samples (added to `k`).
    """
    if k <= 0 or k >= len(y) - 1:
        return 0.0
    y1, y2, y3 = y[k - 1], y[k], y[k + 1]
    denom = y1 - 2.0 * y2 + y3
    if np.abs(denom) < 1e-12:
        return 0.0
    return 0.5 * (y1 - y3) / denom


def _phase_only_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Phase-only (PHAT) cross-correlation.

    Parameters
    ----------
    a, b : np.ndarray
        Real 1D signals.

    Returns
    -------
    np.ndarray
        Correlation sequence (length len(a)+len(b)-1).
    """
    n = len(a) + len(b) - 1
    nfft = 1 << (n - 1).bit_length()
    A = np.fft.rfft(a, nfft)
    B = np.fft.rfft(b, nfft)
    X = A * np.conj(B)
    X /= (np.abs(X) + 1e-12)
    xc = np.fft.irfft(X, nfft)[:n]
    return xc


def _normxcorr(signal_seg: np.ndarray, template: np.ndarray, phase_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalized cross-correlation with optional phase-only (PHAT) weighting.

    Parameters
    ----------
    signal_seg : np.ndarray
        Segment of the trace to search.
    template : np.ndarray
        Template wavelet (same units/sampling).
    phase_only : bool
        If True, use phase-only correlation.

    Returns
    -------
    xc : np.ndarray
        Normalized correlation sequence.
    lags : np.ndarray
        Corresponding lags (samples).
    """
    s = _zscore(signal_seg.astype(float))
    t = _zscore(template.astype(float))
    if phase_only:
        xc = _phase_only_corr(s, t)
        xc /= (np.sqrt(np.sum(s ** 2)) * np.sqrt(np.sum(t ** 2)) + 1e-12)
    else:
        xc = signal.correlate(s, t, mode="full")
        denom = np.sqrt(np.sum(s ** 2)) * np.sqrt(np.sum(t ** 2)) + 1e-12
        xc = xc / denom
    lags = signal.correlation_lags(len(s), len(t), mode="full")
    return xc, lags


# ------------------------- Template construction -----------------------------

def _extract_template(evt: np.ndarray, ch_idx: int, samp_idx: int, L: int) -> np.ndarray:
    """
    Extract a z-scored template window centered at sample index.

    Raises
    ------
    ValueError
        If the requested window exceeds the trace bounds.
    """
    half = L // 2
    s0 = int(samp_idx) - half
    s1 = s0 + L
    if s0 < 0 or s1 > evt.shape[1]:
        raise ValueError("Template window hits edge; increase padding or shorten temp_length.")
    tpl = evt[int(ch_idx), s0:s1].astype(float)
    return _zscore(tpl)


def build_moveout_and_templates(
    evt: np.ndarray,
    xPe_m: Iterable[float],
    tPe_s: Iterable[float],
    dx: float,
    dt: float,
    temp_length: int,
    use_nearest_template: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build predicted moveout (interpolated from masters) and templates.

    Parameters
    ----------
    evt : np.ndarray
        Section (n_traces, n_samples).
    xPe_m : Iterable[float]
        Master pick positions along depth [m].
    tPe_s : Iterable[float]
        Master pick times [s].
    dx : float
        Spatial sampling [m/trace].
    dt : float
        Temporal sampling [s/sample].
    temp_length : int
        Template length in samples.
    use_nearest_template : bool
        If True, return a bank of templates (per master) and positions.

    Returns
    -------
    ld_P : np.ndarray
        Predicted moveout (seconds) for each channel.
    master_tpl : np.ndarray
        Mean template (z-scored).
    template_bank : np.ndarray | None
        Templates per master if `use_nearest_template` is True.
    template_positions : np.ndarray | None
        Channel indices of masters corresponding to `template_bank`.
    """
        
    xPe = np.asarray(list(xPe_m), dtype=float)
    tPe = np.asarray(list(tPe_s), dtype=float)
    if xPe.ndim != 1 or tPe.ndim != 1 or len(xPe) != len(tPe):
        raise ValueError("xPe and tPe must be 1D and equal length.")

    ntr = evt.shape[0]
    channels = np.arange(ntr)

    ch_master = (xPe / dx).astype(int)
    ch_master = np.clip(ch_master, 0, ntr - 1)

    ld_P = np.interp(channels, ch_master, tPe)

    tpl_list = []
    for xm, tm in zip(ch_master, tPe):
        tpl_list.append(_extract_template(evt, int(xm), int(round(tm / dt)), temp_length))
    tpl_bank = np.stack(tpl_list, axis=0)
    master_tpl = _zscore(np.mean(tpl_bank, axis=0))

    if use_nearest_template:
        return ld_P, master_tpl, tpl_bank, ch_master.astype(int)
    else:
        return ld_P, master_tpl, None, None


# ------------------------------- Picking -------------------------------------

def _pick_trace(
    trace: np.ndarray,
    t_pred_samp: float,
    template: np.ndarray,
    W: int,
    phase_only: bool,
    upsample: int = 1,            # NEW
) -> Tuple[float, float]:
    """
    Pick a single trace by normalized x-correlation within a window.

    Parameters
    ----------
    trace : np.ndarray
        1D trace array.
    t_pred_samp : float
        Predicted pick time (samples).
    template : np.ndarray
        Template wavelet.
    W : int
        Half-window in samples to search around prediction.
    phase_only : bool
        Use phase-only correlation if True.
    upsample : int
        Integer upsampling factor for both segment and template.

    Returns
    -------
    pick : float
        Picked sample index (can be fractional).
    quality : float
        Correlation value at detected lag.
    """

    n = trace.shape[0]
    s0 = int(max(0, np.floor(t_pred_samp - W)))
    s1 = int(min(n, np.ceil(t_pred_samp + W)))
    seg = trace[s0:s1]
    if len(seg) < len(template):
        return np.nan, np.nan

    if upsample > 1:
        # Bandlimited upsampling of both segment and template
        seg_u = resample_poly(seg.astype(float), up=upsample, down=1)
        tpl_u = resample_poly(template.astype(float), up=upsample, down=1)
        xc, lags = _normxcorr(seg_u, tpl_u, phase_only=phase_only)
        k = int(np.argmax(xc))
        delta = _parabolic_refine(xc, k)
        lag_u = lags[k] + delta             # lag in upsampled samples
        # anchor to template center and convert back to original-sample units
        pick = s0 + (lag_u + (len(tpl_u) // 2)) / upsample
        q = float(xc[k])
        return float(pick), q
    else:
        xc, lags = _normxcorr(seg, template, phase_only=phase_only)
        k = int(np.argmax(xc))
        delta = _parabolic_refine(xc, k)
        lag = lags[k] + delta
        pick = s0 + lag + (len(template) // 2)
        q = float(xc[k])
        return float(pick), q


def pick_all_traces(
    evt: np.ndarray,
    dx: float,
    dt: float,
    xPe_m: Iterable[float],
    tPe_s: Iterable[float],
    params: PickerParams = PickerParams(),
) -> PickerResult:
    """
    Pick all traces using cross-correlation guided by master picks.

    Parameters
    ----------
    evt : np.ndarray
        Section (n_traces, n_samples).
    dx : float
        Spatial sampling [m/trace].
    dt : float
        Temporal sampling [s/sample].
    xPe_m : Iterable[float]
        Master positions [m].
    tPe_s : Iterable[float]
        Master times [s].
    params : PickerParams
        Picking and preprocessing parameters.

    Returns
    -------
    PickerResult
        Object containing picks, quality, moveout, and templates.
    """

    evt_work = np.array(evt, dtype=float, copy=True)
    ntr, ns = evt_work.shape

    if params.detrend:
        evt_work = signal.detrend(evt_work, axis=1, type="linear")
    if params.band is not None:
        low, high = params.band
        if low <= 0 or high >= 0.5 / dt:
            raise ValueError("Band edges must be within (0, Nyquist).")
        sos = signal.butter(4, [low, high], btype="band", fs=1.0 / dt, output="sos")
        if params.zero_phase:
            evt_work = signal.sosfiltfilt(sos, evt_work, axis=1)
        else:
            evt_work = signal.sosfilt(sos, evt_work, axis=1)

    ld_P, master_tpl, tpl_bank, tpl_pos = build_moveout_and_templates(
        evt_work, xPe_m, tPe_s, dx, dt, params.temp_length, params.use_nearest_template
    )

    W = int(round(params.window_ms * 1e-3 / dt))
    if W < 3:
        W = 3

    picks = np.full(ntr, np.nan, dtype=float)
    qual = np.full(ntr, np.nan, dtype=float)

    for i in range(ntr):
        t_pred = ld_P[i] / dt
        if params.use_nearest_template and tpl_bank is not None and tpl_pos is not None:
            j = int(np.argmin(np.abs(tpl_pos - i)))
            tpl = tpl_bank[j]
        else:
            tpl = master_tpl
        p, q = _pick_trace(
            evt_work[i, :],
            t_pred,
            tpl,
            W,
            params.phase_only,
            upsample=params.upsample,   # NEW
        )

        picks[i], qual[i] = p, q

    ok = (qual >= params.quality_thresh) & np.isfinite(picks)
    if params.inpaint:
        idx = np.arange(ntr)
        if np.count_nonzero(ok) >= 2:
            picks[~ok] = np.interp(idx[~ok], idx[ok], picks[ok])

    if params.smoothing_window and params.smoothing_window > 1:
        win = int(params.smoothing_window)
        if win % 2 == 0:
            win += 1
        win = max(win, 5)
        poly = int(params.smoothing_poly)
        picks = signal.savgol_filter(picks, win, poly, mode="interp")

    return PickerResult(
        picks_samples=picks,
        picks_seconds=picks * dt,
        quality=qual,
        ld_P_seconds=ld_P,
        template=master_tpl,
        template_bank=tpl_bank,
        template_positions=tpl_pos,
        meta={"dt": dt, "dx": dx, "params": params, "n_traces": ntr, "n_samples": ns},
    )


# ------------------------------- I/O helpers ---------------------------------

def save_picks_csv(path: str, depth_axis_m: np.ndarray, picks_seconds: np.ndarray, ld_seconds: Optional[np.ndarray] = None) -> None:
    """
    Save picks (and optional moveout) to CSV.

    Parameters
    ----------
    path : str
        Output CSV path.
    depth_axis_m : np.ndarray
        Depth vector [m], length = n_traces.
    picks_seconds : np.ndarray
        Picks [s], length = n_traces.
    ld_seconds : np.ndarray, optional
        Predicted moveout [s], length = n_traces.
    """
        
    depth_axis_m = np.asarray(depth_axis_m)
    picks_seconds = np.asarray(picks_seconds)
    if ld_seconds is not None:
        ld_seconds = np.asarray(ld_seconds)
        arr = np.vstack([depth_axis_m, picks_seconds, ld_seconds]).T
        header = "depth_m,pick_s,ld_pred_s"
    else:
        arr = np.vstack([depth_axis_m, picks_seconds]).T
        header = "depth_m,pick_s"
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


# ------------------------------- Plot helpers --------------------------------

def plot_section_with_picks(evt: np.ndarray, dx: float, dt: float, picks_samples: np.ndarray, ld_P_seconds: Optional[np.ndarray] = None, ax=None):
    """
    Plot section with picks and optional interpolated moveout.

    Parameters
    ----------
    evt : np.ndarray
        Section (n_traces, n_samples).
    dx : float
        Spatial sampling [m/trace].
    dt : float
        Temporal sampling [s/sample].
    picks_samples : np.ndarray
        Picks (samples).
    ld_P_seconds : np.ndarray, optional
        Predicted moveout (seconds).
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for plotting.
    """
    ntr, ns = evt.shape
    depth = np.arange(ntr) * dx
    time = np.arange(ns) * dt
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(evt.T, aspect="auto", cmap="gray", extent=[depth[0], depth[-1], time[-1], time[0]])
    ax.set_xlabel("Distance along fiber [m]")
    ax.set_ylabel("Time [s]")
    ax.plot(depth, picks_samples * dt, lw=2, label="Picks")
    if ld_P_seconds is not None:
        ax.plot(depth, ld_P_seconds, lw=1.5, label="Predicted moveout")
    # ax.invert_yaxis()
    ax.legend()
    import matplotlib.pyplot as plt
    plt.colorbar(im, ax=ax, label="Amplitude (a.u.)")
    return ax


# ----------------------- Interactive master picking UI -----------------------

def pick_masters_interactive(evt: np.ndarray, dx: float, dt: float, n_required: Optional[int] = None, title: str = "Pick P-wave masters (LMB add, RMB remove, Enter finish)", cmap: str = "gray", vmin: Optional[float] = None, vmax: Optional[float] = None, overlay_wiggles: bool = True, wiggle_decim: int = 5, wiggle_scale: float = 1.5):
    """
    Interactive picker for master picks.

    Parameters
    ----------
    evt : np.ndarray
        Section (n_traces, n_samples).
    dx : float
        Spatial sampling [m/trace].
    dt : float
        Temporal sampling [s/sample].
    n_required : int, optional
        If provided, close after this many picks; otherwise press Enter.
    title : str
        Window title.
    cmap : str
        Colormap for imshow.
    vmin, vmax : float, optional
        Contrast limits for imshow.
    overlay_wiggles : bool
        If True, overlay decimated wiggle traces.
    wiggle_decim : int
        Plot every Nth trace as wiggle.
    wiggle_scale : float
        Horizontal scale of wiggle overlay (in trace spacings).

    Returns
    -------
    xPe_m : list[float]
        Picked positions along depth [m], left-to-right sorted.
    tPe_s : list[float]
        Picked times [s], paired with `xPe_m`.
    """

    ntr, ns = evt.shape
    depth = np.arange(ntr) * dx
    time = np.arange(ns) * dt
    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(evt.T[::-1, :], aspect="auto", cmap=cmap, extent=[depth[0], depth[-1], time[0], time[-1]], vmin=vmin, vmax=vmax, interpolation="none")
    ax.invert_yaxis()
    ax.set_xlabel("Distance along fiber [m]")
    ax.set_ylabel("Time [s]")
    plt.colorbar(im, ax=ax, label="Amplitude (a.u.)")
    if overlay_wiggles:
        for i in range(0, ntr, max(1, wiggle_decim)):
            trace = evt[i, :]
            tr = trace / (np.max(np.abs(trace)) + 1e-12)
            ax.plot(depth[i] + tr * wiggle_scale * dx, time, lw=0.6, alpha=0.7)
    picks = []
    scat = ax.scatter([], [], s=60, c="lime", marker="o", edgecolors="k", linewidths=0.5, zorder=5)
    txt = ax.text(0.01, 0.02, "0 picks", transform=ax.transAxes, color="w", bbox=dict(facecolor="k", alpha=0.3))
    ax.set_title(title)
    def redraw():
        if picks:
            xs, ys = zip(*picks)
        else:
            xs, ys = [], []
        scat.set_offsets(np.c_[xs, ys])
        txt.set_text(f"{len(picks)} pick(s)" + (f" / {n_required}" if n_required else ""))
        fig.canvas.draw_idle()
    def on_click(event):
        if event.inaxes is None:
            return
        if event.button == 1:
            picks.append((event.xdata, event.ydata))
            if n_required and len(picks) >= n_required:
                plt.close(fig)
                return
            redraw()
        elif event.button == 3 and picks:
            d2 = [(px - event.xdata) ** 2 + (py - event.ydata) ** 2 for px, py in picks]
            j = int(np.argmin(d2))
            picks.pop(j)
            redraw()
    def on_key(event):
        if event.key in ("enter", "return"):
            plt.close(fig)
        elif event.key == "escape" and picks:
            picks.pop()
            redraw()
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if picks:
        picks.sort(key=lambda p: p[0])
        xPe_m = [p[0] for p in picks]
        tPe_s = [p[1] for p in picks]
    else:
        xPe_m, tPe_s = [], []
    return xPe_m, tPe_s

def plot_overview_with_quality(
    evt: np.ndarray,
    dx: float,
    dt: float,
    xPe_m: Iterable[float],
    tPe_s: Iterable[float],
    picks_seconds: np.ndarray,
    ld_P_seconds: np.ndarray,
    quality: np.ndarray,
    quality_thresh: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """
    Overview plot with masters, interpolated moveout, final picks, and QC shading.

    Low-quality channels (quality < `quality_thresh`) are shaded, and the
    corresponding pick is marked.

    Parameters
    ----------
    evt, dx, dt : see :func:`plot_section_with_picks`.
    xPe_m, tPe_s : Iterable[float]
        Manual master picks (depth [m], time [s]).
    picks_seconds : np.ndarray
        Final picks [s].
    ld_P_seconds : np.ndarray
        Interpolated moveout [s].
    quality : np.ndarray
        Correlation-at-peak per trace.
    quality_thresh : float
        QC threshold below which traces are highlighted.
    vmin, vmax : float, optional
        Contrast limits for imshow.

    Returns
    -------
    matplotlib.axes.Axes
        The axes used for plotting.
    """
    import matplotlib.pyplot as plt
    ntr, ns = evt.shape
    depth = np.arange(ntr) * dx
    time = np.arange(ns) * dt

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        evt.T, aspect="auto", cmap="gray",
        extent=[depth[0], depth[-1], time[-1], time[0]],
        vmin=vmin, vmax=vmax
    )
    ax.set_xlabel("Distance along fiber [m]")
    ax.set_ylabel("Time [s]")

    # Final picks (cyan) and interpolated moveout (orange)
    ax.plot(depth, picks_seconds, lw=2, color="cyan", label="Final picks")
    ax.plot(depth, ld_P_seconds, lw=1.2, color="tab:orange", label="Interpolated moveout")

    # Manual master clicks (green dots)
    ax.scatter(xPe_m, tPe_s, s=40, c="lime", edgecolors="k", linewidths=0.5, zorder=5, label="Master picks")

    # Low-quality channels: shade vertical bands + mark pick with 'x'
    low = np.where(np.isfinite(quality) & (quality < quality_thresh))[0]
    for i in low:
        x0 = depth[i] - dx * 0.5
        x1 = depth[i] + dx * 0.5
        ax.axvspan(x0, x1, color="red", alpha=0.12, lw=0)
        ax.scatter(depth[i], picks_seconds[i], s=20, c="red", marker="x", zorder=6)

    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="Amplitude (a.u.)")
    ax.set_title(f"Overview with QC (threshold={quality_thresh:.2f}; low-quality channels shaded)")
    fig.tight_layout()
    return ax



