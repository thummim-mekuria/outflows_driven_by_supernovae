"""new_fig3: make_modified_fig2_runs.py with a second row of insets - the top row of the paper's Fig 3 (n-T phase
diagrams colored by f_ion for HD / RHD / SNe), drawn from the same ~10 Myr snapshots as the Sigma insets and stacked
directly above them. The Sigma insets keep the size they have in modified_fig2_runs; the main axes (and figure) are
made taller so that the two-row stack still fits under the curves.

Reuses the helpers in make_paper_figs.ipynb (RUNS, snap_paths, is_valid, nearest_valid, load, FIELD_*), selecting the
cells by content rather than by index.
"""
import json, os, sys
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import yt
from scipy.ndimage import gaussian_filter

HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "make_paper_figs.ipynb")))
for key in ("RUNS = {", "def snap_paths", "FIELD_IONFRAC ="):   # imports/RUNS/paths, snapshot helpers, FIELD_* names
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and key in "".join(c["source"])]
    exec(src[0], globals())

# which figure: "fig3" (HD / RHD / SNe at ~10 Myr, default), "fig1" (the fiducial SNe+RHD run at 5 / 10 / 15 Myr) or
# "figZ" (SNe+RHD at 0.01 / 0.1 / 1 Zsun at ~10 Myr; the 0.1 Zsun column is an empty placeholder - no run on disk)
FIGURE = sys.argv[1] if len(sys.argv) > 1 else "fig3"
assert FIGURE in ("fig3", "fig1", "figZ"), FIGURE
# optional second argument "contours": overlay gas-mass contours on the phase diagrams (writes new_<fig>_contours.png)
PH_CONTOURS = len(sys.argv) > 2 and sys.argv[2] == "contours"
OUTFILE = os.path.join(HERE, "..", "..", "outputs", "figures", f"new_{FIGURE}{'_contours' if PH_CONTOURS else ''}.png")
T_FF = 2.5           # Myr; top axis is t / t_ff, 0-8 over 0-20 Myr
FIG_W, FIG_H = 38.0, 19.0            # inches (2:1); the inset size is derived below so the stack fits under the curves
MARGIN = dict(left=2.0, right=2.2, bottom=1.5, top=1.6)   # inches around the main axes
INSET_BOTTOM_IN = 1.2                # actual gap between the bottom (phase-diagram) row and the x axis [in]: its tick labels
ROW_GAP_ACT_IN = 0.0                 # actual gap between the phase-diagram row and the Sigma row above it [in]
TOP_CLEAR_IN = 0.25                  # clearance between the top of the stack and the lowest curve
PH_ASPECT = 0.65                     # phase-diagram height / width (width = Sigma-map side)
SCALE_PC = 500.0                     # length of the external scale bar [pc]
SNAP_10 = 100                        # snapshot index at ~10 Myr
CURVE_LW = 5                         # M_* curves
GUIDE_LW = 3                         # vertical guide lines from the curves to the columns
if FIGURE == "fig3":
    # curves: draw order (later = on top), label, color, linestyle
    CURVES = [("SNe", "SNe", "blue", "-"), ("RHD", "RHD", "red", "-"), ("HD", "HD", "green", "-")]
    # columns, left to right: (run, snapshot index, label, label/guide color, left-edge time [Myr], marked time [Myr])
    INSETS = [("HD", SNAP_10, "HD", "green", 7.5, 10.0), ("RHD", SNAP_10, "RHD", "red", 11.75, 10.0),
              ("SNe", SNAP_10, "SNe", "blue", 16.0, 10.0)]
    EPS_RUN = "SNe"                  # run whose terminal SFE is drawn as the grey dashed line
    MARK_TIMES = True                # cyan triangles on the time axis at the marked times
    COLUMN_LABELS = True             # column labels inside the Sigma maps
    GUIDE_TO_AXIS = False            # guide lines run from the curve to the bottom of the column (True: to the time axis)
    GUIDE_TILT = False
    INSET_IN_FIXED = None            # inset side [in]; None = largest stack that fits under the curves at the first column
    CB_RIGHT = False                 # colorbars at the top right of the axes (False: top left)
    T_DET = None                     # Myr; if set: grey dashed t_det line + labels for t_det and eps_term
    V_REF_FIXED = {"RHD": 500, "HD": 500}   # km/s; columns of these runs use this key speed instead of their p95
elif FIGURE == "figZ":
    CURVES = [("SNeRHD,f", "0.01 Zsun", "k", "-"), ("SNeRHD,Zsun", "Zsun", "darkorange", "-")]
    # run None = placeholder column (frames and labels only): the 0.1 Zsun run has no directory under data/sims
    INSETS = [("SNeRHD,f", SNAP_10, r"$0.01\,Z_\odot$", "k", 7.5, 10.0),
              (None, SNAP_10, r"$0.1\,Z_\odot$", "gray", 11.75, 10.0),
              ("SNeRHD,Zsun", SNAP_10, r"$Z_\odot$", "darkorange", 16.0, 10.0)]
    EPS_RUN = "SNeRHD,f"
    MARK_TIMES = True
    COLUMN_LABELS = True
    GUIDE_TO_AXIS = False
    GUIDE_TILT = False
    INSET_IN_FIXED = None
    CB_RIGHT = False
    T_DET = None
    V_REF_FIXED = {}
else:
    CURVES = [("SNeRHD,f", "SNe+RHD (fiducial)", "k", "-")]
    # each column's left edge sits at its own snapshot time; the guide line down to the time axis marks it
    # columns at the same positions as new_fig3 (under the plateau); a slanted line from each column's bottom-left
    # corner points to its snapshot time on the time axis
    INSETS = [("SNeRHD,f", 50, "", "gray", 7.5, 5.0), ("SNeRHD,f", 100, "", "gray", 11.75, 10.0),
              ("SNeRHD,f", 150, "", "gray", 16.0, 15.0)]
    EPS_RUN = "SNeRHD,f"
    MARK_TIMES = False
    COLUMN_LABELS = False
    GUIDE_TO_AXIS = False
    GUIDE_TILT = True                # slanted line from the column's bottom-left corner to its time on the time axis
    INSET_IN_FIXED = None
    CB_RIGHT = True                  # top right is empty above the single plateau; top left carries the labels
    T_DET = 6.3                      # first SN onset in the fiducial run (first nonzero ssne, snapshot 0063)
    V_REF_FIXED = {}

FS = 46                              # one size for every piece of text in the figure
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": FS})

CMAP = "magma_r"                     # inverted magma for the Sigma maps
SIGMA_LIM = (1e1, 1e5)               # Msun/pc^2 color limits (must match set_zlim in inset_snapshot)
STAR_COLOR = "cyan"                  # sink-particle marker color (black outline)
STAR_SIZE = 500                      # sink-particle marker size
WIDTH_KPC = 2.8                      # side of the Sigma maps [kpc]
# velocity arrows: v_x, v_y projected along z and VEL_WEIGHT-weighted ("mass": density-weighted, "volume": path-length
# weighted), sampled on an ARROW_N x ARROW_N grid. Each panel has its own scale: its key speed is that panel's
# 95th-percentile arrow speed, rounded to 1/2/5 x 10^k, drawn ARROW_REF_IN long, with a labelled key arrow per panel.
VEL_WEIGHT = "mass"
ARROW_N = 12                         # arrows per side
ARROW_KEY_FS = 28                    # velocity key label
ARROW_REF_IN = 0.50                  # length of the key arrow [in]
ARROW_WIDTH = 0.004                  # shaft width (fraction of the map width)
ARROW_HEAD = (4.5, 5.0, 4.5)         # quiver (headwidth, headlength, headaxislength), in shaft widths

# phase diagrams (as make_fig_3 in the notebook / the paper's Fig 3)
PH_MASS_LEVELS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)   # contours enclosing these fractions of the panel gas mass
PH_CONTOUR_SMOOTH = 1.0              # gaussian smoothing of the mass histogram for the contours only [bins]
PH_NBINS = 100                       # log-spaced bins per axis for the n-T histograms
PH_CMAP = "coolwarm_r"
PH_FION_LIM = (1e-2, 1.0)
PH_XLIM = (2e-4, 2e4)                # n [cm^-3]; covers the data on disk (max ~4e3, see note in the docstring)
PH_YLIM = (2e1, 1e9)                 # T [K]
PH_LABEL_FS = 34                     # phase-diagram tick and axis labels, and the run labels on the Sigma maps

MSUN, MYR = 1.989e33, 3.15576e13


def read_particles(fn):
    """(t [Myr], tag, mass [Msun]) from one FLASH particle file; raises if unreadable / never fully written."""
    st = os.stat(fn)
    if st.st_size == 0 or st.st_blocks * 512 < 0.9 * st.st_size:
        raise IOError("sparse or empty file")
    with h5py.File(fn, "r") as h:
        t = [v for k, v in h["real scalars"][:] if k.decode().strip() == "time"][0] / MYR
        if "tracer particles" in h:      # absent before the first sink forms
            names = [x.decode().strip() for x in h["particle names"][:].ravel()]
            a = h["tracer particles"][:]
            if not np.isfinite(a).all():
                raise IOError("non-finite particle data")
            return t, a[:, names.index("tag")].astype(int), a[:, names.index("mass")] / MSUN
    return t, np.array([], int), np.array([])


def mformed_series(run, n_snap=N_SNAP):
    """Cumulative formed stellar mass vs time for all n_snap snapshots of `run`, read from sims/ particle files.
    M_formed(t) = sum over every sink (by tag) seen in snapshots <= t of its largest mass (stars that left the box
    still count). Snapshots whose particle file is missing/corrupt are filled by linear interpolation in index.
    Returns t [Myr], M_formed [1e10 Msun], and the list of filled snapshot indices."""
    seen, t, m, good = {}, np.full(n_snap, np.nan), np.full(n_snap, np.nan), []
    for i in range(n_snap):
        _, fn = snap_paths(run, i)
        try:
            if fn is None:
                raise FileNotFoundError
            ti, tag, mm = read_particles(fn)
        except Exception:
            continue
        for g, x in zip(tag, mm):
            seen[g] = max(seen.get(g, 0.0), x)
        t[i], m[i] = ti, sum(seen.values())
        good.append(i)
    idx = np.arange(n_snap)
    filled = [i for i in idx if i not in set(good)]
    if filled:
        t[filled] = np.interp(filled, good, t[good])
        m[filled] = np.interp(filled, good, m[good])
    return t, m / 1e10, filled


def velocity_grid(ds, width_kpc=WIDTH_KPC):
    """Projected (VEL_WEIGHT-weighted along z) v_x, v_y [km/s] on an ARROW_N x ARROW_N grid of bin centres [kpc about
    the domain centre]."""
    weight = ("index", "ones") if VEL_WEIGHT == "volume" else ("gas", "density")
    prj = ds.proj([("gas", "velocity_x"), ("gas", "velocity_y")], "z", weight_field=weight)
    frb = prj.to_frb((width_kpc, "kpc"), (ARROW_N, ARROW_N), center=ds.domain_center)
    vx = frb["gas", "velocity_x"].to("km/s").v; vy = frb["gas", "velocity_y"].to("km/s").v   # indexed [y, x]
    c = (np.arange(ARROW_N) + 0.5) / ARROW_N * width_kpc - width_kpc / 2
    X, Y = np.meshgrid(c, c)
    return X, Y, vx, vy


def nice_round(v):
    """v rounded to the nearest 1, 2 or 5 x 10^k."""
    k = np.floor(np.log10(v))
    return min((m * 10 ** k for m in (1, 2, 5, 10)), key=lambda x: abs(np.log(x / v)))


def inset_snapshot(run, ds, fig, rect, vgrid, v_ref, key, width_kpc=WIDTH_KPC):
    """Fig 1 panel (make_snapshot in the notebook) drawn into a small axes of the main figure, with velocity arrows on
    the panel's own scale (v_ref km/s = ARROW_REF_IN; key arrow if `key`) and the sinks as STAR_COLOR stars."""
    p = yt.ProjectionPlot(ds, "z", FIELD_DENS, width=(width_kpc, "kpc"))
    p.set_unit(FIELD_DENS, "Msun/pc**2"); p.set_zlim(FIELD_DENS, *SIGMA_LIM)
    p.set_cmap(FIELD_DENS, CMAP); p.set_buff_size(1024)
    p.hide_axes(draw_frame=False)
    ax = fig.add_axes(rect)
    cax = fig.add_axes([0, 0, 0.01, 0.01]); cax.set_visible(False)   # yt wants a colorbar axes; keep it hidden
    pl = p.plots[FIELD_DENS]
    pl.figure, pl.axes, pl.cax = fig, ax, cax
    p._setup_plots()
    ax.set_position(rect)               # yt resizes the axes to fill the figure; put it back
    fig.set_size_inches(FIG_W, FIG_H)   # ... and resizes the figure canvas; restore it
    ax.set_zorder(5)
    p._ds_keepalive = ds
    # overlay axes in kpc about the domain centre for the arrows and sinks
    ov = fig.add_axes(rect, zorder=6); ov.set_axis_off()
    half = width_kpc / 2
    ov.set(xlim=(-half, half), ylim=(-half, half))
    q = ov.quiver(*vgrid, color="k", scale=v_ref / ARROW_REF_IN, scale_units="inches", width=ARROW_WIDTH,
                  headwidth=ARROW_HEAD[0], headlength=ARROW_HEAD[1], headaxislength=ARROW_HEAD[2], zorder=2)
    if key:   # key arrow, bottom right, on a white backing
        ov.add_patch(mpl.patches.Rectangle((0.615, 0.015), 0.37, 0.14, transform=ov.transAxes, facecolor="white",
                                           alpha=0.8, edgecolor="none", zorder=4))
        qk = ov.quiverkey(q, 0.80, 0.05, v_ref, f"{v_ref:g} km/s", labelpos="N", coordinates="axes",
                          fontproperties={"size": ARROW_KEY_FS}, labelsep=0.08)
        qk.set_zorder(5)
    try:
        ad = ds.all_data(); c = ds.domain_center.to("kpc").v
        x = ad["nbody", "particle_position_x"].to("kpc").v - c[0]
        y = ad["nbody", "particle_position_y"].to("kpc").v - c[1]
        msink = ad["nbody", "particle_mass"].to("Msun").v
        k = (np.abs(x) < half) & (np.abs(y) < half)
        order = np.argsort(msink[k])                           # most massive drawn on top
        ov.scatter(x[k][order], y[k][order], s=STAR_SIZE, marker="*", color=STAR_COLOR, edgecolors="k",
                   linewidths=0.8, zorder=3)
    except Exception:
        pass                                                   # no sinks yet in this snapshot
    return p


def phase_hist(ds):
    """Gas mass [Msun] and mass-weighted mean f_ion in PH_NBINS x PH_NBINS log-spaced n-T bins over every cell."""
    ad = ds.all_data()
    n = ad[FIELD_NUM].to("cm**-3").v; T = ad[FIELD_TEMP].to("K").v; f = ad[FIELD_IONFRAC].v
    m = ad["gas", "mass"].to("Msun").v
    xe = np.logspace(*np.log10(PH_XLIM), PH_NBINS + 1); ye = np.logspace(*np.log10(PH_YLIM), PH_NBINS + 1)
    M, _, _ = np.histogram2d(n, T, bins=(xe, ye), weights=m)
    MF, _, _ = np.histogram2d(n, T, bins=(xe, ye), weights=m * f)
    with np.errstate(invalid="ignore", divide="ignore"):
        fbar = np.where(M > 0, MF / M, np.nan)                     # mass-weighted mean f_ion per bin
    print(f"  phase: {n.size} cells, n in [{n.min():.3g}, {n.max():.3g}], T in [{T.min():.3g}, {T.max():.3g}]; "
          f"{100 * M.sum() / m.sum():.2f}% of the mass inside the axis limits")
    return xe, ye, M, fbar


def inset_phase(hist, fig, rect, first, xlabel):
    """n-T phase diagram: bins colored by the mass-weighted mean f_ion. Drawn into a small axes of the figure."""
    norm = mpl.colors.LogNorm(*PH_FION_LIM)
    ax = fig.add_axes(rect, zorder=5)
    if hist is not None:   # hist None = placeholder column: empty axes with the same limits and ticks
        xe, ye, M, fbar = hist
        ax.pcolormesh(xe, ye, np.ma.masked_invalid(np.clip(fbar, *PH_FION_LIM)).T, cmap=PH_CMAP, norm=norm,
                      rasterized=True)
        if PH_CONTOURS:   # mass contours: bins above each threshold hold the PH_MASS_LEVELS fractions of the mass
            Ms = gaussian_filter(M, PH_CONTOUR_SMOOTH)
            srt = np.sort(Ms.ravel())[::-1]; cum = np.cumsum(srt) / srt.sum()
            levels = sorted({srt[np.searchsorted(cum, q)] for q in PH_MASS_LEVELS})
            xc, yc = np.sqrt(xe[1:] * xe[:-1]), np.sqrt(ye[1:] * ye[:-1])
            ax.contour(xc, yc, Ms.T, levels=levels, colors="k", linewidths=1.0)
    for T0 in (T_COLD, T_HOT):
        ax.axhline(T0, color="k", ls="--", lw=1.5)
    ax.set(xscale="log", yscale="log", xlim=PH_XLIM, ylim=PH_YLIM)
    # with guide lines down to the time axis (fig1) the left edge carries the line, so keep tick labels off it
    ax.set_xticks([1e-2, 1e1, 1e4] if GUIDE_TO_AXIS else [1e-3, 1e0, 1e3]); ax.set_yticks([1e3, 1e5, 1e7])
    ax.minorticks_off()
    ax.tick_params(direction="in", length=8, width=1.5, labelsize=PH_LABEL_FS, top=True, right=True)
    if first:
        ax.set_ylabel(r"$T$ [K]", fontsize=PH_LABEL_FS, labelpad=2,
                      bbox=dict(facecolor="white", edgecolor="none", pad=1))
        for tl in ax.get_yticklabels():   # these sit over the curves left of the first column
            tl.set_bbox(dict(facecolor="white", edgecolor="none", pad=1))
    else:
        ax.set_yticklabels([])
    if xlabel:
        ax.set_xlabel(r"$n\ [{\rm cm^{-3}}]$", fontsize=PH_LABEL_FS, labelpad=1)
    return ax, mpl.cm.ScalarMappable(norm=norm, cmap=PH_CMAP)   # opaque mappable for the colorbar


# ---- curves first, so the figure height can be set from the fiducial curve at the first inset's left edge ---------
M0 = float(load("SNeRHD,f", 0).all_data()["gas", "mass"].to("Msun").sum())   # initial gas mass [Msun], snapshot 0
series = {}
for run in dict.fromkeys([r for r, *_ in CURVES] + [EPS_RUN]):
    t, m, filled = mformed_series(run)
    series[run] = (t, m)
    msg = f"{run}: read {N_SNAP - len(filled)}/{N_SNAP} snapshots, M_formed(end) = {m[-1]:.4f}e10"
    if filled:
        msg += f"; interpolated {len(filled)} missing/corrupt snapshots"
    print(msg)

# inset size: the two-row stack must fit under the lowest plotted curve at the first column's left edge
# (the curves only rise to the right of it, so that corner is the binding one)
ax_h_in = FIG_H - MARGIN["bottom"] - MARGIN["top"]
frac_corner = min(np.interp(INSETS[0][4], *series[r]) for r, *_ in CURVES) / (M0 / 1e10)   # lowest curve, axes frac
INSET_IN = (INSET_IN_FIXED or
            (frac_corner * ax_h_in - INSET_BOTTOM_IN - ROW_GAP_ACT_IN - TOP_CLEAR_IN) / (1 + PH_ASPECT))   # Sigma side
print(f"inset size {INSET_IN:.2f} in; curve at t = {INSETS[0][4]} Myr: {frac_corner:.3f} of the y range -> axes height {ax_h_in:.2f} in, "
      f"figure {FIG_W:.2f} x {FIG_H:.2f} in")

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([MARGIN["left"] / FIG_W, MARGIN["bottom"] / FIG_H,
                   1 - (MARGIN["left"] + MARGIN["right"]) / FIG_W, ax_h_in / FIG_H])
# downward triangles on the time axis marking the snapshot times of the columns
for t_mark in (sorted({c[5] for c in INSETS}) if MARK_TIMES else []):
    ax.plot([t_mark], [0], marker="v", ms=30, color="cyan", mec="k", mew=1.5, clip_on=False, zorder=12,
            transform=mpl.transforms.offset_copy(mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                                                 fig=fig, y=15, units="points"))   # tip touches the axis
# the curves are drawn on a transparent copy of the main axes above the insets (zorder 7 > insets' 5-6), so a column
# that reaches up past the curve (fig1's 5 Myr column) sits under the curve instead of hiding it
ax_curves = fig.add_axes(ax.get_position(), zorder=7); ax_curves.set_axis_off(); ax_curves.patch.set_alpha(0)
ax_curves.set(xlim=(0, 20), ylim=(0, M0 / 1e10))
for run, label, color, ls in CURVES:
    ax_curves.plot(*series[run], color=color, ls=ls, lw=CURVE_LW, label=label)
# terminal SFE of EPS_RUN, eps_term = M_formed(t_end) / M_g(t=0), as a grey dashed line (behind the curves)
eps_m = series[EPS_RUN][1][-1]
print(f"eps_term {EPS_RUN}: {eps_m * 1e10 / M0:.3f} (M_formed = {eps_m:.4f}e10 Msun at t = {series[EPS_RUN][0][-1]:.2f} Myr)")
ax.axhline(eps_m, color="gray", ls="--", lw=3, zorder=1)
if T_DET is not None:   # reference lines and labels as in the original new_fig1 (make_modified_fig2.py)
    ax.axvline(T_DET, color="gray", ls="--", lw=3, zorder=1)
    ax.text(T_DET - 0.15, 1.62, rf"$t_{{\rm det}} = {T_DET}$ Myr", color="gray", fontsize=FS, ha="right")
    ax.text(0.47, eps_m + 0.04, rf"$\epsilon_{{\rm term}} = {eps_m * 1e10 / M0:.2f}$", color="gray", fontsize=FS)

ax.set(xlim=(0, 20), ylim=(0, M0 / 1e10), xlabel="time [Myr]", ylabel=r"$M_*\ [10^{10}\,M_\odot]$")
ax.set_xticks([0, 5, 10, 15, 20]); ax.set_yticks([0, 0.5, 1.0, 1.5])
ax.tick_params(direction="in", length=7, width=1.5, top=False, right=False)

ax_top = ax.secondary_xaxis("top", functions=(lambda t: t / T_FF, lambda x: x * T_FF))
ax_top.set_xlabel(r"t / t$_{ff}$"); ax_top.set_xticks([0, 2, 4, 6, 8])
ax_top.tick_params(direction="in", length=7, width=1.5)
ax_r = ax.secondary_yaxis("right", functions=(lambda y: y * 1e10 / M0, lambda y: y * M0 / 1e10))
ax_r.set_ylabel(r"$M_*\,/\,M_{\rm g}$"); ax_r.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_r.tick_params(direction="in", length=7, width=1.5)

# ---- insets: Sigma snapshots (top row) and phase diagrams (bottom row), one column per INSETS entry ---------------
bb = ax.get_position()
w, h = INSET_IN / FIG_W, INSET_IN / FIG_H
plots = []
# per column; run None (placeholder) -> None everywhere
snaps = [nearest_valid(c[0], c[1]) if c[0] else None for c in INSETS]
dss = [load(c[0], i) if c[0] else None for c, i in zip(INSETS, snaps)]
vgrids = [velocity_grid(ds) if ds else None for ds in dss]
phists = [phase_hist(ds) if ds else None for ds in dss]
v_refs = [(V_REF_FIXED.get(c[0]) or nice_round(np.percentile(np.hypot(g[2], g[3]), 95))) if g else None
          for c, g in zip(INSETS, vgrids)]
print(f"{VEL_WEIGHT}-weighted arrows (per-panel scale, key = {ARROW_REF_IN} in): " +
      ", ".join(f"{c[0]}@{i:04d} median {np.median(np.hypot(g[2], g[3])):.0f} km/s, key {v:g} km/s"
                for c, i, g, v in zip(INSETS, snaps, vgrids, v_refs) if g))
for n, (run, _, label, color, t_left, t_mark) in enumerate(INSETS):
    i, ds = snaps[n], dss[n]
    prect = [bb.x0 + bb.width * t_left / 20.0, bb.y0 + INSET_BOTTOM_IN / FIG_H, w, PH_ASPECT * h]   # phase diagram
    frect = [prect[0], prect[1] + prect[3] + ROW_GAP_ACT_IN / FIG_H, w, h]                           # Sigma map
    if run is None:   # placeholder Sigma panel: empty framed axes
        pa = fig.add_axes(frect, zorder=5); pa.set_xticks([]); pa.set_yticks([])
        pa.text(0.5, 0.5, "no data", transform=pa.transAxes, ha="center", va="center", color="gray",
                fontsize=PH_LABEL_FS)
        print(f"inset {label}: placeholder (no run on disk)")
    else:
        p = inset_snapshot(run, ds, fig, frect, vgrids[n], v_refs[n], key=True)
        plots.append(p)
        print(f"inset {run}: snapshot {i:04d}, t = {t_myr(ds):.2f} Myr")
    pax, sc = inset_phase(phists[n], fig, prect, first=(n == 0), xlabel=(n == 1))
    # column label inside the top-left corner of the Sigma inset
    if COLUMN_LABELS:
        fig.text(frect[0] + 0.08 / FIG_W, frect[1] + h - 0.08 / FIG_H, label, color=color, fontsize=PH_LABEL_FS,
                 ha="left", va="top", zorder=11, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=3))
    # vertical guide down the left side of the column, from this run's curve at t_left, in the column's color
    if run is None:
        continue                        # placeholder: no curve to hang a guide line from
    t_r, m_r = series[run]
    y_curve = bb.y0 + bb.height * np.interp(t_left, t_r, m_r) / (M0 / 1e10)
    fig.add_artist(mpl.lines.Line2D([frect[0], frect[0]], [bb.y0 if GUIDE_TO_AXIS else prect[1], y_curve], color=color,
                                    lw=GUIDE_LW,
                                    transform=fig.transFigure, zorder=10))
    if GUIDE_TILT:   # continue from the column's bottom-left corner, slanting to its snapshot time on the time axis
        fig.add_artist(mpl.lines.Line2D([frect[0], bb.x0 + bb.width * t_mark / 20.0], [prect[1], bb.y0], color=color,
                                        lw=GUIDE_LW, transform=fig.transFigure, zorder=10))
    if n != 0:
        continue                        # scale bar only on the first (leftmost) column (all share the same scale)
    # vertical 500 pc scale bar just outside the left edge of the Sigma map, bottom-aligned with it, label to its left
    Lv = SCALE_PC / 1e3 / WIDTH_KPC * h                        # bar length [figure fraction, vertical]
    xb, ym = frect[0] - 0.25 / FIG_W, frect[1] + Lv / 2
    fig.add_artist(mpl.lines.Line2D([xb, xb], [ym - Lv / 2, ym + Lv / 2], color="k", lw=3, transform=fig.transFigure,
                                    zorder=11, solid_capstyle="butt"))
    for ye in (ym - Lv / 2, ym + Lv / 2):
        fig.add_artist(mpl.lines.Line2D([xb - 0.08 / FIG_W, xb + 0.08 / FIG_W], [ye, ye], color="k", lw=3,
                                        transform=fig.transFigure, zorder=11))
    fig.text(xb - 0.15 / FIG_W, ym, f"{SCALE_PC:.0f} pc", ha="right", va="center", rotation=90,
             fontsize=PH_LABEL_FS, zorder=11)
# ---- colorbars stacked top-down (Sigma, f_ion) in the empty upper-left of the main axes, labels to the right
# of each bar, tick labels below. The stack is kept left of / above where the curves reach M_* ~ 1.25 (t ~ 6 Myr).
CB_W_IN, CB_H_IN = 8.5, 0.65                       # bar width / height [in]
CB_X0_IN = 1.0                                     # left edge of the bars, from the left axis [in] (labels go right of them)
CB_TICK_FS = 28                                    # colorbar tick labels (labels are FS)
CB_LABEL_GAP_IN = 0.90                             # gap between each label and its bar [in]
CB_TOP_IN = 0.30                                   # gap between the top axis and the top bar [in]
CB_STEP_IN = 1.60                                  # centre-to-centre spacing of the bars [in] (bar + tick labels + gap)


def side_colorbar(mappable, k, label):
    """k-th bar from the top, ticks below, label left-aligned just right of the bar."""
    if CB_RIGHT:   # bars packed against the right axis (CB_X0_IN margin), labels right-aligned to their left
        x0 = bb.x1 - (CB_X0_IN + CB_W_IN) / FIG_W
    else:
        x0 = bb.x0 + CB_X0_IN / FIG_W
    yc = bb.y1 - (CB_TOP_IN + CB_H_IN / 2 + k * CB_STEP_IN) / FIG_H
    cax = fig.add_axes([x0, yc - CB_H_IN / 2 / FIG_H, CB_W_IN / FIG_W, CB_H_IN / FIG_H])
    cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=CB_TICK_FS, length=8, width=1.5)
    if CB_RIGHT:
        fig.text(x0 - CB_LABEL_GAP_IN / FIG_W, yc, label, ha="right", va="center", fontsize=FS)
    else:
        fig.text(x0 + (CB_W_IN + CB_LABEL_GAP_IN) / FIG_W, yc, label, ha="left", va="center", fontsize=FS)
    return cb


side_colorbar(mpl.cm.ScalarMappable(mpl.colors.LogNorm(*SIGMA_LIM), cmap=CMAP), 0, r"$\Sigma\ [M_\odot\,{\rm pc}^{-2}]$")
side_colorbar(sc, 1, r"$f_{\rm ion}$")

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
fig.savefig(OUTFILE, dpi=200)
print("wrote", OUTFILE)
