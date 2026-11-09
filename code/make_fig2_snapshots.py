"""Snapshot grids: the inset panels of new_fig1 / new_fig3 (make_new_fig3.py) without the M_* curves.

Each column: projected gas surface density with mass-weighted velocity arrows and sinks (top row), and the n-T phase
diagram colored by the mass-weighted mean f_ion (bottom row). Panel code is copied from make_new_fig3.py.

  fig2 (default): fiducial SNe+RHD run at t = 5, 10, 15, 20 Myr (snapshots 50 / 100 / 150 / 200)
  fig4:           HD / RHD / SNe / fiducial at ~10 Myr (snapshot 100, or the nearest valid one)
  fig6:           fiducial (0.01 Zsun) / 0.1 Zsun / Zsun at ~10 Myr. There are no FLASH outputs for the 0.1 Zsun run, only
                  Z1_GRIDS (uniform 128^3 grids of [rho, T, vx, vy, vz] per snapshot, array order [z, y, x]). Its Sigma
                  map is the density summed along z, its arrows the density-weighted mean v_x, v_y along z sampled on
                  the ARROW_N grid, and its phase diagram the n-T histogram colored by the mass-weighted mean f_ion
                  taken from Z1_FION (the astro9 grids, which carry the hp field). The grids carry no sinks; their
                  positions come from Z1_SINKS, read off frame 100 (t = 10.0 Myr) of owzb_tenth_Zsun_zhat_dens.mp4
                  (projection matches the grid Sigma, corr 0.93, offset < 1 cell). No masses, so no size ordering.

Usage:  ~/miniconda3/bin/python make_fig2_snapshots.py [fig2|fig4|fig6]
Writes  ../../outputs/figures/new_fig2_snapshots.png  or  new_fig4_snapshots.png
"""
import json, os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import yt

HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "make_paper_figs.ipynb")))
for key in ("RUNS = {", "def snap_paths", "FIELD_IONFRAC ="):   # imports/RUNS/paths, snapshot helpers, FIELD_* names
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and key in "".join(c["source"])]
    exec(src[0], globals())
N_SNAP = 201                         # the notebook stops at 0199; snapshot 0200 (t = 20 Myr) exists for this run

FIGURE = sys.argv[1] if len(sys.argv) > 1 else "fig2"
assert FIGURE in ("fig2", "fig4", "fig6"), FIGURE
OUTFILE = os.path.join(HERE, "..", "..", "outputs", "figures", f"new_{FIGURE}_snapshots.png")
# columns: (run, snapshot index, title; None = "t = <snapshot time> Myr")
if FIGURE == "fig2":
    COLUMNS = [("SNeRHD,f", i, None) for i in (50, 100, 150, 200)]   # 5, 10, 15, 20 Myr
    V_REF_FIXED = {}
elif FIGURE == "fig6":   # run None = placeholder column (frames and labels only)
    COLUMNS = [("SNeRHD,f", 100, r"$0.01\,Z_\odot$ (fiducial)"), ("Z-1 grid", 100, r"$0.1\,Z_\odot$"),
               ("SNeRHD,Zsun", 100, r"$Z_\odot$")]
    V_REF_FIXED = {"SNeRHD,f": 1000, "Z-1 grid": 1000, "SNeRHD,Zsun": 1000}   # one arrow scale for all columns
else:
    COLUMNS = [("HD", 100, r"$\mathtt{HD}$"), ("RHD", 100, r"$\mathtt{RHD}$"), ("SNe", 100, r"$\mathtt{SNe}$"),
               ("SNeRHD,f", 100, r"$\mathtt{SNeRHD}$ (fiducial)")]
    V_REF_FIXED = {"HD": 500, "RHD": 500}   # km/s key speed for these runs (as in new_fig3) instead of their p95
NCOL = len(COLUMNS)
# 0.1 Zsun run: uniform grids only, shape (200 snapshots, 5 fields [rho cgs, T, vx, vy, vz], 128, 128, 128), array order
# [z, y, x] (checked: sum over axis 0 matches the fiducial yt projection along z at 2 and 4 Myr, corr 0.999, mass to 1%)
Z1_GRIDS = os.path.expanduser("~/Downloads/SNeRHDZ-1grids.npy")
# f_ion for the 0.1 Zsun run: astro9 grid file, shape (3 metallicities [fid, 10x, 100x], 6 fields [rho, T, hp, vx, vy, vz],
# 128, 128, 128) in [x, y, z] order. Checked: its [1, 0].transpose(2, 1, 0) equals Z1_GRIDS[100, 0] exactly.
Z1_FION = (os.path.expanduser("~/Documents/teaching/astro9/t10_rhoTvs_grid.fits"), 100)   # (file, matching snapshot)
Z1_SINKS = (os.path.join(HERE, "..", "..", "data", "cache", "sinks_SNeRHDZ-1_snap100_from_movie.txt"), 100)  # x, y kpc
MU_MH = 0.9928 * 1.6726e-24          # g; yt's number_density = rho / (0.9928 m_H) for these FLASH runs (constant)

# layout [inches]
PANEL_IN = 6.0                       # side of the Sigma maps (= width of the phase diagrams)
PH_ASPECT = 0.65                     # phase-diagram height / width
COL_GAP_IN = 0.25                    # gap between columns
MARGIN = dict(left=1.9, right=2.9, bottom=1.45, top=0.9)
FIG_W = MARGIN["left"] + NCOL * PANEL_IN + (NCOL - 1) * COL_GAP_IN + MARGIN["right"]
TROW = FIGURE == "fig6"              # extra row: volume-averaged T along z
XZROW = False                        # extra row: x-z Sigma (projection along y); switched off
FIG_H = MARGIN["bottom"] + PANEL_IN * (1 + PH_ASPECT + TROW + XZROW) + MARGIN["top"]

FS = 32                              # titles, axis and colorbar labels
TICK_FS = 26
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": FS})

# Sigma maps (as make_new_fig3.py)
CMAP = "magma_r"
SIGMA_LIM = (1e1, 1e5)
STAR_COLOR, STAR_SIZE = "cyan", 350
WIDTH_KPC = 2.8
SCALE_PC = 500.0
ARROW_N = 12
ARROW_KEY_FS = 24
ARROW_REF_IN = 0.50
ARROW_WIDTH = 0.004
ARROW_HEAD = (4.5, 5.0, 4.5)

# volume-averaged temperature maps (TROW)
T_CMAP = "cubehelix"
T_LIM = (1e4, 1e8)                   # data at snapshot 100: 3.8e3-6.1e7 K, medians 4-6e6 K
T_BUFF = 512


def temp_map_yt(ds):
    """Volume-weighted mean T along z on a T_BUFF^2 grid over the Sigma-map field of view."""
    prj = ds.proj(FIELD_TEMP, "z", weight_field=("index", "cell_volume"))
    frb = prj.to_frb((WIDTH_KPC, "kpc"), (T_BUFF, T_BUFF), center=ds.domain_center)
    return frb[FIELD_TEMP].to("K").v


def map_panel(fig, rect, img, cmap, lim):
    ax = fig.add_axes(rect); ax.set_xticks([]); ax.set_yticks([])
    half = WIDTH_KPC / 2
    ax.imshow(img, origin="lower", extent=(-half, half, -half, half), cmap=cmap,
              norm=mpl.colors.LogNorm(*lim), interpolation="nearest")
    return ax


def temp_panel(fig, rect, Tmap, label, vgrid, v_ref, sinks):
    """<T>_V map with the same velocity arrows and sink stars as the Sigma map above it."""
    map_panel(fig, rect, Tmap, T_CMAP, T_LIM)
    ov = add_arrows(fig, rect, vgrid, v_ref)
    add_sinks(ov, *sinks)
    print(f"  <T>_V {label}: {Tmap.min():.3g}-{Tmap.max():.3g} K (median {np.median(Tmap):.3g})")


def xz_maps_yt(ds):
    """Sigma [Msun/pc^2] projected along y and its density-weighted v_x, v_z grid, as [z, x] arrays (x right, z up)."""
    c = ds.domain_center
    sig = ds.proj(FIELD_DENS, "y").to_frb((WIDTH_KPC, "kpc"), (1024, 1024), center=c)[FIELD_DENS]
    sig = sig.to("Msun/pc**2").v.T                 # yt's y-projection images are [x, z]; checked against the z-projection
    prj = ds.proj([("gas", "velocity_x"), ("gas", "velocity_z")], "y", weight_field=("gas", "density"))
    frb = prj.to_frb((WIDTH_KPC, "kpc"), (ARROW_N, ARROW_N), center=c)
    vx = frb["gas", "velocity_x"].to("km/s").v.T; vz = frb["gas", "velocity_z"].to("km/s").v.T
    cc = (np.arange(ARROW_N) + 0.5) / ARROW_N * WIDTH_KPC - WIDTH_KPC / 2
    X, Z = np.meshgrid(cc, cc)
    return sig, (X, Z, vx, vz)


def xz_panel(fig, rect, sig, vgrid, label):
    """x-z Sigma map with velocity arrows (no sinks: the 0.1 Zsun sink positions have no z)."""
    map_panel(fig, rect, sig, CMAP, SIGMA_LIM)
    v_ref = nice_round(np.percentile(np.hypot(vgrid[2], vgrid[3]), 95))
    add_arrows(fig, rect, vgrid, v_ref)
    print(f"  x-z Sigma {label}: {sig.min():.3g}-{sig.max():.3g} Msun/pc^2, arrow key {v_ref:g} km/s")

# phase diagrams (as make_new_fig3.py)
PH_NBINS = 100
PH_CMAP = "coolwarm_r"
PH_FION_LIM = (1e-2, 0.76)           # the hp field is the H+ mass fraction: fully ionized = X = 0.76
PH_XLIM = (2e-4, 2e4)
PH_YLIM = (2e1, 1e9)


def velocity_grid(ds):
    prj = ds.proj([("gas", "velocity_x"), ("gas", "velocity_y")], "z", weight_field=("gas", "density"))
    frb = prj.to_frb((WIDTH_KPC, "kpc"), (ARROW_N, ARROW_N), center=ds.domain_center)
    vx = frb["gas", "velocity_x"].to("km/s").v; vy = frb["gas", "velocity_y"].to("km/s").v
    c = (np.arange(ARROW_N) + 0.5) / ARROW_N * WIDTH_KPC - WIDTH_KPC / 2
    X, Y = np.meshgrid(c, c)
    return X, Y, vx, vy


def nice_round(v):
    k = np.floor(np.log10(v))
    return min((m * 10 ** k for m in (1, 2, 5, 10)), key=lambda x: abs(np.log(x / v)))


def add_arrows(fig, rect, vgrid, v_ref):
    """Velocity arrows (v_ref km/s = ARROW_REF_IN) and key on a transparent overlay axes in kpc; returns it."""
    ov = fig.add_axes(rect, zorder=6); ov.set_axis_off()
    half = WIDTH_KPC / 2
    ov.set(xlim=(-half, half), ylim=(-half, half))
    q = ov.quiver(*vgrid, color="k", scale=v_ref / ARROW_REF_IN, scale_units="inches", width=ARROW_WIDTH,
                  headwidth=ARROW_HEAD[0], headlength=ARROW_HEAD[1], headaxislength=ARROW_HEAD[2], zorder=2)
    ov.add_patch(mpl.patches.Rectangle((0.615, 0.015), 0.37, 0.14, transform=ov.transAxes, facecolor="white",
                                       alpha=0.8, edgecolor="none", zorder=4))
    qk = ov.quiverkey(q, 0.80, 0.05, v_ref, f"{v_ref:g} km/s", labelpos="N", coordinates="axes",
                      fontproperties={"size": ARROW_KEY_FS}, labelsep=0.08)
    qk.set_zorder(5)
    return ov


def sigma_panel(ds, fig, rect, vgrid, v_ref):
    p = yt.ProjectionPlot(ds, "z", FIELD_DENS, width=(WIDTH_KPC, "kpc"))
    p.set_unit(FIELD_DENS, "Msun/pc**2"); p.set_zlim(FIELD_DENS, *SIGMA_LIM)
    p.set_cmap(FIELD_DENS, CMAP); p.set_buff_size(1024)
    p.hide_axes(draw_frame=True)
    ax = fig.add_axes(rect)
    cax = fig.add_axes([0, 0, 0.01, 0.01]); cax.set_visible(False)
    pl = p.plots[FIELD_DENS]
    pl.figure, pl.axes, pl.cax = fig, ax, cax
    p._setup_plots()
    ax.set_position(rect); fig.set_size_inches(FIG_W, FIG_H)   # yt resizes axes and canvas; restore both
    p._ds_keepalive = ds
    ov = add_arrows(fig, rect, vgrid, v_ref)
    add_sinks(ov, *sinks_xy(ds))
    return p


def sinks_xy(ds):
    """Sink x, y [kpc, relative to the box centre] inside the field of view, sorted by mass; empty if none."""
    half = WIDTH_KPC / 2
    try:
        ad = ds.all_data(); c = ds.domain_center.to("kpc").v
        x = ad["nbody", "particle_position_x"].to("kpc").v - c[0]
        y = ad["nbody", "particle_position_y"].to("kpc").v - c[1]
        msink = ad["nbody", "particle_mass"].to("Msun").v
    except Exception:
        return np.empty(0), np.empty(0)
    k = (np.abs(x) < half) & (np.abs(y) < half)
    order = np.argsort(msink[k])
    return x[k][order], y[k][order]


def add_sinks(ov, x, y):
    if len(x):
        ov.scatter(x, y, s=STAR_SIZE, marker="*", color=STAR_COLOR, edgecolors="k", linewidths=0.8, zorder=3)


def style_phase(ax, first, xlabel):
    for T0 in (T_COLD, T_HOT):
        ax.axhline(T0, color="k", ls="--", lw=1.5)
    ax.set(xscale="log", yscale="log", xlim=PH_XLIM, ylim=PH_YLIM)
    ax.set_xticks([1e-3, 1e0, 1e3]); ax.set_yticks([1e3, 1e5, 1e7])
    ax.minorticks_off()
    ax.tick_params(direction="in", length=8, width=1.5, labelsize=TICK_FS, top=True, right=True)
    if first:
        ax.set_ylabel(r"$T$ [K]")
    else:
        ax.set_yticklabels([])
    if xlabel:
        ax.set_xlabel(r"$n\ [{\rm cm^{-3}}]$")


def phase_panel(ds, fig, rect, first, xlabel):
    norm = mpl.colors.LogNorm(*PH_FION_LIM)
    if ds is None:   # placeholder: empty axes with the same limits, ticks and phase lines
        ax = fig.add_axes(rect)
        style_phase(ax, first, xlabel)
        return mpl.cm.ScalarMappable(norm=norm, cmap=PH_CMAP)
    ad = ds.all_data()
    n = ad[FIELD_NUM].to("cm**-3").v; T = ad[FIELD_TEMP].to("K").v; f = ad[FIELD_IONFRAC].v
    m = ad["gas", "mass"].to("Msun").v
    xe = np.logspace(*np.log10(PH_XLIM), PH_NBINS + 1); ye = np.logspace(*np.log10(PH_YLIM), PH_NBINS + 1)
    M, _, _ = np.histogram2d(n, T, bins=(xe, ye), weights=m)
    MF, _, _ = np.histogram2d(n, T, bins=(xe, ye), weights=m * f)
    with np.errstate(invalid="ignore", divide="ignore"):
        fbar = np.where(M > 0, MF / M, np.nan)
    print(f"  phase: {100 * M.sum() / m.sum():.2f}% of the mass inside the axis limits")
    ax = fig.add_axes(rect)
    ax.pcolormesh(xe, ye, np.ma.masked_invalid(np.clip(fbar, *PH_FION_LIM)).T, cmap=PH_CMAP, norm=norm,
                  rasterized=True)
    style_phase(ax, first, xlabel)
    return mpl.cm.ScalarMappable(norm=norm, cmap=PH_CMAP)


fig = plt.figure(figsize=(FIG_W, FIG_H))
w, h, hp = PANEL_IN / FIG_W, PANEL_IN / FIG_H, PH_ASPECT * PANEL_IN / FIG_H
y_ph = MARGIN["bottom"] / FIG_H
y_T = y_ph + hp                      # temperature row (only used when TROW)
y_xz = y_T + (h if TROW else 0)      # x-z Sigma row (only used when XZROW)
y_sig = y_xz + (h if XZROW else 0)
plots = []
for col, (run, i0, title) in enumerate(COLUMNS):
    x0 = (MARGIN["left"] + col * (PANEL_IN + COL_GAP_IN)) / FIG_W
    if run == "Z-1 grid":   # Sigma from the uniform density grid; empty phase panel (no f_ion in the grids)
        g = np.load(Z1_GRIDS, mmap_mode="r")[i0]
        rho, T = np.asarray(g[0]), np.asarray(g[1])                                 # [z, y, x]; g/cm^3, K
        vx, vy = np.asarray(g[2]) / 1e5, np.asarray(g[3]) / 1e5                     # km/s (index 2 = v_x, 3 = v_y)
        dx_cm = WIDTH_KPC * 3.0857e21 / rho.shape[0]
        sig = rho.sum(axis=0) * dx_cm / 1.989e33 * 3.0857e18 ** 2                  # Msun/pc^2, [y, x]
        pa = fig.add_axes([x0, y_sig, w, h]); pa.set_xticks([]); pa.set_yticks([])
        half = WIDTH_KPC / 2
        pa.imshow(sig, origin="lower", extent=(-half, half, -half, half), cmap=CMAP,
                  norm=mpl.colors.LogNorm(*SIGMA_LIM), interpolation="nearest")
        # arrows: density-weighted mean along z, sampled at the ARROW_N x ARROW_N bin centres (as yt's to_frb)
        pvx, pvy = (rho * vx).sum(0) / rho.sum(0), (rho * vy).sum(0) / rho.sum(0)
        k = ((np.arange(ARROW_N) + 0.5) / ARROW_N * rho.shape[1]).astype(int)
        c = (np.arange(ARROW_N) + 0.5) / ARROW_N * WIDTH_KPC - half
        X, Y = np.meshgrid(c, c)
        gvx, gvy = pvx[np.ix_(k, k)], pvy[np.ix_(k, k)]
        v_ref = V_REF_FIXED.get("Z-1 grid") or nice_round(np.percentile(np.hypot(gvx, gvy), 95))
        ov = add_arrows(fig, [x0, y_sig, w, h], (X, Y, gvx, gvy), v_ref)
        sinks = np.loadtxt(Z1_SINKS[0], unpack=True) if i0 == Z1_SINKS[1] else (np.empty(0), np.empty(0))
        add_sinks(ov, *sinks)   # sink positions read off the movie frame at the same time
        if XZROW:
            # x-z: Sigma summed along y (axis 1 of [z, y, x]); arrows = density-weighted mean v_x, v_z along y
            sig_xz = rho.sum(axis=1) * dx_cm / 1.989e33 * 3.0857e18 ** 2                # [z, x]
            vz = np.asarray(g[4]) / 1e5
            qvx, qvz = (rho * vx).sum(1) / rho.sum(1), (rho * vz).sum(1) / rho.sum(1)
            xz_panel(fig, [x0, y_xz, w, h], sig_xz, (X, Y, qvx[np.ix_(k, k)], qvz[np.ix_(k, k)]), "0.1 Zsun grid")
        if TROW:
            # uniform grid: the volume average along z is the plain mean
            temp_panel(fig, [x0, y_T, w, h], T.mean(axis=0), "0.1 Zsun grid", (X, Y, gvx, gvy), v_ref, sinks)
        # phase diagram from rho and T, colored by the mass-weighted mean f_ion (hp) from the astro9 grid
        assert i0 == Z1_FION[1], "the f_ion grid on disk is for snapshot %d only" % Z1_FION[1]
        from astropy.io import fits
        fg = fits.getdata(Z1_FION[0])[1]                                            # 10x run, [x, y, z]
        assert np.array_equal(fg[0].transpose(2, 1, 0), rho), "f_ion grid does not match the density grid"
        f = fg[2].transpose(2, 1, 0).astype(float).ravel()
        n = (rho / MU_MH).ravel(); m = (rho * dx_cm ** 3 / 1.989e33).ravel()
        xe = np.logspace(*np.log10(PH_XLIM), PH_NBINS + 1); ye = np.logspace(*np.log10(PH_YLIM), PH_NBINS + 1)
        M, _, _ = np.histogram2d(n, T.ravel(), bins=(xe, ye), weights=m)
        MF, _, _ = np.histogram2d(n, T.ravel(), bins=(xe, ye), weights=m * f)
        with np.errstate(invalid="ignore", divide="ignore"):
            fbar = np.where(M > 0, MF / M, np.nan)
        print(f"  phase: {100 * M.sum() / m.sum():.2f}% of the mass inside the axis limits")
        ax_ph = fig.add_axes([x0, y_ph, w, hp])
        ax_ph.pcolormesh(xe, ye, np.ma.masked_invalid(np.clip(fbar, *PH_FION_LIM)).T, cmap=PH_CMAP,
                         norm=mpl.colors.LogNorm(*PH_FION_LIM), rasterized=True)
        style_phase(ax_ph, first=(col == 0), xlabel=False)
        fig.text(x0 + w / 2, y_sig + h + 0.2 / FIG_H, title, ha="center", va="bottom")
        print(f"column {col}: 0.1 Zsun grid snapshot {i0}, arrow key {v_ref:g} km/s, Sigma {sig.min():.3g}-{sig.max():.3g} Msun/pc^2, "
              f"M = {sig.sum() * (WIDTH_KPC * 1e3 / sig.shape[0]) ** 2:.3g} Msun")
        continue
    if run is None:   # placeholder column: empty framed Sigma panel + empty phase axes
        pa = fig.add_axes([x0, y_sig, w, h]); pa.set_xticks([]); pa.set_yticks([])
        pa.text(0.5, 0.5, "no data", transform=pa.transAxes, ha="center", va="center", color="gray")
        phase_panel(None, fig, [x0, y_ph, w, hp], first=(col == 0), xlabel=False)
        fig.text(x0 + w / 2, y_sig + h + 0.2 / FIG_H, title, ha="center", va="bottom")
        print(f"column {col}: placeholder (no run on disk)")
        continue
    i = nearest_valid(run, i0)
    ds = load(run, i)
    t = t_myr(ds)
    vg = velocity_grid(ds)
    v_ref = V_REF_FIXED.get(run) or nice_round(np.percentile(np.hypot(vg[2], vg[3]), 95))
    print(f"{run} snapshot {i:04d}: t = {t:.2f} Myr, arrow key {v_ref:g} km/s")
    plots.append(sigma_panel(ds, fig, [x0, y_sig, w, h], vg, v_ref))
    if XZROW:
        xz_panel(fig, [x0, y_xz, w, h], *xz_maps_yt(ds), run)
    if TROW:
        temp_panel(fig, [x0, y_T, w, h], temp_map_yt(ds), run, vg, v_ref, sinks_xy(ds))
    sm_f = phase_panel(ds, fig, [x0, y_ph, w, hp], first=(col == 0), xlabel=False)
    fig.text(x0 + w / 2, y_sig + h + 0.2 / FIG_H, title or rf"$t = {t:.0f}$ Myr", ha="center", va="bottom")
    if col == 0:   # vertical 500 pc scale bar left of the first Sigma map, bottom-aligned with it
        Lv = SCALE_PC / 1e3 / WIDTH_KPC * h
        xb = x0 - 0.25 / FIG_W
        fig.add_artist(mpl.lines.Line2D([xb, xb], [y_sig, y_sig + Lv], color="k", lw=3, transform=fig.transFigure,
                                        solid_capstyle="butt"))
        for ye in (y_sig, y_sig + Lv):
            fig.add_artist(mpl.lines.Line2D([xb - 0.08 / FIG_W, xb + 0.08 / FIG_W], [ye, ye], color="k", lw=3,
                                            transform=fig.transFigure))
        fig.text(xb - 0.15 / FIG_W, y_sig + Lv / 2, f"{SCALE_PC:.0f} pc", ha="right", va="center", rotation=90,
                 fontsize=TICK_FS)

# one n label centred under the whole grid
grid_w = (NCOL * PANEL_IN + (NCOL - 1) * COL_GAP_IN) / FIG_W
fig.text(MARGIN["left"] / FIG_W + grid_w / 2, 0.15 / FIG_H, r"$n\ [{\rm cm^{-3}}]$", ha="center", va="bottom")

# vertical colorbars right of the grid, one per row
cb_x = (MARGIN["left"] + NCOL * PANEL_IN + (NCOL - 1) * COL_GAP_IN + 0.3) / FIG_W
cb_w = 0.35 / FIG_W
cb_gap = 0.2 / FIG_H                 # half the gap between the two bars, so their end tick labels don't collide
cb1 = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.LogNorm(*SIGMA_LIM), cmap=CMAP),
                   cax=fig.add_axes([cb_x, (y_xz if XZROW else y_sig) + cb_gap, cb_w, (2 if XZROW else 1) * h - cb_gap]))
cb1.set_label(r"$\Sigma\ [M_\odot\,{\rm pc}^{-2}]$")
if TROW:
    cbT = fig.colorbar(mpl.cm.ScalarMappable(mpl.colors.LogNorm(*T_LIM), cmap=T_CMAP),
                       cax=fig.add_axes([cb_x, y_T + cb_gap, cb_w, h - 2 * cb_gap]))
    cbT.set_label(r"$\langle T \rangle_V$ [K]")
    cbT.ax.tick_params(labelsize=TICK_FS, length=8, width=1.5)
cb2 = fig.colorbar(sm_f, cax=fig.add_axes([cb_x, y_ph, cb_w, hp - cb_gap]))
cb2.set_label(r"H$^+$ mass fraction", fontsize=TICK_FS)
cb2.set_ticks([0.01, 0.1, 0.76]); cb2.set_ticklabels(["0.01", "0.1", "0.76"]); cb2.ax.minorticks_off()
for cb in (cb1, cb2):
    cb.ax.tick_params(labelsize=TICK_FS, length=8, width=1.5)

os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
fig.savefig(OUTFILE, dpi=200)
print("wrote", OUTFILE)
