"""Outflow phase summary (paper Figs 5 and 7) from the local phase series made by recompute_phase_series.py.

Bulk values per run and phase over T0..T1 (as _bulk in make_paper_figs.ipynb): <v_out> = mean of the per-snapshot
mass-flux-weighted velocity (error bar = std over time), and M_out, p_out, E_out = time integrals (trapezoid over the
snapshots on disk) of Mdot, pdot, Edot.

Usage:  ~/miniconda3/bin/python make_phase_figs.py [threshold]        (default 0.38)
Writes  ../../outputs/figures/phases_runs_thr<thr>.png, phases_Z_thr<thr>.png and
        ../../outputs/phase_tables/bulk_thr<thr>.csv
"""
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
CACHE = os.path.join(ROOT, "data", "cache")
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.38
T0, T1 = 6.7, 20.0                    # integration window [Myr], as in the paper's M_out definition
V_ESC = 665.0                         # km/s at r = 1 kpc (paper text, baryons + DM halo)
PHASES = ["All", "HIM", "WIM", "WNM", "CNM"]
STYLE = {"All": dict(color="k"), "HIM": dict(color="hotpink"), "WIM": dict(color="g", mfc="none"),
         "WNM": dict(color="g"), "CNM": dict(color="cornflowerblue")}
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix", "font.size": 16})


def load_series(run):
    return dict(np.load(os.path.join(CACHE, f"series_thr{THR:.2f}_{run.replace(',', '_')}.npz")))


def bulk(run):
    s = load_series(run); t = s["t"]; m = (t >= T0) & (t <= T1)
    out = {}
    for ph in PHASES:
        a = s[ph][m]
        out[ph] = dict(v=np.nanmean(a[:, 3]), vsig=np.nanstd(a[:, 3]), M=np.trapezoid(a[:, 0], t[m]),
                       p=np.trapezoid(a[:, 1], t[m]), E=np.trapezoid(a[:, 2], t[m]))
    return out, int(m.sum())


def e_sne():
    d = dict(np.load(os.path.join(CACHE, "snrate_SNeRHD_f.npz")))
    m = (d["t"] >= T0) & (d["t"] <= T1)
    return np.trapezoid(d["ndot"][m], d["t"][m]) * 1e51          # N_SNe(T0..T1) x 1e51 erg


def figure(cols, outfile, xlabel=None):
    """cols: list of (label, run or None); None = placeholder column (no recomputed data)."""
    fig, axs = plt.subplots(1, 4, figsize=(20, 5.2))
    titles = [r"$\overline{\langle v_{\rm out}\rangle}$", r"$M_{\rm out}$", r"$p_{\rm out}$", r"$E_{\rm out}$"]
    ylabels = [r"[km s$^{-1}$]", r"[$M_\odot$]", r"[$M_\odot$ km s$^{-1}$]", r"[erg]"]   # units on the y axes
    for x, (lab, run) in enumerate(cols):
        if run is None:
            for ax in axs:
                ax.text(x, 0.5, "not\nrecomputed", transform=mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes),
                        ha="center", va="center", color="gray", fontsize=12)
            continue
        b, _ = bulk(run)
        for ph in PHASES[::-1]:
            st = STYLE[ph]
            axs[0].errorbar(x, b[ph]["v"], yerr=b[ph]["vsig"], fmt="o", ms=12, color=st["color"],
                            mfc=st.get("mfc", st["color"]), label=ph if x == len(cols) - 1 else None)
            for ax, key in zip(axs[1:], "M p E".split()):
                if b[ph][key] > 0:
                    ax.plot(x, b[ph][key], "o", ms=12, color=st["color"], mfc=st.get("mfc", st["color"]))
    axs[0].axhline(V_ESC, color="lightgray", zorder=0); axs[0].text(0.03, V_ESC + 40, r"$v_{\rm esc}$", color="gray",
                                                                     transform=axs[0].get_yaxis_transform())
    axs[0].axhline(0, color="gray", ls="--", lw=1, alpha=0.6, zorder=0)
    E = e_sne()
    axs[3].axhline(E, color="lightgray", zorder=0); axs[3].text(0.05, E * 0.6, r"$E_{\rm SNe}$", color="gray",
                                                                transform=axs[3].get_yaxis_transform())
    for ax, ttl, yl in zip(axs, titles, ylabels):
        ax.set_title(ttl); ax.set_ylabel(yl, labelpad=2); ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[0] for c in cols])
        ax.set_xlim(-0.6, len(cols) - 0.4)
    for ax in axs[1:]:
        ax.set_yscale("log")
    axs[1].set_ylim(1e8, 5e9); axs[2].set_ylim(1e11, 1e14); axs[3].set_ylim(5e56, 1e59)
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[::-1], labels[::-1], loc="upper left", fontsize=12)
    if xlabel:
        fig.supxlabel(xlabel)
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    print("wrote", outfile)


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "outputs", "phase_tables"), exist_ok=True)
    runs = ["HD", "RHD", "SNe", "SNeRHD,f", "SNeRHD,Zsun"]
    rows = ["run,n_snap,phase,v_mean_kms,v_std_kms,M_out_Msun,p_out_Msun_kms,E_out_erg,M_frac"]
    for r in runs:
        b, n = bulk(r)
        for ph in PHASES:
            rows.append(f"{r},{n},{ph},{b[ph]['v']:.1f},{b[ph]['vsig']:.1f},{b[ph]['M']:.4g},{b[ph]['p']:.4g},"
                        f"{b[ph]['E']:.4g},{b[ph]['M'] / b['All']['M']:.3f}")
    fn = os.path.join(ROOT, "outputs", "phase_tables", f"bulk_thr{THR:.2f}.csv")
    open(fn, "w").write("\n".join(rows) + "\n"); print("wrote", fn)
    print(f"E_SNe({T0}-{T1} Myr) = {e_sne():.3g} erg")
    tt = lambda s: rf"$\mathtt{{{s}}}$"
    figure([(tt("HD"), "HD"), (tt("RHD"), "RHD"), (tt("SNe"), "SNe"), (tt("SNeRHD"), "SNeRHD,f")],
           os.path.join(ROOT, "outputs", "figures", f"phases_runs_thr{THR:.2f}.png"))
    figure([(r"$10^{-2}_{\rm fid}$", "SNeRHD,f"), (r"$10^{-1}$", None), ("1", "SNeRHD,Zsun")],
           os.path.join(ROOT, "outputs", "figures", f"phases_Z_thr{THR:.2f}.png"), xlabel=r"Z [$Z_\odot$]")
