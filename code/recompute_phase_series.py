"""Outflow time series through the r = 1 kpc shell, split by phase, at several WNM/WIM thresholds in one pass.

Same method as snapshot_quantities / get_series in make_paper_figs.ipynb (thin shell r0 +- dr, outflowing cells only),
but each snapshot is read once and the phase split is done for every threshold in THRESHOLDS on the hp field
(the H+ mass fraction; fully ionized = X = 0.76, so 0.38 = half of the hydrogen ionized).

Usage:   ~/miniconda3/bin/python recompute_phase_series.py <run>        (run = HD | RHD | SNe | SNeRHD,f | SNeRHD,Zsun)
Writes:  ../../data/cache/series_thr<thr>_<run>.npz  with t, mstar and, per phase (All, CNM, WNM, WIM, HIM), an
         (n_snap, 4) array of (Mdot [Msun/Myr], pdot [Msun km/s/Myr], Edot [erg/Myr], <v_out> [km/s]).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "make_paper_figs.ipynb")))
for key in ("RUNS = {", "def snap_paths", "FIELD_IONFRAC ="):
    src = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code" and key in "".join(c["source"])]
    exec(src[0], globals())

THRESHOLDS = (0.5, 0.38)
PHASES = ["CNM", "WNM", "WIM", "HIM"]
DR_KPC = 0.05
PCMYR, MSUN_KMS2_ERG = 1.0227, 1.989e43     # km/s -> pc/Myr; Msun (km/s)^2 -> erg


def phase_masks(T, fion, thr):
    return dict(CNM=T < T_COLD, WNM=(T >= T_COLD) & (T < T_HOT) & (fion < thr),
                WIM=(T >= T_COLD) & (T < T_HOT) & (fion >= thr), HIM=T >= T_HOT)


def snapshot(run, i):
    ds = load(run, i)
    sph = ds.sphere(ds.domain_center, (R_OUT_KPC + DR_KPC, "kpc"))
    shell = sph["index", "radius"].to("kpc").v > R_OUT_KPC - DR_KPC
    g = lambda f, u: sph[f].to(u).v[shell]
    rho = g(FIELD_DENS, "Msun/pc**3"); dV = g(("gas", "cell_volume"), "pc**3")
    vr = g(("gas", "radial_velocity"), "km/s"); T = g(FIELD_TEMP, "K")
    fion = sph[FIELD_IONFRAC].v[shell]
    e_int = (g(("gas", "specific_thermal_energy"), "km**2/s**2")
             if ("gas", "specific_thermal_energy") in ds.derived_field_list else 0.0 * vr)
    v2 = g(("gas", "velocity_magnitude"), "km/s") ** 2
    dr_pc = 2 * DR_KPC * 1e3
    out = vr > 0
    res = {}
    for thr in THRESHOLDS:
        r = {}
        for name, m in dict(All=np.ones_like(T, bool), **phase_masks(T, fion, thr)).items():
            k = m & out
            mdot = np.sum(rho[k] * vr[k] * dV[k]) / dr_pc
            pdot = np.sum(rho[k] * vr[k] ** 2 * dV[k]) / dr_pc
            edot = np.sum(rho[k] * vr[k] * (0.5 * v2[k] + e_int[k]) * dV[k]) / dr_pc
            vavg = np.sum(rho[k] * vr[k] ** 2 * dV[k]) / np.sum(rho[k] * vr[k] * dV[k]) if k.any() else np.nan
            r[name] = (mdot * PCMYR, pdot * PCMYR, edot * PCMYR * MSUN_KMS2_ERG, vavg)
        res[thr] = r
    try:
        mstar = float(ds.all_data()["nbody", "particle_mass"].to("Msun").sum())
    except Exception:
        mstar = 0.0
    return t_myr(ds), mstar, res


if __name__ == "__main__":
    yt.set_log_level("error")
    run = sys.argv[1]
    t, ms = [], []
    q = {thr: {p: [] for p in ["All"] + PHASES} for thr in THRESHOLDS}
    skipped = [i for i in range(N_SNAP) if not is_valid(run, i)]
    if skipped:
        print(f"[warn] {run}: skipping {len(skipped)} missing/corrupt snapshots: {skipped}", flush=True)
    for i in range(N_SNAP):
        if i in skipped:
            continue
        try:
            ti, mi, res = snapshot(run, i)
        except Exception as e:   # some files pass is_valid but are corrupt inside (e.g. SNeOnly)
            print(f"[warn] {run}: snapshot {i:04d} unreadable ({type(e).__name__}: {e}); skipped", flush=True)
            continue
        t.append(ti); ms.append(mi)
        for thr in THRESHOLDS:
            for p in q[thr]:
                q[thr][p].append(res[thr][p])
        if i % 20 == 0:
            print(f"{run}: snapshot {i:04d} t = {ti:.2f} Myr", flush=True)
    for thr in THRESHOLDS:
        fn = os.path.join(CACHE, f"series_thr{thr:.2f}_{run.replace(',', '_')}.npz")
        np.savez(fn, t=np.array(t), mstar=np.array(ms), threshold=thr, **{p: np.array(v) for p, v in q[thr].items()})
        print("wrote", fn, flush=True)
