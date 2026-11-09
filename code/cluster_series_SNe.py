"""Convert the cluster outflow_rates_SNeOnly.npy into the series format of recompute_phase_series.py.

Most SNeOnly snapshots on this computer are missing or corrupt (124 of 200), so the SNe run uses the outflow rates
computed on the cluster instead. Layout of data/outflow_rates/outflow_rates_<run>.npy, (5, 6, 3, 200):
  quantity: 0 <v> [pc/s], 1 Mdot [Msun/s], 2 pdot [Msun pc/s^2], 3 Edot_kin, 4 Edot_th [Msun pc^2/s^3]
  phase:    0 All, 1 HIM, 2 WIM, 3 WNM, 4 CNM, 5 (unused)
  radius:   index 1 = 1 kpc;  time: index i = snapshot i (t = i / 10 Myr)
Checked against the local fiducial series (threshold 0.5): time-median All values agree to ~10%
(Mdot 1.58e8 vs 1.67e8 Msun/Myr, <v> 1040 km/s, Edot 4.0e57 vs 4.8e57 erg/Myr); per-snapshot correlation ~0.8.
The WIM/WNM split is baked in on the cluster (old 0.5 cut on hp), so the same values are written for both
thresholds; without radiation there is almost no WIM, so the threshold barely matters for this run.

Usage:  ~/miniconda3/bin/python cluster_series_SNe.py
Writes  ../../data/cache/series_thr0.50_SNe.npz and series_thr0.38_SNe.npz
"""
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
PC_KM, MYR_S, MSUN_G, PC_CM = 3.0857e13, 3.15576e13, 1.989e33, 3.0857e18
R_IDX = 1
PH = dict(All=0, HIM=1, WIM=2, WNM=3, CNM=4)

cl = np.load(os.path.join(DATA, "outflow_rates", "outflow_rates_SNeOnly.npy"))[:, :, R_IDX, :]   # (q, phase, t)
t = np.arange(cl.shape[-1]) / 10.0
out = dict(t=t, mstar=np.full_like(t, np.nan))
for name, j in PH.items():
    v = cl[0, j] * PC_KM                                              # km/s
    mdot = cl[1, j] * MYR_S                                           # Msun/Myr
    pdot = cl[2, j] * PC_KM * MYR_S                                   # Msun km/s / Myr
    edot = (cl[3, j] + cl[4, j]) * MSUN_G * PC_CM ** 2 * MYR_S        # erg/Myr (kinetic + thermal)
    out[name] = np.stack([mdot, pdot, edot, np.where(mdot > 0, v, np.nan)], axis=1)
for thr in (0.5, 0.38):
    fn = os.path.join(DATA, "cache", f"series_thr{thr:.2f}_SNe.npz")
    np.savez(fn, threshold=thr, source="cluster outflow_rates_SNeOnly.npy (WIM/WNM split at the old 0.5 cut)", **out)
    print("wrote", fn)
m = (t >= 6.7) & (t <= 20)
print("NaN fraction in 6.7-20 Myr (All Mdot):", np.isnan(out["All"][m, 0]).mean())
for name in PH:
    print(f"{name}: M_out(6.7-20) = {np.trapezoid(np.nan_to_num(out[name][m, 0]), t[m]):.3g} Msun")
