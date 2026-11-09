"""Fig 8 (A_V and M_UV vs time) restacked vertically from the existing fig8.png, without redrawing anything.

The data/code behind fig8.png are not on this computer, so this only rearranges the original pixels: each panel's
frame, contents, ticks and tick labels are copied unchanged; the titles ("A_V [mag]", "M_UV [mag]") are rotated 90 deg
and placed as y labels; "time [Myr]" is placed once under the bottom panel. Pixel boxes were measured on fig8.png
(1920 x 815): frames x 154-911 / 1085-1843, y 88-639; titles y 18-85; tick labels to y 706; "time [Myr]" y 747-805.

Usage:  ~/miniconda3/bin/python make_fig8_stacked.py   (run from code/)
Writes  ../../outputs/figures/fig8_stacked.png
"""
import os
import numpy as np, matplotlib.image as mpimg, matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
im = mpimg.imread(os.path.join(HERE, "..", "fig8.png"))[..., :3].copy()
assert im.shape[:2] == (815, 1920), "fig8.png changed size; re-measure the pixel boxes"
Y0, Y1 = 55, 711                                   # block rows: top tick label (y 57) to bottom tick labels (y 706)
TL = (18, 86, 389, 677); TR = (18, 86, 1294, 1635)   # title boxes (y0, y1, x0, x1), exclusive ends
L = im[Y0:Y1, 20:955].copy(); L[max(0, TL[0] - Y0):TL[1] - Y0, TL[2] - 20:TL[3] - 20] = 1      # blank the moved titles
R = im[Y0:Y1, 955:1880].copy(); R[max(0, TR[0] - Y0):TR[1] - Y0, TR[2] - 955:TR[3] - 955] = 1
# rows 86-87 are the frame line's anti-aliasing, but the "g" descenders of the titles reach into them: restore those
# rows under each title from the plain frame edge just left of it (same rows, no text)
for blk, x0b, (y0, y1, x0, x1) in ((L, 20, TL), (R, 955, TR)):
    blk[86 - Y0:88 - Y0, x0 - x0b:x1 - x0b] = im[86:88, x0 - 40:x0 - 39]
tL = np.rot90(im[TL[0]:TL[1], TL[2]:TL[3]], 1); tR = np.rot90(im[TR[0]:TR[1], TR[2]:TR[3]], 1)
tT = im[745:808, 860:1140]                          # "time [Myr]"
fL, fR = 154 - 20, 1085 - 955                       # frame left edge inside each block
off = fL - fR
pad, gap, vgap = 10, 18, 20
lab_w = max(tL.shape[1], tR.shape[1]); xb0 = pad + lab_w + gap
W = xb0 + max(L.shape[1], off + R.shape[1]) + 10
H = 10 + L.shape[0] + vgap + R.shape[0] + 12 + tT.shape[0] + 10
C = np.ones((H, W, 3), dtype=im.dtype)


def paste(a, y, x):
    C[y:y + a.shape[0], x:x + a.shape[1]] = np.minimum(C[y:y + a.shape[0], x:x + a.shape[1]], a)


yA = 10; paste(L, yA, xb0)
yB = yA + L.shape[0] + vgap; paste(R, yB, xb0 + off)
fc = (88 - Y0 + 639 - Y0) // 2                      # frame centre row inside a block
for t, y in ((tL, yA), (tR, yB)):
    paste(t, y + fc - t.shape[0] // 2, pad + (lab_w - t.shape[1]) // 2)
xc = xb0 + fL + (911 - 154) // 2
paste(tT, yB + R.shape[0] + 12, xc - tT.shape[1] // 2)
out = os.path.join(HERE, "..", "..", "outputs", "figures", "fig8_stacked.png")
plt.imsave(out, C)
okA = np.array_equal(C[yA + 33:yA + 585, xb0 + fL:xb0 + fL + 758], im[88:640, 154:912])
okB = np.array_equal(C[yB + 33:yB + 585, xb0 + off + fR:xb0 + off + fR + 759], im[88:640, 1085:1844])
print("wrote", out, C.shape[1::-1], "| frames + interiors pixel-identical:", okA, okB)
