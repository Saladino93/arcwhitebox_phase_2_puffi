"""Switch-off ledger for submission 337631 on nets 600-629 (numpy shim raw) plus metered GF on 600-602.
Ratio = geometric mean over nets of (variant raw / control raw), with a 95% CI from the per-net log-ratios; 'worse' = nets whose raw rose."""
import json, math, sys
from pathlib import Path
A = Path(sys.argv[1]); V = json.load(open(A / "variants.json"))
def rows(p): d = json.load(open(p)); return {r["net"]: r for r in d["rows"]}
np30 = {v: rows(A / "np30" / f"{v}.json") for v in V if (A / "np30" / f"{v}.json").exists()}
met3 = {v: rows(A / "met3" / f"{v}.json") for v in V if (A / "met3" / f"{v}.json").exists()}
ctrl = np30["control"]; nets = sorted(ctrl)
ref = json.load(open("results/mse_vs_layer_30nets_600_629.json"))
out = ["| switched off | variant | edit | MSE ratio (geo mean) | 95% CI | worse | mean raw (30 nets) | GF 600-602 | dF vs control |", "|---|---|---|---|---|---|---|---|---|"]
gf0 = sum(r["GF"] for r in met3["control"].values()) / len(met3["control"]) if "control" in met3 else None
for v in V:
    if v not in np30: continue
    r = np30[v]; lr = [math.log(r[n]["raw"] / ctrl[n]["raw"]) for n in nets if n in r]
    m = sum(lr) / len(lr); s = (sum((x - m) ** 2 for x in lr) / max(1, len(lr) - 1)) ** 0.5 / len(lr) ** 0.5
    worse = sum(1 for x in lr if x > 0); mean_raw = sum(r[n]["raw"] for n in nets if n in r) / len(lr)
    gf = sum(x["GF"] for x in met3[v].values()) / len(met3[v]) if v in met3 else None
    edit = "; ".join(f"{f}: {o.strip()} -> {nw.strip()}" for f, o, nw in V[v]) or "(none)"
    out.append(f"| {v} | {v} | `{edit}` | x{math.exp(m):.3f} | [{math.exp(m-1.96*s):.3f}, {math.exp(m+1.96*s):.3f}] | {worse}/{len(lr)} | {mean_raw:.4e} | {gf:.1f} | {(gf-gf0):+.1f} |" if gf else
               f"| {v} | {v} | `{edit}` | x{math.exp(m):.3f} | [{math.exp(m-1.96*s):.3f}, {math.exp(m+1.96*s):.3f}] | {worse}/{len(lr)} | {mean_raw:.4e} | n/a | n/a |")
# verification anchors against the stored receipts (numpy shim, 2026-10-27)
anch = []
for key, v in (("ex_37_submitted_337631", "control"), ("ex_34_submitted_337479", "cap320_eq_337577")):
    if v in np30:
        pn = ref[key]["per_net"]; d = max(abs(np30[v][n]["raw"] / pn[str(n)][-1] - 1) for n in nets if n in np30[v] and str(n) in pn)
        anch.append(f"- {v} vs stored receipt {key}: max |ratio-1| over nets = {d:.2e}")
print("\n".join(out)); print(); print("\n".join(anch))
