"""Switch-off variants of submission 337631: the exact uploaded files plus one declared edit each (each pattern must match once).
Dropped after a net-600 smoke test: SLICE_UROW=False (constant unused in this version: output bitwise identical) and
STAR_START/STAR_TERM off (no-op: PK4 owns every star age). sphere_off = Gaussian input: single-site kappa4 scale 2 (not 2-4/(n+2)), zero layer-0 input core, radius factor 1 (a 2-edit version keeping the input core is inconsistent: x7.7)."""
import shutil, sys, json
from pathlib import Path
A = Path(sys.argv[1]); SRC = Path(sys.argv[2]); ROOTS = A / "roots"
V = {
 "control": [],
 "cap320_eq_337577": [("bank.py", "kmax = max(1, min(ncol, int(M.shape[0]), n // 4))", "kmax = max(1, min(ncol, int(M.shape[0]), 5 * n // 16))")],
 "sphere_off_gaussian_input": [("estimator.py", 'stfscale = _scalar(xp.subtract(2.0, xp.divide(_scalar(4.0), dim["n+2"])), dtype)', 'stfscale = _scalar(xp.subtract(2.0, xp.divide(_scalar(0.0), dim["n+2"])), dtype)'),
                ("estimator.py", "if self.layer == 0: gm = pc / self.K.negative_n2; return gm, pc, _diag(gm), v", "if self.layer == 0: gm = pc * 0.0; return gm, pc, _diag(gm), v"),
                ("estimator.py", "return radius_mean(n).astype(self.dtype) * xp.stack(rows)", "return xp.stack(rows)")],
 "gauge_off": [("estimator.py", 'GAUGE_POLICY = "unit_prevariance"', 'GAUGE_POLICY = "identity"')],
 "tc_off": [("bank.py", "TCP_START = 4 ", "TCP_START = 99 "), ("bank.py", "TERM_TC, TERM_TC_TAU = True, 0.3", "TERM_TC, TERM_TC_TAU = False, 0.3"),
            ("estimator.py", "TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, None, None", "TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, 4, None")],
 "tch0_no_tc_history": [("bank.py", "TCH = 3  #", "TCH = 0  #")],
 "folds_off": [("bank.py", "TC_SLICE_FOLD = True", "TC_SLICE_FOLD = False"), ("bank.py", "HIST_FOLD = True", "HIST_FOLD = False")],
 "e6_off": [("bank.py", "E_TU, E_RADIAL = True, 'mehler1'", "E_TU, E_RADIAL = False, 'mehler1'"), ("bank.py", "E_PROP = True", "E_PROP = False")],
 "pk4_off": [("estimator.py", 'PK4_SINGLE, PK4_LEGS, PK4_CORESUB, PK4_TERM = True, True, "lite", True', 'PK4_SINGLE, PK4_LEGS, PK4_CORESUB, PK4_TERM = False, False, "lite", True')],
 "pk4_fold_response_off": [("pk4.py", "FOLD = True   #", "FOLD = False   #")],
 "elders_dense_no_switch": [("bank.py", "REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, 7 ", "REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, -1 ")],
 "repair_rank1": [("bank.py", "REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 2, 7 ", "REPAIR_RANK, AGE_SWITCH_MAX_BIRTH = 1, 7 ")],
 "slice_rows_dense": [("bank.py", 'SLICE_ROWS, SLICE_SAT, SLICE_DEAD, SLICE_ROWS_TC = True, 0.015, 0.0, "rows"', 'SLICE_ROWS, SLICE_SAT, SLICE_DEAD, SLICE_ROWS_TC = False, 0.015, 0.0, "rows"')],
 "edges_all_layers": [("estimator.py", "TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, None, None", "TERM_TC_DQ, EDGE_END, GM_OFFDIAG_LAYERS = True, 16, None")],
 "thinning_off": [("bank.py", "THIN_TAU, THIN_OUT, ROW_THIN, V_SAT, TERM_KEY = 0.04, 0.015, 0.01, 0.003, 0.01", "THIN_TAU, THIN_OUT, ROW_THIN, V_SAT, TERM_KEY = 0.0, 0.0, 0.0, 0.003, 0.01")],
 "terminal_z3sq_term_off": [("estimator.py", "(z3t * z3t / 72.0) * w16", "(z3t * z3t * 0.0) * w16")],
}
for name, edits in V.items():
    dst = ROOTS / name
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(SRC, dst, ignore=shutil.ignore_patterns("__pycache__", "graded"))
    for fn, old, new in edits:
        p = dst / fn; t = p.read_text(); c = t.count(old); assert c == 1, (name, fn, old, c); p.write_text(t.replace(old, new))
    print(name, len(edits), "edits")
(A / "variants.json").write_text(json.dumps(V, indent=1))
