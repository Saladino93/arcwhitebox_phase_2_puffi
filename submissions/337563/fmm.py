"""Metered (2,2,2;7) matrix products for the estimator and the history bank, evaluated with FUSED calls: one recursion level
acts along one 7-slot axis of the operand cube, and one logged call writes several slots at once through arithmetic-progression
slices of that axis (the residual cap counts calls, the bill counts elements). The per-level schedule is the DESIGN table
below (found and verified exactly by research/clean2e9_2026-09-29/fmm_design): stored quadrant signs, slot of every product
and the calls of each side; every call is copy-free under NumPy's overlap rule (an operand is the identical view as the output,
or disjoint from it with the same slot stride). One arena per (levels, dtype) holds the leaf cubes of both operands, the 7^L
leaf products and the combine stages; the shallower ("guest") depth one level down borrows merged leading slices of the host.
Operand transforms are cached by caller key (b_key / a_key certify identical operands). Every view x[...] is a client RPC on
the official runtime, so views are built once per (shape, batch) and reused. Symmetric products rely on the caller's symmetry
contract. The op sequence is part of the residual bill: never reorder."""
import flopscope.numpy as fnp

# DESIGN[side]: raw = slots of the four stored quadrants (an arithmetic progression), view = (swap, flip rows, flip cols) of
# the 2x2 quadrant grid as stored at raw positions 0..3, signs = stored sign per raw position, basis = in-place calls inside
# the raw block (cheap), fill = calls of the expanding pass. A call is (ufunc sign, x slots, y slots, out slots); a slot tuple
# with one repeated slot is a broadcast operand. DESIGN["c"]: calls (ufunc sign, source, x, y, out) with source "P" (both
# operands products), "QP" (x outputs, y products), "QQ" (both outputs, in place); trailing "QQ" calls are deferred to the
# final 4^L block. The four outputs carry the common sign +1. On the official client a VIEW costs the residual of a logged
# call, so views are created sparingly: "fill_split" lists the split points of a fill pass whose operands are all contiguous
# pieces of one partition of the slot axis (one base view + one split instead of three views per call), equal slot tuples
# share one view, and every index tuple is compiled once per depth (_compiled).
# SPLIT_FILL (residual repair, 2026-10-14): on the official client fnp.split is ONE logged round trip returning every piece
# as a write-through view (measured: 1 trip, 0 FLOPs), so a fill pass whose operands are contiguous pieces of one partition
# of the slot axis costs 2 round trips per level (base view + split) instead of 3 (A) / 5 (B) views. Same ufunc calls on the
# same memory: bitwise identical results. fill_split lists the partition points (A: [0:3 | 3 | 4:7], B: [0:2 | 2 | 3 | 4 | 5:7]).
DESIGN = {   # alternative basis, 9 calls per level: A 1 + 1, B 1 + 2, combine 4; 14 expanding units, copy-free
    "a": dict(raw=(0, 1, 2, 3), view=(0, 1, 0), signs=(1, 1, 1, 1), basis=((-1, (2, 1, 0), (3, 3, 3), (1, 2, 3)),),
              fill=((-1, (3, 3, 3), (0, 1, 2), (4, 5, 6)),), fill_split=(3, 4)),   # SPLIT_FILL: one view + one split per level
    "b": dict(raw=(3, 4, 5, 6), view=(0, 0, 0), signs=(1, 1, 1, -1), basis=((1, (3, 3, 3), (6, 5, 4), (3, 4, 5)),),
              fill=((-1, (3, 3), (5, 6), (0, 1)), (-1, (3,), (4,), (2,))), fill_split=(2, 3, 4, 5)),   # SPLIT_FILL (was M10: views)
    "c": dict(view=(0, 0, 0), calls=((-1, "P", (1, 3), (4, 1), (0, 1)), (-1, "P", (3, 6), (2, 0), (2, 3)),
                                     (-1, "QP", (1, 2), (5, 4), (1, 2)), (-1, "QP", (1, 2), (0, 6), (1, 2)))),
}
# exact_r2 (2026-10-13): the (11 calls, 12 units) front design of the same search (Karstadt-Schwartz optimum; same A and B
# sides and slot map): 6 expanding units per combine level instead of 8, plus a 3-unit output-basis change (the two trailing
# "QQ" calls, deferred to the final 4^L block: 3/4 per output element per level). Chosen per combine step (_c12): a step
# saves 2 units of (7/4)^(L-1-s) / 4 per output element and pays 3/4, so every step but the last one (smallest units) uses it.
DESIGN_C12 = dict(view=(0, 0, 0), calls=((-1, "P", (1, 2), (4, 5), (0, 1)), (-1, "P", (3, 6), (2, 0), (2, 3)),
                                         (-1, "QP", (2,), (4,), (2,)), (-1, "QP", (2,), (0,), (2,)),
                                         (1, "QQ", (1,), (2,), (1,)), (-1, "QQ", (1, 2), (0, 3), (1, 2))))
# C12_TRIP: resource-only price of one extra logged call (FLOPs). The 12-unit combine adds 2 calls per step; a step uses it
# when its bill saving exceeds 2 C12_TRIP (0: wherever the bill is lower). C12 False: the 9-call design everywhere.
# 1e6 (dev ID 30): -4.89 GF for +1.2k calls, residual unchanged in paired guards; 0: -5.36 GF for +2.6k calls, +0.02-0.04 s.
C12 = True
# CALL_PRICE (lane5 call-priced resource policy, 2026-10-13): declared resource price of ONE logged FlopScope operation, in
#   FLOPs (the grader's residual is ~10.7 us per logged op and the 0.40 s cap makes ops a hard rail). Used consistently by
#   every resource decision of this module: the plain-product depth (_depth), the deep level (_go_deep, _sf_deep) and the
#   12-unit combine (_c12, which earlier priced a call at C12_TRIP = 1e6 while _depth used the bill alone). Calls are counted
#   exactly (_pcalls). A resource constant selected by the op budget, not a fitted knob: it never changes what is computed,
#   only how (rounding-level differences between product schedules).
CALL_PRICE = 5000000.0   # M lane: call price for the op cap (was 3500000.0)
# ARENA_SAFE_DEPTHS (memory rail): the priced plain-product depth is chosen among {0 (plain), the caller's depth} only, so no
#   product runs at an intermediate depth that would allocate its own full-width arena (+0.1 to +0.8 GB measured on ID 30).
ARENA_SAFE_DEPTHS = True
C12_TRIP = CALL_PRICE   # kept as an alias (bank/estimator never read it)
_sign_cache, _compiled_cache = {}, {}
_HAS_SIGNS = {side: any(x != 1 for x in DESIGN[side]["signs"]) for side in "ab"}

def _sl(t):  # slot tuple -> basic slice along the slot axis (a broadcast operand keeps the axis with length one)
    if len(set(t)) == 1: return slice(t[0], t[0] + 1)
    step = t[1] - t[0]; stop = t[-1] + step
    return slice(t[0], stop if stop >= 0 else None, step)

def _key(t): return (t[0],) if len(set(t)) == 1 else tuple(t)   # broadcast tuples of one slot are the same view

def _compiled(side, L, alt=False):
    """Index programs of one side at depth L, built once: operand sides -> (basis, fill) with basis = [(ufunc sign, ix, iy,
    io)] (index tuples into the cube; ix / iy None when the operand is the output view) and fill = [(base index, axis, split
    points or None, [(ufunc sign, x, y, o)])] per level, x / y / o being index tuples (no split) or piece numbers (split);
    combine -> (live, late) slot programs per step, [(ufunc sign, source, kx, ky, ko)] with view keys."""
    if (c := _compiled_cache.get((side, L, alt))) is not None: return c
    D = DESIGN_C12 if side == "c" and alt else DESIGN[side]
    if side == "c":
        calls = [(uf, src, _key(x), _key(y), _key(o)) for uf, src, x, y, o in D["calls"]]; late = []
        while calls and calls[-1][1] == "QQ": late.insert(0, calls.pop())   # trailing output-basis calls: deferred to the final block
        c = (calls, late, {k: _sl(k) for call in calls + late for k in call[2:]})
    else:
        raw, basis, fill = _sl(D["raw"]), [], []
        for a in range(L, 0, -1):
            pre, post = (_S,) + (raw,) * (a - 1), (raw,) * (L - a)
            for uf, x, y, o in D["basis"]:
                basis.append((uf, None if x == o else pre + (_sl(x),) + post, None if y == o else pre + (_sl(y),) + post, pre + (_sl(o),) + post))
        pts = D["fill_split"]
        for a in range(L, 0, -1):  # innermost slot axis first: outer axes still raw, inner axes complete (long contiguous slabs last)
            pre = (_S,) + (raw,) * (a - 1)
            if pts is None: prog = [(uf, None if x == o else pre + (_sl(x),), None if y == o else pre + (_sl(y),), pre + (_sl(o),)) for uf, x, y, o in D["fill"]]
            else:
                lo = (0,) + tuple(pts); number = {(lo[i], (tuple(pts) + (7,))[i]): i for i in range(len(lo))}
                piece = lambda t: number[(t[0], t[0] + 1) if len(set(t)) == 1 else (t[0], t[-1] + 1)]
                prog = [(uf, piece(x), piece(y), piece(o)) for uf, x, y, o in D["fill"]]
            fill.append((pre, a, None if pts is None else list(pts), prog))
        c = (basis, fill)
    _compiled_cache[side, L, alt] = c; return c

def _flip(L, view):  # index that reverses the row / column bit axes of a (b, 2 x L, rt, 2 x L, ct) matrix reshape
    rev = slice(None, None, -1)
    return (_S,) + ((rev if view[1] else _S),) * L + (_S,) + ((rev if view[2] else _S),) * L + (_S,)

def _signs(L, side, dtype):  # stored-sign tensor in the matrix-reshape layout (sides with a stored sign flip only)
    if L == 0: return fnp.asarray(1.0, dtype=dtype)
    D = DESIGN[side]
    if (t := _sign_cache.get(key := (L, side, str(dtype)))) is None:
        swap, fr, fc = D["view"]; quad = [[1.0, 1.0], [1.0, 1.0]]
        for q in range(4):
            first, second = q >> 1, q & 1; r, c = (second, first) if swap else (first, second)
            quad[r ^ fr][c ^ fc] = float(D["signs"][q])
        q2, t = fnp.asarray(quad, dtype=dtype), None    # one 2x2 pattern per level, on its row bit and its column bit
        for j in range(L):
            shape = [1] * (2 * L + 3); shape[1 + j] = shape[L + 2 + j] = 2
            level = fnp.reshape(q2, tuple(shape)); t = level if t is None else fnp.multiply(t, level)
        _sign_cache[key] = t
    return t

# MAX_LEVELS: resource-only cap on the Strassen depth (package_d5: 5, was 6; depth 6 costs ~38 % more wall for ~5 % fewer
# FLOPs). The depth actually used is still derived from the runtime dims by levels_for() (divisibility, leaves >= LEAF_EXTENT).
# HOST_LEVELS stays 6 as in the measured lab run (tcf_calls_fp_d5): no depth-6 host arena exists, so every depth keeps its own.
MAX_LEVELS, LEAF_EXTENT, SYM_CHILDREN, HOST_LEVELS = 5, 16, 6, 6   # L3: depth-5 (graded-wall) variant
QUANTUM = 4 * LEAF_EXTENT   # resource block in leaf units, bounded by runtime width by callers
# Depth of a plain product (no shared or keyed operand): argmin over depths of  bill + TRIP_FLOPS * residual price, the price
# counted in logged calls (measured on the official client: a logged call costs about 4.6 microseconds of residual, a view
# about half of that). A level costs TRIPS_LEVEL (9 calls + 1 split + the views of its plan, which is rebuilt for most
# products because thinning gives them their own shapes), a product TRIPS_FIXED on top; a plain product is one call.
# fmm_depth (2026-10-12): the depths are priced with the exact metered bill (_bill plus the delivery; EXACT_DEPTH) and the
# residual price is nominal (TRIP_FLOPS 1: the bill decides, one FLOP per call only breaks ties; the residual stays far
# below its cap), over every depth from 1 (MIN_DEPTH). Measured on dev ID 30: -0.4 % FLOPs, and the own depth-3/4 arenas
# are no longer touched (complete peak 7.63 -> 6.74 GB together with the deep level).
TRIP_FLOPS, TRIPS_LEVEL, TRIPS_FIXED, MIN_DEPTH = 1.0, 16, 9, 1
EXACT_DEPTH = True   # False: the earlier 2mkn price model (leaf products billed 2mkn instead of m n (2k - 1))
_UNITS = {side: (sum(len(c[3]) for c in DESIGN[side]["basis"]), sum(len(c[3]) for c in DESIGN[side]["fill"])) for side in "ab"}
_UNITS["c"] = sum(len(c[4]) for c in DESIGN["c"]["calls"])
_depth_cache = {}

def _depth(B, m, k, n, levels, build=0):
    """Depth in {0 (plain), MIN_DEPTH .. levels} of the product (B, m, k) x (B, k, n) with the least bill + round-trip price
    (PLAIN_PRICE per exact call; build = the view round trips a fused plan spends on first use)."""
    if (d := _depth_cache.get(key := (B, m, k, n, levels, int(build)))) is None:
        a, b, c, best = B * m * k, B * k * n, B * m * n, None
        for L in ([0, levels] if ARENA_SAFE_DEPTHS else [0] + list(range(min(MIN_DEPTH, levels), levels + 1))):
            if EXACT_DEPTH: price = ((B * m * n * (2 * k - 1) if L == 0 else _bill(L, B, B, m, k, n) + 2.0 * B * m * n)
                                     + PLAIN_PRICE * (_pcalls(L, B * m * n) + (build if L else 0)))   # PLAIN_PRICE: bill + P x exact round trips
            elif L == 0: price = 2.0 * B * m * k * n + TRIP_FLOPS
            else:
                grow = sum(1.75 ** t for t in range(L))
                price = (2.0 * 7 ** L * (m >> L) * (k >> L) * (n >> L) * B + (_UNITS["a"][1] * a + _UNITS["b"][1] * b + _UNITS["c"] * c) / 4 * grow
                         + (_UNITS["a"][0] * a + _UNITS["b"][0] * b) / 4 * L + 2 * (a + b + c) + TRIP_FLOPS * (TRIPS_LEVEL * L + TRIPS_FIXED))
            if best is None or price < best[0]: best = (price, L)
        d = _depth_cache[key] = best[1]
    return d
# Deep level: one recursion level below the host depth (leaves >= DEEP_EXTENT), run inside the host arena's flat storage at
# the batch that fits it (no new arena memory), chosen per product when its exact metered bill is lower than at the host
# depth. _bill is the metered bill of one fused product: leaf products m n (2k - 1) per item, copy-in plus its billed reshape
# (2 per element), basis (3/4 per element and level) and fill (3/4 per element x grow) of each transformed side, and the
# combine stages (2 x grow per output element); the delivery is depth independent and left out.
DEEP, DEEP_EXTENT = True, 8

def _grow(L): return sum(1.75 ** t for t in range(L))

def _c12(L, elems):
    """Combine steps (step s writes slot axis 1 + s; a unit there is (7/4)^(L-1-s) / 4 per output element) that use the
    12-unit design for an output of `elems` elements (B m n): 2 expanding units fewer, 3/4 per output element for that
    axis's output-basis calls, and C12_TRIP per extra logged call. Every step but the last (L - 1, where the units are
    smallest) when C12_TRIP is 0. The design is chosen per axis: each axis carries its own output basis."""
    if not C12: return ()
    return tuple(s for s in range(L) if (0.5 * 1.75 ** (L - 1 - s) - 0.75) * elems > 2 * CALL_PRICE)

# PLAIN_PRICE (residual repair, 2026-10-14): every client round trip, VIEWS included, costs residual on the official runtime
# (measured ~8.6 us each on the grader; the op counters omit views). A fused Strassen product spends, besides its run calls,
# ~BUILD_OUT views per level for its combine program and ~BUILD_SIDE per level per operand side the first time its plan is
# used in a prediction (fmm.configure drops the plans). A fused product or family sum is therefore admitted only when its
# exact bill saving over the plain metered matmul exceeds PLAIN_PRICE x the round trips it adds (run calls + first-use view
# building); CALL_PRICE keeps pricing the choices among fused forms (deep level, 12-unit combine). Resource constants of the
# view programs and of the grader residual rail, selected to keep every network under the 0.4 s cap; no target input.
PLAIN_PRICE, BUILD_OUT, BUILD_SIDE = 2000000.0, 11, 9
# WARM_SCHEDULE retired (2026-10-14 rule: no cross-prediction state or warm schedules; KEEP_PLANS = False).
WARM_SCHEDULE = (2000000.0,)
_predictions = [0]
_requests = {}   # per prediction (configure): how often each product shape asked for a price; a plan's first-use view building
                 # is amortised over the requests seen so far (a shape that recurs layer after layer is built once)

def _build_trips(buf, m, k, n, L, B, a=True, b=True):
    """View round trips a product (m, k) x (k, n) at batch B would spend on first use of its plan in this arena, amortised
    over the requests of that shape so far in this prediction."""
    trips, plans = 0, buf.get("plans", {})
    rk = (int(m), int(k), int(n), int(B)); seen = _requests[rk] = _requests.get(rk, 0) + 1
    if not any(key[:4] == (int(m), int(k), int(n), int(B)) for key in plans):
        tree = buf.get("outputs", {}).get((int(m), int(n)))
        trips += BUILD_OUT * L if tree is None else (0 if int(B) in tree["batches"] else BUILD_OUT * L // 2)
    ops = buf.get("operands", {})
    if a and "ops" not in ops.get((int(m), int(k), "a", int(B)), {}): trips += BUILD_SIDE * L
    if b and "ops" not in ops.get((int(k), int(n), "b", int(B)), {}): trips += BUILD_SIDE * L
    return -(-trips // seen)

def _pcalls(L, elems, a=True, b=True, deliver=3):
    """Exact logged calls of one steady-state fused product at depth L (calls_probe.py): A side (copy-in + reshape + 2 per
    level), B side (2 + 3 per level), the leaf matmul, the combine (4 per level, + 2 per 12-unit step) and the delivery
    (3 detached: empty + reshape + copy; 2 into a given output); a plain product is 1 call. A / B resident under a key: 0."""
    if L == 0: return 1
    return (2 + 2 * L if a else 0) + (2 + 3 * L if b else 0) + 1 + 4 * L + 2 * len(_c12(L, elems)) + deliver

def _comb(L, elems):  # combine bill per output element (expanding units of every step + the output-basis calls)
    steps = _c12(L, elems)
    return sum(1.5 * 1.75 ** (L - 1 - s) + 0.75 if s in steps else 2.0 * 1.75 ** (L - 1 - s) for s in range(L))

def _bill(L, B, Bb, m, k, n, a=True, b=True, comb=True):
    g, mt, kt, nt = _grow(L), m >> L, k >> L, n >> L
    unit = 0.75 * (g + L) + 2.0
    return (B * 7 ** L * mt * nt * (2 * kt - 1) + unit * ((B * m * k if a else 0) + (Bb * k * n if b else 0))
            + (_comb(L, B * m * n) * B * m * n if comb else 0.0))

def _deep_cap(dtype, levels, dims):
    """Batch capacity of depth levels + 1 inside the host arena's storage for these dims, 0 when the deep level is off,
    the product is not at the host depth, or a dim is not divisible / would leave leaves below DEEP_EXTENT."""
    if not DEEP or int(levels) != HOST_LEVELS or (host := _buffers.get((HOST_LEVELS, str(dtype)))) is None: return 0
    L, kmax = int(levels) + 1, int(host["kmax"])
    if host["L"] != L - 1 or kmax % (1 << L) or any(x % (1 << L) or x >> L < DEEP_EXTENT or x > kmax for x in dims): return 0
    T = kmax >> L
    return min(int(host["qa"].shape[0]) // (7 ** L * T * T), int(host["c1"].shape[0]) // (4 * 7 ** (L - 1) * T * T))
SPLIT_BATCH, SPLIT_LEAF = 19, 4                                    # _sym_split: 19 quarter products, leaf >= 4
_split_ws, _buffers, _sym_ws, _resident_b, _resident_a, _layout = {}, {}, {}, {}, {}, {}
_MAXB, _chunk_skip_b, _S, _src_memo = 1, [False], slice(None), [None]

KEEP_PLANS = False


# Fixed setup-only view templates. Runtime planning dictionaries remain cold.
_STATIC_PREPARING = False
_STATIC_SIZES = (1024, 960, 896, 832, 768)

def _static_bank(buf):
    if _STATIC_PREPARING: return None
    bank = buf.get("static_templates")
    if bank is not None and all(buf[name] is value for name, value in bank["backing"]): return bank
    if bank is not None: buf.pop("static_templates", None)
    return None

def _static_operand(v):
    bank = _static_bank(v["buf"])
    if bank is None: return None
    L, rt, ct = v["tiles"]; side, B = v["side"], v["B"]
    template = bank["operands"].get((rt << L, ct << L, side, B))
    if template is None: return None
    layout = v["buf"]["exact"].get(("q" + side, (7,) * L, rt, ct))
    return template if layout is not None and layout["R"] == template["R"] else None

def _static_output(tree, b, cut):
    bank = _static_bank(tree["buf"])
    if bank is None: return None
    L, mt, nt = tree["tiles"]
    key = (mt << L, nt << L, int(b), "cut" if cut else "direct", tree["capacity"] if cut else int(b))
    template = bank["outputs"].get(key)
    if template is None: return None
    exact = tree["buf"]["exact"]
    for name, lead, R in template["rows"]:
        layout = exact.get((name, lead, mt, nt))
        if layout is None or layout["R"] != R: return None
    return dict(template["program"])

def _static_destination(dest, L, rt, ct, B, side):
    view = DESIGN[side]["view"]
    key = (B, L, rt, ct, True, bool(view[0]))
    shp = _layout.get(key)
    if shp is None: shp = _layout_of(key)
    inv = [0] * len(shp[1])
    for i, a in enumerate(shp[1]): inv[a] = i
    value = fnp.transpose(dest, tuple(inv))
    return value[_flip(L, view)] if view[1] or view[2] else value

def prepare_static_views(width: int) -> None:
    """Fixed public allocation/view catalogue; no weights, transforms or matrix products."""
    global HOST_LEVELS, _STATIC_PREPARING
    if width != 1024: return
    levels = sum(width % (1 << L) == 0 and width >> L >= LEAF_EXTENT for L in range(1, MAX_LEVELS + 1))
    HOST_LEVELS = levels
    set_max_batch(SYM_CHILDREN)
    reserve(levels, width, SYM_CHILDREN)
    reserve(levels - 1, width // 2, SPLIT_BATCH)
    dtype = fnp.dtype("float32")
    specs = ((levels, SYM_CHILDREN), (levels + 1, _deep_cap(dtype, levels, (width, width, width))))
    if any(cap <= 1 for _, cap in specs): raise ValueError("Static catalogue arena capacity changed")
    arenas = [(L, cap, _get_buffers(width, width, width, L, dtype, cap)) for L, cap in specs]
    if all((bank := _static_bank(buf)) is not None and bank["catalogue"] == (L, cap, _STATIC_SIZES) for L, cap, buf in arenas):
        return
    _STATIC_PREPARING = True
    try:
        for L, cap, buf in arenas:
            bank = {"catalogue": (L, cap, _STATIC_SIZES), "backing": tuple((name, buf[name]) for name in ("qa", "qb", "c0", "c1", "std")),
                    "operands": {}, "outputs": {}}
            for name in ("plans", "operands", "outputs"): buf.pop(name, None)
            for rows in _STATIC_SIZES:
                for cols in _STATIC_SIZES:
                    for side in ("a", "b"):
                        for B in (cap, 1):
                            v = _operand_views(rows, cols, L, side, buf, B)
                            _operand_transform(v)
                            rt, ct = rows >> L, cols >> L
                            bank["operands"][rows, cols, side, B] = {
                                "R": buf["exact"]["q" + side, (7,) * L, rt, ct]["R"], "ops": v["ops"], "dest4": v["dest4"],
                                "dest4T": _static_destination(v["dest4"], L, rt, ct, B, side)}
            for m in _STATIC_SIZES:
                for n in _STATIC_SIZES:
                    tree = _output_tree(m, n, L, buf, cap)
                    tree["batches"] = {}
                    direct1 = _out_batch(tree, 1)
                    direct_cap = _out_batch(tree, cap)
                    tree["batches"].pop(1)
                    cut1 = _out_batch(tree, 1)
                    for b, kind, origin, program in ((1, "direct", 1, direct1), (cap, "direct", cap, direct_cap), (1, "cut", cap, cut1)):
                        # Delivery views have the same addresses for direct and capacity-derived arithmetic programs.
                        tree["batches"][b] = program
                        plan = dict(batch=b, m22=program["m22"], _out=(tree, b))
                        pstd = {name: _pstd(plan, name) for name in ("std", "std_morton")}
                        mt, nt = m >> L, n >> L
                        row_layouts = tuple((name, lead, buf["exact"][name, lead, mt, nt]["R"])
                                            for name, lead in (("c0", (7,) * L), ("c1", (4,) + (7,) * (L - 1))))
                        prepared = {name: program[name] for name in ("combine", "pp", "m22", "cinv", "tiles")}
                        prepared["_static_pstd"] = pstd
                        prepared["_static_m22T"] = _m10_split_src(plan, L, mt, nt)
                        bank["outputs"][m, n, b, kind, origin] = {"rows": row_layouts, "program": prepared}
            buf["static_templates"] = bank
    finally:
        _STATIC_PREPARING = False
        # Retain the fixed templates and shape-capacity views, never a logical warm plan or data certificate.
        for buf in _buffers.values():
            for name in ("plans", "operands", "outputs"): buf.pop(name, None)
        for ws in _sym_ws.values(): ws.pop("bkey", None)
        _resident_a.clear(); _resident_b.clear(); _requests.clear(); _chunk_skip_b[0] = False
        set_max_batch(SYM_CHILDREN)


def configure():
    """Per-prediction reset: drop resident transforms (data) and, unless KEEP_PLANS, the shape-only plans/views."""
    global PLAIN_PRICE
    price = WARM_SCHEDULE[min(_predictions[0], len(WARM_SCHEDULE) - 1)] if KEEP_PLANS else WARM_SCHEDULE[0]; _predictions[0] += 1
    if price != PLAIN_PRICE: PLAIN_PRICE = price; _depth_cache.clear()
    if not KEEP_PLANS:
        for arena in _buffers.values():
            for name in ("plans", "operands", "outputs"): arena.pop(name, None)
    for ws in _sym_ws.values(): ws.pop("bkey", None)
    _resident_b.clear(); _resident_a.clear(); _requests.clear(); _chunk_skip_b[0] = False; set_max_batch(SYM_CHILDREN)

def set_max_batch(b):
    global _MAXB; _MAXB = max(1, int(b))

def levels_for(a, b):
    """Largest recursion depth dividing every dimension with leaves of at least LEAF_EXTENT."""
    if len(a.shape) != len(b.shape) or len(a.shape) not in (2, 3) or int(b.shape[-2]) != int(a.shape[-1]): return 0
    dims = (int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1]))
    return sum(all(x % (1 << L) == 0 and x >> L >= LEAF_EXTENT for x in dims) for L in range(1, MAX_LEVELS + 1))

def _matrix_view(x, L, rt, ct, side, *, quadrants=False):
    """Interleave matrix axes into forward-output or reversed-input (quadrants) recursion order, in the side's arrangement."""
    view = DESIGN[side]["view"]
    shp = _layout.get(key := (1 if len(x.shape) == 2 else int(x.shape[0]), L, rt, ct, quadrants, bool(view[0])))
    if shp is None: shp = _layout_of(key)
    x = fnp.reshape(x, shp[0])
    return fnp.transpose(x[_flip(L, view)] if view[1] or view[2] else x, shp[1])

def _split(x, L, rt, ct, dest, cache, name, side, *, quadrants=False):
    """(reshape(x) in plain split order, dest viewed in that same order): copyto(second, first) == copyto(dest, _matrix_view(x));
    the arrangement's flips sit on the cached destination view."""
    view = DESIGN[side]["view"]
    shp = _layout.get(key := (1 if len(x.shape) == 2 else int(x.shape[0]), L, rt, ct, quadrants, bool(view[0])))
    if shp is None: shp = _layout_of(key)
    if (d := cache.get(name)) is None:
        inv = [0] * len(shp[1])
        for i, a in enumerate(shp[1]): inv[a] = i
        d = fnp.transpose(dest, tuple(inv)); d = cache[name] = d[_flip(L, view)] if view[1] or view[2] else d
    if (memo := _src_memo[0]) is None: return fnp.reshape(x, shp[0]), d
    if (hit := memo.get(mkey := (id(x), shp[0]))) is None or hit[0] is not x: hit = memo[mkey] = (x, fnp.reshape(x, shp[0]))
    return hit[1], d   # a persistent workspace operand (memo set by its owner): its split view is built once

def _layout_of(key):
    b, L, rt, ct, quadrants, swap = key
    pairs = [(L + 2 + j, 1 + j) if swap else (1 + j, L + 2 + j) for j in (range(L - 1, -1, -1) if quadrants else range(L))]
    shp = _layout[key] = ((b,) + (2,) * L + (rt,) + (2,) * L + (ct,), tuple([0] + [x for pair in pairs for x in pair] + [1 + L, 2 * L + 2]))
    return shp

def _get_buffers(m, k, n, L, dtype, batch, own=False):
    """One arena per depth/dtype sized for (batch, max dim) as FLAT buffers (operand cubes qa/qb, products c0, first combine
    stage c1) plus a plain std scratch; each buffer is reshaped once per (lead layout, tile width ct) into a (cap, lead, T, ct)
    capacity view (_capacity) and every product shape cuts its exact tiles (contiguous rt x ct slabs per slot) from it by
    basic slicing (_exact): one reshape per tile width instead of one per tile shape.
    A guest depth (one level shallower, same tile) shares the host's flat buffers.
    own: the arena of the products that run shallower than their caller asked (full width at that depth, its own storage;
    allocation is lazy, only the touched part of an arena is ever resident)."""
    key, bmax, kmax = (int(L), str(dtype)) + (("own",) if own else ()), max(_MAXB, int(batch)), max(int(k), int(m), int(n))
    if (not own and DEEP and (host := _buffers.get((HOST_LEVELS, str(dtype)))) is not None and int(L) == host["L"] + 1
            and kmax <= host["kmax"] and int(batch) <= _deep_cap(dtype, host["L"], (host["kmax"],))):
        cap, kh = _deep_cap(dtype, host["L"], (host["kmax"],)), int(host["kmax"])
        # A previous host can become this host's deep view after a shape/depth
        # change while still sharing the same flat operands. Its old std batch
        # capacity/layout is then invalid even though qa identity matches.
        if ((entry := _buffers.get(key)) is None or entry["qa"] is not host["qa"]
                or entry["bmax"] != cap or entry["kmax"] != kh or entry["L"] != int(L)):
            if entry is not None: entry.pop("static_templates", None)
            _resident_a.pop(key, None); _resident_b.pop(key, None)                    # at the batch they hold (_go_deep / _sf_deep)
            entry = _buffers[key] = dict(bmax=cap, kmax=kh, T=kh >> int(L), L=int(L), key=key, qa=host["qa"], qb=host["qb"], c0=host["c0"],
                                         c1=host["c1"], std=fnp.empty((cap, kh, kh), dtype=dtype))
        return entry
    if own and (host := _buffers.get((HOST_LEVELS, str(dtype)))) is not None: kmax = max(kmax, host["kmax"])
    if (entry := _buffers.get(key)) is not None and entry["bmax"] >= batch and entry["kmax"] >= kmax: return entry
    if entry is not None:  # growth (never reached on the tested widths): fresh, larger arena
        entry.pop("static_templates", None)
        bmax, kmax = max(bmax, entry["bmax"]), max(kmax, entry["kmax"]); _resident_a.pop(key, None); _resident_b.pop(key, None); _buffers.pop(key)
    T = kmax >> L; ncube, nc1 = bmax * 7 ** L * T * T, bmax * 4 * 7 ** (L - 1) * T * T; out = None
    if not own and int(L) != HOST_LEVELS and (host := _buffers.get((HOST_LEVELS, str(dtype)))) is not None:
        if host["T"] == T and host["L"] == L + 1 and -(-bmax // 7) <= host["bmax"]:
            out = [host["qa"], host["qb"], host["c0"], host["c1"]]
    if out is None: out = [fnp.empty((ncube,), dtype=dtype), fnp.empty((ncube,), dtype=dtype), fnp.empty((ncube,), dtype=dtype), fnp.empty((nc1,), dtype=dtype)]
    std = fnp.empty((bmax, kmax, kmax), dtype=dtype)
    return _buffers.setdefault(key, dict(bmax=bmax, kmax=kmax, T=T, L=int(L), key=key, qa=out[0], qb=out[1], c0=out[2], c1=out[3], std=std))

def _tile(B, axes, rt):  # basic-slice index of the (B, <axes>, rt, ct) exact tile inside a (cap, <axes>, R, ct) capacity view
    return (slice(0, int(B)),) + (_S,) * axes + (slice(0, int(rt)),)

ROW_STEPS = 4   # resource-only: a new capacity class holds tiles of up to rt rows rounded up to a 1/ROW_STEPS step of the arena tile T

def _class_rows(buf, name, lead, rt, ct, B):
    """Row extent R of the capacity class serving an exact (rt, ct) tile of flat buffer `name` (rt <= R <= T): an existing
    class that already holds batch B (smallest R), else the smallest existing class with R >= rt (it is regrown), else a new
    class with rt rounded up to a T / ROW_STEPS step. The reshape bill of a class is cap x lead x R x ct, so tiles much
    shorter than the arena tile (own arenas: T = full width >> L) no longer pay for T rows."""
    rows = buf.setdefault("rows", {}).setdefault((name, lead, int(ct)), []); caps = buf.setdefault("cap", {})
    fit = [R for R in rows if R >= rt]
    held = [R for R in fit if int(caps[(name, lead, int(ct), R, 0)].shape[0]) >= B]
    if held or fit: return min(held or fit)
    T = int(buf["T"]); q = max(1, T // ROW_STEPS); R = min(T, -(-int(rt) // q) * q); rows.append(R); return R

def _capacity(buf, name, lead, ct, B, corners=0, R=None):
    """The ONE reshape of flat buffer `name` per (lead layout, tile width ct, row extent R <= T): a (cap,) + lead + (R, ct)
    view (every exact tile of width ct and at most R rows is its leading rows, so each slot's tile stays one contiguous
    rt x ct slab), built at the first batch asked and regrown once to the arena batch (leading batches keep their
    addresses: the layout does not depend on cap). corners=L: its (cap, (2,2) x L, R, ct) quadrant view of the side's raw
    block (one more reshape; splitting a 4-slot axis into (2, 2) is a view on the strided capacity)."""
    R = int(buf["T"]) if R is None else int(R)
    caps = buf.setdefault("cap", {}); key = (name, lead, int(ct), R, int(corners))
    if (c := caps.get(key)) is None or int(c.shape[0]) < B:
        unit = R * int(ct)
        for x in lead: unit *= int(x)
        cap = int(B) if c is None else max(int(B), min(int(buf[name].shape[0]) // unit, int(buf["bmax"])))
        if corners:
            base = _capacity(buf, name, lead, ct, cap, R=R)
            c = fnp.reshape(base[:cap][(_S,) + (_sl(DESIGN[name[1]]["raw"]),) * corners], (cap,) + (2, 2) * corners + (R, int(ct)))
        else: c = fnp.reshape(buf[name][:cap * unit], (cap,) + lead + (R, int(ct)))
        caps[key] = c
    return c

def _exact(buf, name, lead, rt, ct, B, corners=0):
    """(B,) + lead + (rt, ct) tile view of flat buffer `name`: a basic-slice cut [:B, ..., :rt, :] of a (lead, ct, R)
    capacity view (no per-shape reshape); cached per shape at the largest batch seen (regrown in the same class, so
    smaller batches are leading slices at unchanged addresses). corners=L: the (B, (2,2)*L, rt, ct) quadrant view of its
    raw block (the copy-in destination), cut from the same class."""
    ex = buf.setdefault("exact", {}); key, lead = (name, tuple(lead), int(rt), int(ct)), tuple(int(x) for x in lead)
    if (e := ex.get(key)) is None or e["cap"] < B:
        R = _class_rows(buf, name, lead, rt, ct, B) if e is None else e["R"]
        e = ex[key] = {"cap": int(B), "R": R, "v": _capacity(buf, name, lead, ct, B, R=R)[_tile(B, len(lead), rt)]}
    if corners:
        if (d := e.get("d4")) is None:
            d = e["d4"] = _capacity(buf, name, lead, ct, e["cap"], corners, R=e["R"])[_tile(e["cap"], 2 * corners, rt)]
        return d if e["cap"] == B else d[:B]
    return e["v"] if e["cap"] == B else e["v"][:B]

def _stage(step, pp, c1):
    """Combine stage `step` (-1 = the products) as (arena, index prefix): the stage is arena[prefix], of shape (B, 4 x (step
    + 1), 7 x (L - 1 - step), tile), c1 / c0 ping-pong; its views are taken straight from the arena (prefix + slot slice)."""
    if step < 0: return pp, (_S,)
    return (c1, (_S, _S) + (slice(0, 4),) * step) if step % 2 == 0 else (pp, (_S,) + (slice(0, 4),) * (step + 1))

def _final(L):  # (flat buffer, lead) holding the final combine stage at depth L (c1 / c0 ping-pong)
    return ("c1", (4,) + (7,) * (L - 1)) if (L - 1) % 2 == 0 else ("c0", (7,) * L)

def _m22(buf, L, nt, B, R):  # final combine stage of the (nt, R) class at capacity as (cap, (2,2) x L, R, nt): one reshape per class
    caps = buf.setdefault("cap", {})
    if (c := caps.get(("m22", int(nt), int(R)))) is None or int(c.shape[0]) < B:
        name, lead = _final(L); arena = _capacity(buf, name, lead, nt, B, R=R)
        array, prefix = _stage(L - 1, arena, arena)
        last = array[prefix]; c = caps["m22", int(nt), int(R)] = fnp.reshape(last, (int(last.shape[0]),) + (2, 2) * L + (int(R), int(nt)))
    return c

def _out_views(buf, L, mt, nt, B):  # products pp (c0), first combine arena c1, final stage and its quadrant view
    pp, c1 = _exact(buf, "c0", (7,) * L, mt, nt, B), _exact(buf, "c1", (4,) + (7,) * (L - 1), mt, nt, B)
    key, ex = ("m22", int(L), int(mt), int(nt), int(B)), buf.setdefault("exact", {})
    if (e := ex.get(key)) is None or e[0] is not pp or e[1] is not c1:   # rebuilt only when an exact view was regrown
        array, prefix = _stage(L - 1, pp, c1); last = array[prefix]; name, lead = _final(L)
        e = ex[key] = (pp, c1, _m22(buf, L, nt, B, ex[(name, lead, int(mt), int(nt))]["R"])[_tile(B, 2 * L, mt)], last)
    return pp, c1, e[2], e[3]

def reserve(L, kmax, batch, dtype="float32"):
    if L > 0: _get_buffers(kmax, kmax, kmax, L, dtype, batch)

def _operand_views(rows, cols, L, side, buf, B):  # operand views at exactly batch B: leaf tile now, transform views on first use
    cache = buf.setdefault("operands", {})
    if (v := cache.get(key := (int(rows), int(cols), side, int(B)))) is None:
        rt, ct = rows >> L, cols >> L
        v = cache[key] = {"tiles": (L, rt, ct), "leaf": _exact(buf, "q" + side, (7,) * L, rt, ct, B), "side": side,
                          "buf": buf, "B": int(B)}
    return v

def _operand_transform(v):  # per-level call views of the side: basis calls inside the raw block, then the expanding fills
    if "ops" in v: return v
    if (template := _static_operand(v)) is not None:
        v.update({name: template[name] for name in ("ops", "dest4", "dest4T")}); return v
    (L, rt, ct), cube = v["tiles"], v["leaf"]; (basis, fill), ops, add, subtract = _compiled(v["side"], L), [], fnp.add, fnp.subtract
    for uf, ix, iy, io in basis:
        o = cube[io]; ops.append((add if uf == 1 else subtract, o if ix is None else cube[ix], o if iy is None else cube[iy], o))
    for pre, axis, points, prog in fill:
        if points is None:
            vw = {}   # M10: equal index tuples of one level share one view
            def cv(ix, vw=vw):
                k = tuple((s.start, s.stop, s.step) if isinstance(s, slice) else s for s in ix)
                if k not in vw: vw[k] = cube[ix]
                return vw[k]
            for uf, ix, iy, io in prog:
                o = cv(io); ops.append((add if uf == 1 else subtract, o if ix is None else cv(ix), o if iy is None else cv(iy), o))
        else:
            pieces = fnp.split(cube[pre], points, axis=axis)
            for uf, x, y, o in prog: ops.append((add if uf == 1 else subtract, pieces[x], pieces[y], pieces[o]))
    v["dest4"], v["ops"] = _exact(v["buf"], "q" + v["side"], (7,) * L, rt, ct, v["B"], corners=L), ops; return v

def _output_tree(m, n, L, buf, capacity):  # output views at capacity (combine stages, moved output slots, std delivery)
    cache = buf.setdefault("outputs", {})
    if (tree := cache.get(key := (int(m), int(n)))) is not None and tree["capacity"] >= capacity: return tree
    B, (mt, nt) = int(capacity), (m >> L, n >> L); _out_views(buf, L, mt, nt, B)  # exact views at capacity: batches are leading slices
    std = buf["std"][:B, :m, :n]  # the arena's plain scratch; trees take free sub-views
    tree = cache[key] = {"capacity": B, "tiles": (L, mt, nt), "std": std, "buf": buf, "batches": {}}   # std_morton: on first use (_pstd)
    return tree

def _cut(x, batch, memo=None):  # cut a nested tuple/list of capacity views down to `batch` (ufuncs and scalars pass through)
    if memo is None: memo = {}   # M10: one cut per distinct view object (a view shared by several calls is cut once)
    if isinstance(x, (tuple, list)): return type(x)(_cut(v, batch, memo) for v in x)
    if x is None or isinstance(x, int) or callable(x): return x
    if (c := memo.get(id(x))) is None: c = memo[id(x)] = (x, x[:batch])
    return c[1]

# Final normal combine, transposed to seven input rows by four output columns.
# This is an ordinary dense linear map: every scalar coefficient is publicly billed.
# Its leading singleton batch survives the existing _cut(..., batch) unchanged.
_FINAL_COMBINE_COEFF = ((0., -1., 0., -1.), (1., -1., 0., 0.), (0., 0., -1., 0.),
                        (0., 1., 1., 0.), (-1., 0., -1., 0.), (0., -1., 0., 0.), (0., 0., -1., 1.))
_final_combine_coefficients = {}

def _final_combine_coefficient(dtype):
    key = str(dtype)
    if (coefficient := _final_combine_coefficients.get(key)) is None:
        coefficient = _final_combine_coefficients[key] = fnp.asarray([_FINAL_COMBINE_COEFF], dtype=dtype)
    return coefficient

def _out_batch(tree, b):  # combine/leaf/delivery views at batch b: cut from the capacity set if it exists, else built
    if (d := tree["batches"].get(b)) is not None: return d
    (L, mt, nt), C, buf = tree["tiles"], tree["capacity"], tree["buf"]
    if (prepared := _static_output(tree, b, b != C and C in tree["batches"])) is not None:
        tree["batches"][b] = prepared; return prepared
    if b != C and C in tree["batches"]:
        memo = {}; d = tree["batches"][b] = {key: _cut(tree["batches"][C][key], b, memo) for key in ("combine", "pp", "m22", "cinv")}
        d["tiles"] = (L, mt, nt); return d
    pp, c1, m22, final = _out_views(buf, L, mt, nt, b)
    alt, add, subtract = _c12(L, b * (mt << L) * (nt << L)), fnp.add, fnp.subtract   # steps on the 12-unit design
    comb, cinv = [], []
    for step in range(L):  # the outputs of step s replace slot axis 1 + s in place of the products (same axis order throughout)
        if step == L - 1:
            # The last step is never C12. Read the preceding seven-slot stage
            # and write the existing final four-slot stage in the other arena.
            # Only axis views are created; matmul writes the strided destination.
            parr, ppre = _stage(L - 2, pp, c1)
            axes = tuple(range(L)) + (L + 1, L + 2, L)
            source = fnp.transpose(parr[ppre], axes)
            destination = fnp.transpose(final, axes)
            comb.append((fnp.matmul, source, _final_combine_coefficient(pp.dtype), destination))
            continue
        calls, late, slices = _compiled("c", L, step in alt)
        (parr, ppre), (qarr, qpre), pv, qv = _stage(step - 1, pp, c1), _stage(step, pp, c1), {}, {}
        ppre, qpre = ppre + (_S,) * (1 + step - len(ppre)), qpre[:1 + step]   # slot axis 1 + step: a full 7-axis of p, the last 4-axis of q
        for uf, src, x, y, o in calls:   # equal slot tuples of one array share one view
            xs, ys, os = (pv if src in ("P", "PP") else qv), (qv if src == "QQ" else pv), (pv if src == "PP" else qv)
            for views, k in ((xs, x), (ys, y), (os, o)):
                if k not in views: views[k] = parr[ppre + (slices[k],)] if views is pv else qarr[qpre + (slices[k],)]
            comb.append((add if uf == 1 else subtract, xs[x], ys[y], os[o]))
    for a in alt:   # the output-basis calls of the 12-unit axes, on the final 4^L block
        (_, late, slices), pre, qv = _compiled("c", L, True), (_S,) * (1 + a), {}
        for uf, src, x, y, o in late:
            for k in (x, y, o):
                if k not in qv: qv[k] = final[pre + (slices[k],)]
            cinv.append((add if uf == 1 else subtract, qv[x], qv[y], qv[o]))
    d = tree["batches"][b] = dict(combine=comb, pp=pp, m22=m22, cinv=cinv, tiles=(L, mt, nt))
    return d

def _pstd(plan, name):  # the plan's 'std' / 'std_morton' delivery view at its batch, cut once from the tree
    if (v := plan.get(name)) is None:
        tree, b = plan["_out"]; d = tree["batches"][b]
        if name in d.get("_static_pstd", {}):
            v = d[name] = plan[name] = d["_static_pstd"][name]; return v
        if name not in tree:  # the tree's std_morton reshape is billed only for trees that deliver into std (detach=False)
            L, mt, nt = tree["tiles"]; tree[name] = _matrix_view(tree["std"], L, mt, nt, "c", quadrants=True)
        if (v := d.get(name)) is None: v = d[name] = tree[name] if b == tree["capacity"] else tree[name][:b]
        plan[name] = v
    return v

def _get_plan(m, k, n, L, dtype, batch, b_shared=False, b_group=0, own=False):  # operand + output views for one shape; owned by its arena
    buf = _get_buffers(m, k, n, L, dtype, batch, own); cache = buf.setdefault("plans", {})
    if (plan := cache.get(key := (int(m), int(k), int(n), int(batch), bool(b_shared), int(b_group)))) is None:
        have = buf.get("outputs", {}).get((int(m), int(n))); grown = have is not None and have["capacity"] < batch
        cap = int(buf["bmax"]) if not own and (batch == 1 and 4 * k >= 3 * max(m, n)) or grown else int(batch)
        bb = 1 if b_shared else batch // b_group if b_group else batch
        v = {"a": _operand_views(m, k, L, "a", buf, batch), "b": _operand_views(k, n, L, "b", buf, bb)}
        tree = _output_tree(m, n, L, buf, cap)
        plan = cache[key] = {"batch": int(batch), "v": v, "_out": (tree, int(batch)), "rkey": buf["key"], **_out_batch(tree, int(batch))}
        if "_static_m22T" in plan: plan["m22T"] = plan["_static_m22T"]
        if b_group:  # grouped rows share one B per group: row t of every group as one strided batch (no reshape)
            plan["gleaf"] = [(v["a"]["leaf"][t::b_group], v["b"]["leaf"], plan["pp"][t::b_group]) for t in range(b_group)]
    return plan

def _combine(plan):  # seven products -> four quadrants, level by level, then the deferred output-basis calls
    held = []  # Ordinary public out-return aliases, retained only through this helper.
    try:
        for f, x, y, out in plan["combine"]: held.append(f(x, y, out=out))
        for f, x, y, out in plan["cinv"]: held.append(f(x, y, out=out))
    finally:
        held.clear()  # Release on normal and exceptional exits, inside prediction.

def _run_leaf(plan, a3, b3, b_batch=None, skip_b=False, b_key=None, a_key=None, out=None):
    """Transform both operands (unless resident under the same key), fill the seven factors, multiply the leaves (into `out`
    instead of the plan's product slots when given)."""
    held = []  # These returned aliases add no numerical buffers; plan storage remains the owner.
    try:
        copyto = fnp.copyto; sa, sb = plan["v"]["a"], plan["v"]["b"]
        (L, mt, kt), rkey = sa["tiles"], plan["rkey"]
        if len(rkey) == 2:  # host and guest depths share storage: crossing depth invalidates resident transforms
            for res in (_resident_a, _resident_b):
                if res and (len(res) > 1 or rkey not in res):
                    for other in [key for key in res if key != rkey and len(key) == 2]: del res[other]
        if a_key is None or _resident_a.get(rkey) != (a_key, mt, kt, batch_a := plan["batch"]):
            _operand_transform(sa); src, dst = _split(a3, L, mt, kt, sa["dest4"], sa, "dest4T", "a", quadrants=True)
            if _HAS_SIGNS["a"]: held.append(fnp.multiply(src, _signs(L, "a", src.dtype), out=dst))
            else: copyto(dst, src)
            for f, x, y, o in sa["ops"]: held.append(f(x, y, out=o))
            _resident_a[rkey] = (a_key, mt, kt, batch_a) if a_key is not None else None
        (L, kt, nt), batch_b = sb["tiles"], plan["batch"] if b_batch is None else b_batch
        if not skip_b and b_key is not None and _resident_b.get(rkey) == (b_key, kt, nt, batch_b): skip_b = True
        if not skip_b:
            _operand_transform(sb); src, dst = _split(b3, L, kt, nt, sb["dest4"], sb, "dest4T", "b", quadrants=True)
            if _HAS_SIGNS["b"]: held.append(fnp.multiply(src, _signs(L, "b", src.dtype), out=dst))
            else: copyto(dst, src)
            for f, x, y, o in sb["ops"]: held.append(f(x, y, out=o))
            _resident_b[rkey] = (b_key, kt, nt, batch_b) if b_key is not None else None
        if out is not None: held.append(fnp.matmul(sa["leaf"], sb["leaf"], out=out))
        else:
            for la, lb, lp in plan.get("gleaf", ((sa["leaf"], sb["leaf"], plan["pp"]),)): held.append(fnp.matmul(la, lb, out=lp))
    finally:
        held.clear()  # Release on normal and exceptional exits, inside prediction.

def _fast_ok(a, b, div, shared=False):
    same = len(a.shape) == len(b.shape) and (len(a.shape) == 2 or int(a.shape[0]) == int(b.shape[0]))
    return ((same or (shared and len(a.shape) == 3 and len(b.shape) == 2)) and int(b.shape[-2]) == int(a.shape[-1]) and a.dtype == b.dtype
            and not (int(a.shape[-2]) % div or int(a.shape[-1]) % div or int(b.shape[-1]) % div))

def _scatter(core, s, t, segments):  # copy batch rows [s, t) of core into the (a0, a1, target) segments they overlap
    for a0, a1, tgt in segments:
        lo, hi = max(s, a0), min(t, a1)
        if hi > lo: fnp.copyto(tgt if lo == a0 and hi - a0 == int(tgt.shape[0]) else tgt[lo - a0:hi - a0],
                               core if lo == s and hi - s == int(core.shape[0]) else core[lo - s:hi - s])

def _deliver(plan, detach, out, two_d=False):  # un-interleave the result into `out`, a fresh array (detach) or the std view
    L, mt, nt = plan["tiles"]
    if out is None and not detach:
        fnp.copyto(_pstd(plan, "std_morton"), plan["m22"]); return plan_view(plan, "std0") if two_d else _pstd(plan, "std")
    if out is None and int(L) >= 1:   # M10 (M5 class 4): reshape-copy of the split-order result view
        return fnp.reshape(_m10_split_src(plan, L, mt, nt), (mt << L, nt << L) if two_d else (plan["batch"], mt << L, nt << L))
    if out is None: out = fnp.empty((mt << L, nt << L) if two_d else (plan["batch"], mt << L, nt << L), dtype=plan["m22"].dtype)
    dst, src = _split(out, L, mt, nt, plan["m22"], plan, "m22T", "c", quadrants=True); fnp.copyto(dst, src); return out

def _m10_split_src(plan, L, mt, nt):
    """The plan's m22 result viewed in plain split order: exactly the cached copy source of _split(..., "m22T", "c")."""
    if (d := plan.get("m22T")) is None:
        view = DESIGN["c"]["view"]
        shp = _layout.get(key := (int(plan["batch"]), L, mt, nt, True, bool(view[0])))
        if shp is None: shp = _layout_of(key)
        inv = [0] * len(shp[1])
        for i, a in enumerate(shp[1]): inv[a] = i
        d = fnp.transpose(plan["m22"], tuple(inv)); d = plan["m22T"] = d[_flip(L, view)] if view[1] or view[2] else d
    return d

def plan_view(plan, name):  # derived per-plan views: std0 = std[0], pp0 = pp[0], pp:e = pp[e], acc1 = accumulator[None]
    cache = plan.setdefault("derived", {})
    if (v := cache.get(name)) is None:
        v = cache[name] = (_pstd(plan, "std")[0] if name == "std0" else fnp.expand_dims(plan["leaf_accumulator"], 0) if name == "acc1"
                           else plan["pp"][0 if name == "pp0" else int(name[3:])])
    return v

def _go_deep(a, b, levels, shared2, b_shared, b_key, a_key):
    """Batch step of the deep level for this product when its bill there is lower than at the host depth, else 0. Keyed
    operands are priced as resident at either depth; a shared B is transformed once per product."""
    m, k, n = int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
    if not (cap := _deep_cap(a.dtype, levels, (m, k, n))) or not _fast_ok(a, b, 1 << (int(levels) + 1), shared=shared2): return 0
    B = 1 if len(a.shape) == 2 else int(a.shape[0]); Bb = 1 if b_shared else B; L = int(levels)

    def calls(Lx, step):   # CALL_PRICE: exact calls of the product run in chunks of `step` items at depth Lx
        chunks, per, dl = (-(-B // step) if B > step else 1), min(B, step), 3 if B == 1 else 2
        one = _pcalls(Lx, per * m * n, a=a_key is None, b=b_key is None, deliver=dl)
        return chunks * one - ((chunks - 1) * (2 + 3 * Lx) if b_shared and b_key is None else 0)
    deep = _bill(L + 1, B, Bb, m, k, n, a=a_key is None, b=b_key is None) + CALL_PRICE * calls(L + 1, cap)
    return cap if deep < _bill(L, B, Bb, m, k, n, a=a_key is None, b=b_key is None) + CALL_PRICE * calls(L, _MAXB) else 0

def matmul(a, b, levels, *, detach=True, out=None, out_segments=None, b_shared=False, b_key=None, a_key=None, _chunk=None):
    """Metered product: 2-D, batched 3-D, shared-B (2-D b against a 3-D a) or scattered into out_segments. Keys certify
    identical operands for transform reuse; detach=False returns an arena view (valid until the next call)."""
    shared2, own, step = bool(b_shared) and len(a.shape) == 3 and len(b.shape) == 2, False, _MAXB
    if levels > 0 and TRIP_FLOPS and _chunk != "deep" and not b_shared and b_key is None and a_key is None and out_segments is None and _fast_ok(a, b, 1 << int(levels)):
        Bd, md, kd, nd = 1 if len(a.shape) == 2 else min(int(a.shape[0]), _MAXB), int(a.shape[-2]), int(a.shape[-1]), int(b.shape[-1])
        deep = _depth(Bd, md, kd, nd, int(levels), _build_trips(_get_buffers(md, kd, nd, int(levels), a.dtype, Bd), md, kd, nd, int(levels), Bd))
        own, levels = deep != levels, deep
    if levels > 0 and _chunk is None and not own and (cap := _go_deep(a, b, levels, shared2, b_shared, b_key, a_key)):
        levels, step, _chunk = int(levels) + 1, cap, "deep"   # chunks of the deep level's batch, no further depth choice
    if levels <= 0 or not _fast_ok(a, b, 1 << int(levels), shared=shared2):  # plain product (thin repair rows)
        if shared2: b = fnp.broadcast_to(b, (int(a.shape[0]),) + tuple(int(x) for x in b.shape))
        if out_segments is not None: return _scatter(fnp.matmul(a, b), 0, int(a.shape[0]), out_segments)
        return fnp.matmul(a, b, out=out) if out is not None else fnp.matmul(a, b)
    if len(a.shape) == 2:
        plan = _get_plan(int(a.shape[0]), int(a.shape[1]), int(b.shape[1]), levels, a.dtype, 1, own=own)
        _run_leaf(plan, a, b, b_key=b_key, a_key=a_key); _combine(plan); return _deliver(plan, detach, out, two_d=True)
    B, m, k, n = int(a.shape[0]), int(a.shape[1]), int(a.shape[2]), int(b.shape[-1])
    if B > step:  # chunk the batch; a shared B is transformed once (skip_b on the later chunks)
        out = fnp.empty((B, m, n), dtype=a.dtype) if out is None and out_segments is None else out
        inner = _chunk or "host"
        for s in range(0, B, step):
            t, _chunk_skip_b[0] = min(B, s + step), bool(b_shared and s > 0)
            try:
                bc, key = (b if shared2 else b[s:t]), (b_key if b_shared else None)
                if out_segments is not None: _scatter(matmul(a[s:t], bc, levels, detach=False, b_shared=b_shared, b_key=key, _chunk=inner), s, t, out_segments)
                else: matmul(a[s:t], bc, levels, out=out[s:t], b_shared=b_shared, b_key=key, _chunk=inner)
            finally: _chunk_skip_b[0] = False
        return out
    plan = _get_plan(m, k, n, levels, a.dtype, B, b_shared=bool(b_shared), own=own)
    _run_leaf(plan, a, b, b_batch=1 if b_shared else None, skip_b=_chunk_skip_b[0], b_key=b_key, a_key=a_key); _combine(plan)
    if out_segments is not None: return _scatter(_deliver(plan, False, None), 0, B, out_segments)
    return _deliver(plan, detach, out)

def matmul_grouped(a, b, levels, outs):  # outs[t][i] = a[i*g + t] @ b[i]: g row groups per shared right factor
    g, G, (m, k, n) = len(outs), int(b.shape[0]), (int(a.shape[1]), int(a.shape[2]), int(b.shape[2]))
    if levels > 0 and not any(d % (1 << int(levels)) for d in (m, k, n)):   # PLAIN_PRICE: fused chunks against the plain grouped product
        L, cs = int(levels), max(1, _MAXB // g); buf = _get_buffers(m, k, n, L, a.dtype, cs * g); bill = calls = 0.0
        for s in range(0, G, cs):
            Bc = (min(G, s + cs) - s) * g; bill += _bill(L, Bc, Bc // g, m, k, n); calls += 5 + 5 * L + 1 + 4 * L + 2 * len(_c12(L, Bc * m * n)) + 3 + g + _build_trips(buf, m, k, n, L, Bc)
        if bill + PLAIN_PRICE * calls > 2.0 * m * k * n * G * g + PLAIN_PRICE * (3 + g): levels = 0
    if levels <= 0 or any(d % (1 << int(levels)) for d in (m, k, n)):
        products = fnp.matmul(fnp.reshape(a, (G, g, m, k)), fnp.expand_dims(b, 1))
        for t in range(g): fnp.copyto(outs[t], products[:, t])
        return
    for s in range(0, G, cs := max(1, _MAXB // g)):
        e = min(G, s + cs); plan, whole = _get_plan(m, k, n, levels, a.dtype, (e - s) * g, b_group=g), s == 0 and e == G
        _run_leaf(plan, a if whole else a[s * g:e * g], b if whole else b[s:e], b_batch=e - s)
        _combine(plan); _deliver(plan, False, None)
        if (rows := plan.setdefault("derived", {}).get("grp")) is None:
            std = _pstd(plan, "std"); rows = plan["derived"]["grp"] = [std[t::g] for t in range(g)]
        for t in range(g): fnp.copyto(outs[t] if whole and int(outs[t].shape[0]) == G else outs[t][s:e], rows[t])

def _sf_deep(fams, m, n, levels):
    """Batch step of the deep level for a leaf-space family sum when its bill there (items' leaf products and transforms,
    the leaf-space accumulation of every item after the first, one combine) is lower than at the host depth, else 0."""
    ks = tuple(int(a.shape[2]) for a, _, _ in fams)
    if not (cap := _deep_cap(fams[0][0].dtype, levels, (m, n) + ks)) or any(not _fast_ok(a, b, 1 << (int(levels) + 1)) for a, b, _ in fams): return 0
    items, first_seq = sum(int(a.shape[0]) for a, _, _ in fams), fams[0][2]
    def bill(L):
        X = 7 ** L * (m >> L) * (n >> L)
        return (sum(_bill(L, int(a.shape[0]), int(a.shape[0]), m, int(a.shape[2]), n, comb=False) for a, _, _ in fams)
                + (items - 1 + bool(first_seq)) * X + _comb(L, m * n) * m * n)

    def calls(L, step):   # CALL_PRICE: per chunk A/B staging + transforms + leaf (5 + 5L) and its accumulation; one combine
        sizes = [min(step, int(a.shape[0]) - s) for a, _, _ in fams for s in range(0, int(a.shape[0]), step)]
        return sum(5 + 5 * L + 1 + (c > 1) for c in sizes) + 4 * L + 2 * len(_c12(L, m * n)) + 3
    L = int(levels)
    return cap if bill(L + 1) + CALL_PRICE * calls(L + 1, cap) < bill(L) + CALL_PRICE * calls(L, _MAXB) else 0

def _sf_fused_cheaper(fams, m, n, L, step, buf):
    """PLAIN_PRICE: bill + price x exact round trips of the leaf-space fused family sum (per chunk: transforms, leaf, accumulation;
    once: combine, delivery; first-use view building) against the plain metered sum (batched matmul, batch sum, adds)."""
    X, items, bill, calls, pbill, pcalls, terms, first = 7 ** L * (m >> L) * (n >> L), 0, 0.0, 0, 0.0, 0, 0, None
    for a, b, seq in fams:
        B, k = int(a.shape[0]), int(a.shape[2])
        if first is None: first = seq
        for s in range(0, B, step):
            Bc = min(step, B - s)
            bill += _bill(L, Bc, Bc, m, k, n, comb=False); calls += 5 + 5 * L + 1 + (Bc > 1) + _build_trips(buf, m, k, n, L, Bc)
            items += Bc if seq else 1
        for s in range(0, B, _MAXB):
            Bc = min(_MAXB, B - s); pbill += 2.0 * m * k * n * Bc + (0 if seq else (Bc - 1) * m * n)
            pcalls += 1 + (Bc if seq else (Bc > 1)); terms += Bc if seq else 1
    bill += (items - 1 + bool(first)) * X + _comb(L, m * n) * m * n; calls += 4 * L + 2 * len(_c12(L, m * n)) + 3
    pbill += (terms - 1) * m * n; pcalls += terms   # copy of the first term, one add per later term
    return bill + PLAIN_PRICE * calls <= pbill + PLAIN_PRICE * pcalls

def sum_families(families, levels, *, out=None):
    """Sum of batched products (a_b @ b_b over every b of every family) accumulated in leaf space, combined once; a family
    flagged 'sequential' contributes each batch item as its own term (elder runs)."""
    fams = [(f[0], f[1], len(f) > 2 and bool(f[2])) for f in families if int(f[0].shape[0]) > 0]; a0, b0, _ = fams[0]
    m, k, n = int(a0.shape[1]), int(a0.shape[2]), int(b0.shape[2])
    if levels <= 0 or any(not _fast_ok(a, b, 1 << int(levels)) for a, b, _ in fams):
        total = None
        for a, b, sequential in fams:
            for start in range(0, int(a.shape[0]), _MAXB):
                products = fnp.matmul(a[start:start + _MAXB], b[start:start + _MAXB])
                terms = [products[i] for i in range(int(products.shape[0]))] if sequential else [fnp.sum(products, axis=0)]
                for term in terms: total = term.copy() if total is None else fnp.add(total, term)
        if out is not None: fnp.copyto(out, total); return out
        return total
    step = _MAXB
    if cap := _sf_deep(fams, m, n, levels): levels, step = int(levels) + 1, cap
    buf = _get_buffers(m, max(int(a.shape[2]) for a, b, _ in fams), n, levels, a0.dtype, step)
    if not _sf_fused_cheaper(fams, m, n, int(levels), step, buf):   # PLAIN_PRICE: the plain metered family sum
        total = None
        for a, b, sequential in fams:
            for start in range(0, int(a.shape[0]), _MAXB):
                products = fnp.matmul(a[start:start + _MAXB], b[start:start + _MAXB])
                terms = [products[i] for i in range(int(products.shape[0]))] if sequential else [fnp.sum(products, axis=0) if int(products.shape[0]) > 1 else products[0]]
                for term in terms: total = term.copy() if total is None else fnp.add(total, term)
        if out is not None: fnp.copyto(out, total); return out
        return total
    one = _get_plan(m, k, n, levels, a0.dtype, 1); acc, pp0 = one.get("leaf_accumulator"), plan_view(one, "pp0")
    if acc is None:
        acc = one["leaf_accumulator"] = _exact(buf, "c1", (7,) * levels, int(pp0.shape[-2]), int(pp0.shape[-1]), 1)[0]
    chunks = [(a3[s:s + step], b3[s:s + step], seq) if int(a3.shape[0]) > step else (a3, b3, seq)
              for a3, b3, seq in fams for s in range(0, int(a3.shape[0]), step)]
    last, v = sum(int(ac.shape[0]) if seq else 1 for ac, _, seq in chunks) - 1, 0
    for ac, bc, seq in chunks:
        plan = _get_plan(m, int(ac.shape[2]), n, levels, ac.dtype, Bc := int(ac.shape[0]))
        if v == 0 < last and not seq:  # first of several terms: the product (or its batch sum) lands in the accumulator
            _run_leaf(plan, ac, bc, out=None if Bc > 1 else plan_view(one, "acc1"))
            if Bc > 1: fnp.sum(plan["pp"], axis=0, out=acc)
            v += 1; continue
        _run_leaf(plan, ac, bc)
        if seq: terms = [plan_view(plan, "pp:%d" % e) for e in range(Bc)]  # every batch item is its own term
        else:
            terms = [p0 := plan_view(plan, "pp0")]
            if Bc > 1: fnp.sum(plan["pp"], axis=0, out=p0)
        for x in terms:
            if v == 0 < last: fnp.copyto(acc, x)
            elif 0 < v < last: fnp.add(acc, x, out=acc)
            elif 0 < v == last: fnp.add(x, acc, out=pp0)
            elif x is not pp0: fnp.copyto(pp0, x)
            v += 1
    _combine(one); return _deliver(one, True, out, two_d=True)

def _sym_views(h, kh, dtype, B):  # workspace for the 6-product symmetric scheme at half size h, contraction kh, batch B
    need, ws = 6 * B, _sym_ws.get((h, kh, str(dtype)))
    if ws is None or int(ws["ac"].shape[0]) < need:
        ws = _sym_ws[h, kh, str(dtype)] = {"ac": fnp.empty((need, h, kh), dtype=dtype), "bc": fnp.empty((need, kh, h), dtype=dtype), "views": {}}
    if (v := ws["views"].get(B)) is None:
        A, Bm = ws["ac"][:need], ws["bc"][:need]; A6, B6 = fnp.reshape(A, (6, B, h, kh)), fnp.reshape(Bm, (6, B, kh, h))
        v = ws["views"][B] = {"A": A, "B": Bm, "A4": fnp.reshape(A6[0:4], (2, 2, B, h, kh)), "B4": fnp.reshape(B6[0:4], (2, 2, B, kh, h)),
                              "a": [A6[i] for i in range(6)], "b": [B6[i] for i in range(6)]}
    return v

def _split_ok(m, k, levels):
    lc = int(levels) - 1
    return lc >= 1 and m % 4 == 0 and k % 2 == 0 and m // 4 % (1 << lc) == 0 and k // 2 % (1 << lc) == 0 and m // 4 >> lc >= SPLIT_LEAF

def _sym_split(a2, b2, levels, out, two_d):
    """Symmetric m x m product as 19 quarter-size products (three 6-product symmetric blocks + one general block), workspace
    views made once per (q, kh); output quadrants copied, lower half by transposition."""
    (m, k), ws = (int(a2.shape[0]), int(a2.shape[1])), _split_ws.get(str(a2.dtype)); q, kh = m // 4, k // 2
    if ws is None or ws["q"] < q or ws["kh"] < kh:
        qmax, kmax = max(q, ws["q"] if ws else 0), max(kh, ws["kh"] if ws else 0)
        A, B = fnp.empty((SPLIT_BATCH, qmax, kmax), dtype=a2.dtype), fnp.empty((SPLIT_BATCH, kmax, qmax), dtype=a2.dtype)
        ws = _split_ws[str(a2.dtype)] = dict(q=qmax, kh=kmax, A=A, B=B, A6=fnp.reshape(A[:12], (2, 6, qmax, kmax)),
                                             B6=fnp.reshape(B[:12], (2, 6, kmax, qmax)), views={})
    if (w := ws["views"].get((q, kh))) is None:
        A, Bm, A6, B6 = ws["A"][:, :q, :kh], ws["B"][:, :kh, :q], ws["A6"][:, :, :q, :kh], ws["B6"][:, :, :kh, :q]
        w = ws["views"][q, kh] = dict(A=A, B=Bm, a01=A6[:, 0:2], a3=A6[:, 3], a4=A6[:, 4], a5=A6[:, 5], a2=A6[:, 2],
                                      A12=A[12:18], A60=A6[0], A18=A[18], b0=B6[:, 0], b12=B6[:, 1:3], b3=B6[:, 3],
                                      b4=B6[:, 4], b5=B6[:, 5], B12=Bm[12:18], B61=B6[1], B614=B6[1, 4], B18=Bm[18], P=None)
    X = fnp.transpose(fnp.reshape(a2, (2, 2, q, 2, kh)), (0, 1, 3, 2, 4)); Y = fnp.transpose(fnp.reshape(b2, (2, kh, 2, 2, q)), (2, 0, 3, 1, 4))
    sub, add, copyto = fnp.subtract, fnp.add, fnp.copyto
    copyto(w["a01"], X[:, 0]); add(X[:, 1, 0], X[:, 1, 1], out=w["a3"]); x00 = X[:, 0, 0]
    sub(w["a3"], x00, out=w["a4"]); sub(x00, X[:, 1, 0], out=w["a5"]); sub(X[:, 0, 1], w["a4"], out=w["a2"])
    copyto(w["A12"], w["A60"]); copyto(w["A18"], X[0, 1, 1])
    y00, y01, y11 = Y[:, 0, 0], Y[:, 0, 1], Y[:, 1, 1]; copyto(w["b0"], y00); copyto(w["b12"], Y[:, 1])
    sub(y01, y00, out=w["b3"]); sub(y11, w["b3"], out=w["b4"]); sub(y11, y01, out=w["b5"])
    copyto(w["B12"], w["B61"]); sub(w["B614"], Y[1, 1, 0], out=w["B18"]); old = _MAXB; set_max_batch(SPLIT_BATCH)
    _src_memo[0] = w.setdefault("memo", {})   # w["A"] / w["B"] are this workspace's persistent views: split views memoised
    try: P = matmul(w["A"], w["B"], int(levels) - 1, detach=False)
    finally: set_max_batch(old); _src_memo[0] = None
    if (pv := w["P"]) is None or pv[0] is not P: pv = w["P"] = (P, [P[j:18:6] for j in range(6)], P[18])
    P6 = pv[1]; p0, p3 = P6[0], P6[3]; u2, c11 = add(p0, P6[4]), add(p0, P6[1])
    c12 = add(u2, p3); add(c12, P6[2], out=c12); u3 = add(u2, P6[5]); c22, c21 = add(u3, p3), sub(u3[2], pv[2])
    if out is None:   # M10: one block assembly of the 16 q x q blocks (same values, same numel(out) bill)
        c11t, c12t, c22t = fnp.swapaxes(c11, 1, 2), fnp.swapaxes(c12, 1, 2), fnp.swapaxes(c22, 1, 2)
        full = fnp.block([[c11[0], c12[0], c11[2], c12[2]], [c12t[0], c22[0], c21, c22[2]],
                          [c11t[2], fnp.transpose(c21), c11[1], c12[1]], [c12t[2], c22t[2], c12t[1], c22[1]]])
        return full if two_d else fnp.expand_dims(full, 0)
    h, o2 = m // 2, out if two_d else out[0]
    o = lambda br, i, bc, j: o2[br * h + i * q:br * h + (i + 1) * q, bc * h + j * q:bc * h + (j + 1) * q]  # quadrant (br,bc), sub-block (i,j)
    for blk, (br, bc) in enumerate(((0, 0), (1, 1), (0, 1))):  # (c11 c12; c12^T c22) per symmetric block, c21 for the general one
        c12b = c12[blk]; copyto(o(br, 0, bc, 0), c11[blk]); copyto(o(br, 0, bc, 1), c12b); copyto(o(br, 1, bc, 1), c22[blk])
        copyto(o(br, 1, bc, 0), c21 if blk == 2 else fnp.transpose(c12b))
    copyto(o2[h:, :h], fnp.transpose(o2[:h, h:])); return out

SYM_TAIL = True   # lossless switch: B = 1 symmetric products whose m is not a split multiple run as split(m1) + plain tail
SYM_TAIL_MAX = 0.08   # ... only when the tail m - m1 is at most this fraction of m (measured break-even near 1/11)

SYM_TAIL_FMM = True   # exact_r2: the tail columns (m, m - m1) as ONE metered fmm product at the depth its exact bill picks
                      # (levels_for, then _depth: plain when that is cheaper) instead of two plain products (same values)

def _sym_tail(a2, b2, levels, out, two_d, m1):
    """Symmetric m x m product as the quarter split of the leading m1 rows / columns plus the tail columns: the
    (m, m - m1) block a2 @ b2[:, m1:] (tail and corner) and the transpose of its top part (same products, split shape for
    the bulk)."""
    m = int(a2.shape[0])
    if out is None: out = fnp.empty((m, m) if two_d else (1, m, m), dtype=a2.dtype)
    o2 = out if two_d else out[0]; bt = b2[:, m1:]
    _sym_split(a2[:m1], b2[:, :m1], levels, o2[:m1, :m1], True)
    if SYM_TAIL_FMM and (lt := levels_for(a2, bt)):
        fnp.copyto(o2[:, m1:], matmul(a2, bt, lt, detach=False)); fnp.copyto(o2[m1:, :m1], fnp.transpose(o2[:m1, m1:])); return out
    fnp.matmul(a2[:m1], bt, out=o2[:m1, m1:]); fnp.copyto(o2[m1:, :m1], fnp.transpose(o2[:m1, m1:]))
    fnp.matmul(a2[m1:], bt, out=o2[m1:, m1:]); return out

def matmul_sym(a3, b3, levels, *, out=None, b_key=None):
    """Symmetric product by the caller's contract: quarter split when the shape allows it (B = 1, m = n, m divisible by
    4 * 2^(levels-1)), else the 6-half-product scheme."""
    two_d = len(a3.shape) == 2; B, (m, k, n) = 1 if two_d else int(a3.shape[0]), (int(a3.shape[-2]), int(a3.shape[-1]), int(b3.shape[-1]))
    if levels <= 1 or m != n or not _fast_ok(a3, b3, 1 << int(levels)):
        return fnp.matmul(a3, b3, out=out) if out is not None else fnp.matmul(a3, b3)
    if B == 1 and m == n and _split_ok(m, k, levels): return _sym_split(a3 if two_d else a3[0], b3 if two_d else b3[0], levels, out, two_d)
    if SYM_TAIL and B == 1 and (m1 := m - m % (4 << (int(levels) - 1))) < m and m - m1 <= SYM_TAIL_MAX * m and _split_ok(m1, k, levels):
        return _sym_tail(a3 if two_d else a3[0], b3 if two_d else b3[0], levels, out, two_d, m1)
    h, kh = m // 2, k // 2; v = _sym_views(h, kh, a3.dtype, B); a, b = v["a"], v["b"]
    fnp.copyto(v["A4"], fnp.transpose(fnp.reshape(a3, (B, 2, h, 2, kh)), (1, 3, 0, 2, 4)))
    fnp.subtract(a[0], a[2], out=a[5]); fnp.add(a[2], a[3], out=a[2]); fnp.subtract(a[2], a[0], out=a[4]); fnp.subtract(a[1], a[4], out=a[3])
    ws = _sym_ws[h, kh, str(a3.dtype)]
    if b_key is None or ws.get("bkey") != (b_key, B):
        fnp.copyto(v["B4"], fnp.transpose(fnp.reshape(b3, (B, 2, kh, 2, h)), (3, 1, 0, 2, 4)))
        fnp.subtract(b[3], b[2], out=b[5]); fnp.subtract(b[2], b[0], out=b[2]); fnp.subtract(b[3], b[2], out=b[4])
    ws["bkey"] = (b_key, B) if b_key is not None else None; plan = _get_plan(h, kh, h, levels - 1, a3.dtype, 6 * B)
    _run_leaf(plan, v["A"], v["B"], b_key=None if b_key is None else ("sym", b_key)); _combine(plan); _deliver(plan, False, None)
    if (pv := plan.setdefault("derived", {}).get(("sym", B))) is None:
        std6 = fnp.reshape(_pstd(plan, "std"), (6, B, h, h))
        # opsfrugal F1 (flopscope #275): the left operand of the column add is a cached broadcast view of out's full
        # (B, 2, h, h) shape, so the add is billed for every element it writes (same values; one call per plan, not per add)
        pv = plan["derived"]["sym", B] = (std6[0], std6[1], fnp.broadcast_to(std6[0][:, None], (B, 2, h, h)), std6[4][:, None],
                                          fnp.moveaxis(std6[2::3], 0, 1), fnp.moveaxis(std6[3:1:-1], 0, 1))
    p1, p2, p1_, p6_, p57, p35 = pv
    if out is None: out = fnp.empty((m, n) if two_d else (B, m, n), dtype=a3.dtype)
    o = fnp.reshape(out, (B, 2, h, 2, h)); col = o[:, :, :, 1]; c11, c12, c21 = o[:, 0, :, 0], col[:, 0], o[:, 1, :, 0]
    fnp.add(p1, p2, out=c11); fnp.add(p1_, p6_, out=col); fnp.add(col, p57, out=col); fnp.add(col, p35, out=col)
    fnp.copyto(c21, fnp.swapaxes(c12, 1, 2)); return out

def gen(a, b, a_key=None, b_key=None): return matmul(a, b, L, a_key=a_key, b_key=b_key) if (L := levels_for(a, b)) else fnp.matmul(a, b)
def sym(a, b, b_key=None): return matmul_sym(a, b, L, b_key=b_key) if (L := levels_for(a, b)) else fnp.matmul(a, b)
