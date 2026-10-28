# SPEC: the mathematics of ex_34 as the clean dense program should compute it

Author: Fable B (mathematician). Status: complete (sections 0-8); corrected 2026-10-28 after Fable A's review (2.1, 2.4, 1.6(ii), 1.7, 1.4, 3.3, 6).
Source of truth: `source/estimator_plain/` (abbreviations as in SOURCE_MAP: M = moments.py, MF = moment_formulas.py,
MM = moment_math.py, B = bank.py, K = pk4.py, E = estimator.py). Every equation carries the line(s) it was read from.
Where the README and the code disagree, the code wins and the case is listed in section 7 (Discrepancies).

Design assumptions of the clean version (from PLAN.md): every neuron kept (no COL/S/OUT/row/hub/terminal/core/
covariance crops), every source dense for its whole life (no elders, no shared basis), every third-cumulant source
leg dense (the questions in 1.6 decide the repair and the negative packet), plain `@`, no n^3 object ever formed,
products chunked per source. Everything that is dropped is an "economy" in SOURCE_MAP 2a/2b; everything kept is a
statistic (2c) or an engine closure. TRUNCATION tags mark what remains approximate.

## 0. Notation, objects and loop order

Network: n = width (1024), L = depth (16), layers l = 0..L-1, row vectors, no biases (README 45-50):

    z_0 = x W_0,   z_l = h_(l-1) W_l,   h_l = relu(z_l),   x ~ N(0, I_n).           (0.1)

Layer L-1 is the terminal layer (mean only). Hidden layers are 0..L-2. Source b is the third cumulant born at hidden
layer b (b = 0..L-2). A "consumer" t is the layer whose pre-activation receives deliveries (t = 1..L-1).

Per-layer symbols (all (n,) unless stated; everything in the unit-prevariance gauge of 5.1 except at the terminal):
  mu        pre-activation mean of the layer (gauged: "premean")
  pc, C     pre-activation covariance Cov(z) (n,n), diag pc = 1 in the gauge; x = pc with zero diagonal
  sd, zeta  sd = sqrt(diag pc) (= 1), zeta = mu / sd
  d, e, f   Gaussian vertices: d = Phi(zeta), e = phi(zeta)/sd, f = -zeta e / sd                 (MF:88-106)
  dv,ev,fv  Edgeworth-corrected vertices (transport gate dv, curvature ev, third vertex fv)       (MF:109-127)
  m, ov, k3, k4  corrected mean, Var(h), kappa3(h), kappa4(h) of the post-activation            (MF:181-183)
  z3        kappa3 diagonal of z, z3_i = kappa3(z_i,z_i,z_i)
  b         (2,1) slice of kappa3(z): b_ij = kappa3(z_i, z_i, z_j) (repeated index FIRST)
  rt        partial trace rt_j = sum_i b_ij
  q4        kappa4 diagonal of z
  gm, ga    transported core gm = W^T A W (n,n), ga = diag gm; gv = diag(W^T W); gram = diag(gv) after layer 0
  A = T + sI the fourth-cumulant trace core of the previous layer's h (section 5.3)
  v (cubic_vector)  the covariance-star vector of the engine (section 5.4); pv = v W its transport
  Sym(x,y,z)  sum of the THREE distinct placements when two arguments coincide (B:20-22); Sym6 = all six placements.

Loop order (M:`predict_weights` 476-531, M:`_complete_layer_transition` 405-462, README "Per-layer pipeline"). For
each hidden layer l, with the gauged frame (mu_l, pc_l, W_l) already prepared:

  L1  pv = v W_l                                                          (M:480)
  L2  bank.deposit(l): directional fold (temporary) -> PK4 fold (persistent) -> slice b_l, z3_l, rt_l  (section 1.5, 2, 3, 4)
  L3  l >= 1: add the covariance-star images to b_l, z3_l, rt_l             (M:495-498, section 5.4)
  L4  extra_fourth = TC dq (bank.tc_pending, section 2) + PK4 (single site + leg read, section 3)  (M:501-510)
  L5  _advance_hidden_moments: Gaussian basis; core push gm; E_GA added to ga; q4 = 6 ga gv + extra_fourth;
      Edgeworth rows; unary statistics; Mehler rows; pair powers (Cov(h), post, pv3, u row sums); core update
      (radial solve, edge block if l < TCP_START, new A).                   (M:759-841, section 5.2, 5.3)
  L6  birth: _create_cubic_source (section 1.2) and the covariance-star birth (section 5.4)   (M:302-382)
  L7  if TCP_START <= l+1 <= L-1: tc_stash_engine(v_l = diag pc_l, dv_l, ev_l, a1tc_l, B_l = star-free b_l)  (M:420-422)
  L8  prepare layer l+1: gauged frame (pc_(l+1) = W^T Cov(h_l) W, scale), or the terminal prevariance   (section 5.1)
  L9  l >= 1: response_input (directional context, section 4)             (M:429-439)
  L10 bank.accumulate(l): admit source l, directional pre-read, gate history by dv_l, transport every source to
      l+1, write z3_(l+1), rt_(l+1), slice coefficients (pending), E_GA for l+1; TC state  (section 1.3, 1.4, 4)
  L11 pk4.record_birth(l)                                                   (section 3)
Terminal (l = L-1): L1 then _terminal_mean (section 5.5). Finally every row is multiplied by radius_mean(n) (5.1).

## 1. The cubic source bank (B)

### 1.1 Model (B:12-33, README 76-88)

First-order gate transport of third cumulants:

    kappa3(h_l) ~= D_l^(x3) kappa3(z_l) + B_l,   kappa3(z_(l+1)) = (W_(l+1)^T)^(x3) kappa3(h_l),  D_l = diag(dv_l)   (1.1)
    kappa3(z_t) = sum_(b<t) (Pi_(b->t)^T)^(x3) B_b,   Pi_(b->t) = W_(b+1) D_(b+1) W_(b+2) ... D_(t-1) W_t             (1.2)

TRUNCATION: first-order (linear) gate transport of cumulants; the ReLU's nonlinear response to kappa3 is carried
only through the births (B_l), the TC (section 2), PK4 (3) and the directional response (4).

Representation of a source (B:24-33), hub rows a = neurons of the birth layer b (every hub kept in the clean version):

    B_b = sum_a [ Sym(P_a, H_a, H_a) + Sym(P_a, P_a, R_a) + beta_a P_a^(x3) ] - Sym(K, G, G)                    (1.3)

with, at the birth layer, P_a = delta_a (identity row), H_a the completed square, R_a the repair row, beta_a a
scalar, and (K, G) the rank-one negative packet (1.2.4). All of P, H, R, G, K are rows over the current layer's
neurons and are transported by (1.4); beta is not transported. "Hub" is a fixed label of the row.

Transport of one row X (any of P, H, G, K, RIGHT) from layer s to s+1 (B:2663-2692, 2727-2764, 2490-2498, 911-955):

    X^(s+1) = (dv_s o X^(s)) W_(s+1)      (gate on the neuron axis of layer s, then the matmul)                  (1.4)

Exception: the newest source is born in h-space and is NOT gated on its first transport (B:2437-2440, 917); its P
leg at s+1 is just W_(s+1)[hubs, :] (B:2477-2482). Every later transport gates it like the rest.

### 1.2 Birth of source l (M:`_create_cubic_source` 302-382) [bank-facing part]

Inputs at hidden layer l after L5: C = pc_l (gauged, diag 1), the delivered slice b = b_l (star images included,
M:511, 527), the corrected vertices (dv, ev, fv) (M:317), gm = transported core (M:318), Q22/4 =
`fourth_pair_quarter` (M:862, section 5.2.4), the pair power post (= HiddenMoments.repeated_post, section 5.2.5),
k3 = kappa3(h), ov, pre_vector pv, Cov(h) (M:367).

(a) The "extra" coupling of the incoming cumulants into the square (M:319-324):

    extra_ij = b_ij fv_i dv_j + b_ji ev_i ev_j + C_ij^2 fv_i ev_j + (Q22/4)_ij (2 fv_i) ev_j                      (1.5)

(b) Completed square and its rank-one split (M:326-344):

    s_i = sqrt(ev_i),  pos_i = (s_i > 0),  den_i = 2 s_i where pos else 1
    F_ij = extra_ij / den_i,          fdiag_i = extra_ii / 2 where pos else 0                                   (1.6)
    u_ij = C_ij dv_j                  (the Gaussian leg; u_ii = dv_i since C_ii = 1)
    H_ij = s_i u_ij + F_ij                                                                                      (1.7)

(c) Hub-diagonal repair (MM:202-223, called M:334-341): with coef_i = cubic_self_coefficient_i = the folded left
vector of the post pair power at row (0,0,1), coef = g_1 + (z3/6) g_4 + (q4/24) g_5 (M:915, 927; g = centred-square
family, 5.2.3):

    H_ii <- coef_i / (2 s_i)   where pos_i;  fdiag_i <- fdiag_i + s_i (H_ii_new - H_ii_old)                    (1.8)

so that s_i H_ii = ev_i dv_i + fdiag_i stays true. Purpose (MM:210-211): 2 H_ii H_ij ~ coef_i C_ij dv_j, the first
order term of post.

(d) Negative packet (M:343-348, B:866-868): (L, sigma, G) = rank-1 SVD of F; L <- L sigma (n,1), G (1,n):

    F ~= L G,   K_a = L_a^2 (so K = sum_a L_a^2 P_a at birth),   -Sym(K, G, G) removes Sym(P_a, F_a, F_a) approx.  (1.9)
    crossdiag_i = (L G)_ii = L_i G_i,   left_norm_i = L_i^2

(e) Repair and beta (M:350-371):

    r_ij = post_ij - dv_i^2 b_ij dv_j - 2 H_ii H_ij - H_ji^2 + 2 L_i^2 G_i G_j + G_i^2 L_j^2 - stardelta_ij,  r_ii <- 0   (1.10)
    beta_i = k3_i - dv_i (dv_i^2 z3_i) + 3 (crossdiag_i^2 - H_ii^2) - stardelta_ii                             (1.11)

(ddv = dv^2 and dz = dv^2 z3 from MF:195, 199.) stardelta is the (2,1)-slice image change of the covariance-star
birth (5.4, MM:172-199): the repair owns new-star minus inherited-star. Reading of (1.10): post_ij is the pair law's
value of kappa3(h_i, h_i, h_j); the subtracted terms are the (2,1)-slice images, at the birth layer, of the
transported incoming slice (dv_i^2 b_ij dv_j), of sum_a Sym(P_a,H_a,H_a) (whose (i,i,j) entry is 2 H_ii H_ij +
H_ji^2) and, added back, of the negative packet (whose (i,i,j) entry is -(2 K_i G_i G_j + G_i^2 K_j) with K = L^2).
So R_a is defined to make the represented (2,1) slice at birth equal post exactly (before the rank cut of 1.3(c)).

Source record (M:372-382): u = H (n rows), r, beta, _negL = L, _negG = G, _tc_fdiag = fdiag - s crossdiag (used by
the TC, section 2), _dh = diag H. The clean version has no hub admission: hub_count = h_count = n for every source,
kept = None, hub order = natural (B:2616-2661 and `_hub_masks` 2773-2799 disappear; see TRUNCATION note in 1.7).

### 1.3 accumulate(l): transport to l+1 and the diagonal/trace readouts (B:2382-2607)

Let k = l + 1 sources exist after admission (sources 0..l). Dense per-source state at layer l (generation 0):
H_j, P_j (n_hub x n each), RIGHT_j (3 x n), LEFT_j (n x 3, fixed at birth), beta_j (n), K_j, G_j (1 x n, j >= 1).

(a) Gate (B:2682-2692): for every source j < l: H_j <- H_j o dv_l, P_j <- P_j o dv_l (column axis), RIGHT_j <-
    RIGHT_j o dv_l; packets: [K_j; G_j] <- [K_j; G_j] o dv_l for j < l (B:936). The fresh source l is not gated.
(b) Install the fresh source: H_l = u (B:2440), P_l = I (never stored), repair fit (c), K_l = L o L (B:908).
(c) Repair fit (B:2703-2725, 672-696, 699-739): (U, S, V^T) = rank-2 SVD of r (block-Krylov in the plain version;
    the clean version may use an exact truncated SVD, SOURCE_MAP 2b). LEFT_l = [U S, ones], RIGHT_l = [V^T; vrow] with
    the conservation row vrow (B:699-739):
        delta_i = -(U S V^T)_ii,  colsum_j = sum_i r_ij - sum_i (U S V^T)_ij
        vrow_j = colsum_j / n for a non-hub column; on hub columns (all columns are hub columns when every hub is
        kept): vrow_j = (colsum_j - delta_j) / (n - 1);  beta_l <- beta_l + 3 (delta - vrow)               (1.12)
    The stored repair is R = LEFT RIGHT (n x n per source), RECON (B:2503). See 1.6(i) for the dense alternative.
(d) Transport (1.4) of every leg of every source and of RIGHT, then R^(l+1) = LEFT RIGHT^(l+1) (B:2499-2503);
    packets: K, G of old packets gated and transported, the fresh packet's K uses the full W (B:940-942).
(e) kappa3 diagonal at l+1 (B:2505-2519, header 36):

    z3_j = sum_src sum_a [ 3 P_aj H_aj^2 + 3 P_aj^2 R_aj ] + sum_src sum_a beta_a P_aj^3 - negdiag_j,
    negdiag_j = 3 sum_packets G_j^2 K_j                                                                        (1.13)

(f) Slice coefficients at l+1 (B:2609-2614):  CU = 2 P o H,  CP = H o H + 2 P o R + beta P o P  (per source, hub rows).
(g) Partial trace at l+1 (B:2571-2602): rt_j = sum_i b_ij with b's diagonal set to z3:

    rt_j = sum_src sum_a [ (sum_i CU_ai) H_aj + (sum_i CP_ai) P_aj + (sum_i P_ai^2) R_aj ]
           - sum_src sum_a [ CU_aj H_aj + CP_aj P_aj + P_aj^2 R_aj ]  (the represented diagonal, B:2579)
           + z3_j - (negtrace_j - negdiag_j),                                                                   (1.14)
    negtrace_j = column sums of the negative slice incl. its diagonal = sum_packets (sum_i a_i) kg_j   (B:955)

    i.e. rt = column sums of the off-diagonal slice plus the exact kappa3 diagonal. The repair part uses
    RREAD = (P o P)^T LEFT (n x 3 per source) and rt_rep = (sum_i RREAD_i) RIGHT (B:2514, 2571-2573).
(h) E_GA read (section 4) and pending = (coefficients, legs at l+1, RREAD, RIGHT, z3, rt) for deposit(l+1).
Terminal target (l+1 = L-1, B:2535-2552): only (e) is formed (added to terminal_cubic_diagonal); no slice, no
trace; the legs at the terminal stay for PK4 (3) and the terminal TC (2.6). In the clean version the terminal
column plan is all columns.

### 1.4 deposit(t): the (2,1) slice at consumer t (B:2037-2058, 2122-2169)

Order inside deposit(t) (B:2052-2058, 2134-2136): (1) temporary directional fold on CU, CP, RREAD of sources
< t-1 and on CP of source t-1 (section 4.3); (2) PK4 persistent fold on CU, CP of every source (section 3.4);
(3) the slice; (4) restore the saved copies. The copies are taken inside (1) BEFORE the directional fold (B:2097,
2104: COMB2 rows of every old source, RREAD, and the newest source's CP row) and restored after (3), so they also wipe
the PK4 fold (2) from those rows: the PK4 fold persists ONLY on the newest source's CU row (in the three-leg form its
C_u and C_X rows), which the next accumulate's directional pre-read sees (B:2049-2050, K:81-83 read literally).
[Corrected 2026-10-28; the earlier "with (2) and without (1)" overstated the persistence.]

Non-TC consumer (t < TCP_START = 4), B:2152-2166:

    b_t(i,j) = sum_src sum_a [ CU_ai H_aj + CP_ai P_aj ] + sum_src RREAD_i RIGHT_j - sum_packets (2 K_i G_i G_j + G_i^2 K_j)
               - dirshared_i pc_ij  (directional radial part, section 4.3)                                       (1.15)
    b_t(j,j) <- z3_j (1.13);  third_bank[t] = z3;  trace_override[t] = rt (1.14)

Here CU, CP are the coefficient rows at t (written by accumulate(t-1) = (f), folded by (1)-(2)), H, P the legs at t,
RREAD_i = (P o P)_(.i)^T LEFT, i.e. the repair term is sum_a P_ai^2 R_aj. The negative slice is a^T kg with
a = [G o G; 2 G o K] and kg = [K; G] (B:945-946, 1007).
TC consumer (t >= 4): the same slice, but the fresh source's rows carry the theta-scaled TC fold and the old sources'
sum is formed as the pre-lift query (section 2.4). Per-entry readers of b_t: the Mehler rows 6-9, 14, 15 (5.2.4),
the birth's extra (1.5) and repair (1.10), the next TC stash (B_t), and the (2,2) row sums (5.2.5).

### 1.5 Where each readout goes

  z3_t      -> Edgeworth rows (MF:118-126), unary statistics, Mehler rows 16-18 and the fold rows 21-23, birth beta.
  b_t       -> Mehler rows 6-9, 14, 15 and their (3,1)/(2,2)-side contractions; extra (1.5); repair (1.10); TC stash.
  rt_t      -> radial solve of the core update (M:786-787, 1032): the covariance-star vector fitted to rt.
  TC dq, PK4 -> q4 (M:799-801).   E_GA -> ga (M:793-795).

### 1.6 The two exactness questions

(i) Dense residual leg R instead of rank-2 SVD + conservation column.
    Statistic being approximated: r of (1.10), an (n x n) matrix per source with zero diagonal, entering B_b as
    sum_a Sym(P_a, P_a, R_a) (its (2,1) slice at birth is r itself: Sym(P_a,P_a,R_a)_(iij) = R_ij for a = i, plus
    the P_j placement which is diagonal). The plain code stores R = U S V^T + ones v^T (rank 3) and moves the
    diagonal mismatch into beta (1.12). Its transport is RIGHT (3 x n) through (1.4) with LEFT fixed: exact for a
    rank-3 matrix because Sym(P,P,R) transports leg by leg and R's rows are the only transported object
    (P_a is transported separately). The dense alternative stores R_a (n x n) and transports it by (1.4) like H.
    Is it the same statistic, strictly refined? Yes: in exact arithmetic the rank-3 object equals the dense R only
    when rank r <= 2; otherwise the dense R is the exact term of (1.3) whose column sums (1.14) and (2,1)-slice
    at birth the rank-3 fit only reproduce in part (column sums exactly, by the conservation row; entries at rank 2).
    Every readout of R is linear in R: z3 (3 P^2 R), CP (2 P o R), the trace, the slice (P^2 R), the TC old-repair
    term (section 2.2, where R's role is the same linear term), the directional xr = R . u (4.1). So the dense R is
    the limit of the rank-k fit as k -> n-1 and the dense version is a strict refinement with NO new formula: set
    LEFT = I_n, RIGHT = r (zero diagonal), beta unchanged by (1.12) (delta = 0 and the conservation row vanishes
    because colsum - fit colsum = 0). Equivalently drop LEFT/RIGHT/RREAD and keep one more leg R per source:
        CP = H o H + 2 P o R + beta P o P,  z3 += 3 sum P^2 R,  slice += sum_a P_ai^2 R_aj  (same formulas).  (1.16)
    Exact R: r of (1.10) with r_ii = 0 and the diagonal carried by beta (1.11), nothing else. The hub-diagonal
    repair (1.8) and stardelta stay as they are (they define post's decomposition, not the rank cut).
    Reads stay single matmuls: the slice term sum_a P_ai^2 R_aj is (P o P)^T R, one (n x n)^T (n x n) product per
    source, same shape as the H term. Cost: one more leg transport (2 n^3) and one more slice product per source
    per layer (CENSUS.md). Recorded effect of rank 3 vs 2: -0.03 % MSE (B:304), so the refinement is expected to be
    small; its value is in removing the SVD, the conservation row and the beta_delta bookkeeping.
    Caveat: SOURCE_MAP 2c lists the repair rank as a statistics truncation. The dense R changes the statistic only
    towards the model's own exact term; measure alone (PLAN rule 2).

(ii) Exact removal of the spurious F (x) F term.
    With H_a = s_a u_a + F_a (1.7), F_a = extra_a / (2 s_a):
        Sym(P_a, H_a, H_a) = e_a Sym(P_a, u_a, u_a) + (1/2) Sym6(P_a, u_a, extra_a) + Sym(P_a, F_a, F_a)        (1.17)
    (s_a Sym6(P,u,F) = Sym6(P,u,extra)/2.) The model's intended term is the first two; the third is spurious and the
    plain code removes it only at rank one and only for sources b >= 1 (B:869: the first source has no packet).
    Three exact forms, all the same statistic (the first two terms of (1.17)):
      (A) completed square + dense third leg: keep H, P and store F_a (n x n) as a third transported leg; the
          negative read becomes -Sym(P_a,F_a,F_a) exactly: slice -= sum_a [ 2 (P o F)_ai F_aj + (F o F)_ai P_aj ],
          z3 -= 3 sum_a P_aj F_aj^2. (Coefficient rows CF = 2 P o F on the F leg, and -F o F added to CP.)
      (B) three-leg form: legs u_a, X_a := extra_a (with the repaired diagonal X_aa = coef_a - 2 e_a u_aa = coef_a - 2 e_a dv_a in the gauge (u_aa = C_aa dv_a, C_aa = 1), from (1.8):
          2 fdiag_new = coef - 2 e dv), and P_a; the tensor is e_a Sym(P_a,u_a,u_a) + (1/2) Sym6(P_a,u_a,X_a).
          Readouts (clean):  b(i,j) = sum_a [ (2 e_a P_ai u_ai + P_ai X_ai) u_aj + (P_ai u_ai) X_aj
                                              + (e_a u_ai^2 + u_ai X_ai) P_aj ]  + repair + beta terms,       (1.18)
                             z3_j = sum_a [ 3 e_a P_aj u_aj^2 + 3 P_aj u_aj X_aj ] + ...
          i.e. coefficient rows C_u = 2 e P o u + P o X, C_X = P o u, C_P = e u o u + u o X (+ 2 P o R + beta P o P).
      (C) the brief's "e Sym(P,u,u) + 2 sqrt(e) Sym(P,u,F) + Sym(P,P,Q)": the middle term is (B)'s Sym6 term
          written with F (2 s Sym6(P,u,F)/... note Sym6, not the 3-term Sym); a Sym(P,P,Q) term cannot represent
          Sym6(P,u,X) (it is of a different index type), so (C) as written is (B) plus the repair term, with Q = R.
    Simplest for a reader: (B). It has no square root, no division by 2 sqrt(e), no e = 0 guard (M:325-328; the
    guard's effect, H_a = extra_a when e_a = 0, is replaced by the correct limit e Sym(P,u,u) -> 0 with the
    Sym6(P,u,X) term kept), no fdiag bookkeeping (_tc_fdiag becomes 0, section 2.1), no SVD, no negative bank,
    and no "first source has no packet" exception. All reads stay single matmuls: three legs (u, X, P), three
    coefficient^T @ leg products per source for the slice, elementwise for z3. Cost: three transported legs
    instead of two legs + a 2-row packet: +1 leg transport (2 n^3) per source per layer and +1 slice product.
    Memory: 3 n x n per source per generation instead of 2 (CENSUS.md).
    Strict refinement? Yes for sources b >= 1: the rank-one packet removes Sym(P, LG, LG) where LG is the best
    rank-one approximation of F; the exact forms remove Sym(P,F,F) itself, so the difference is the rank-one
    residual of F squared, which the plain code carries as a spurious term. For source 0 the plain code keeps
    Sym(P,F,F) entirely (no packet): the exact form CHANGES source 0's statistic (extra at layer 0 is not zero:
    b_0 = 0 but C o C fv ev and the (Q22/4) 2 fv ev terms of (1.5) are present). This is a deliberate decision to
    record: the model's own text calls the term spurious, so removing it for source 0 is the faithful reading of
    the model, not of the code. Measure alone.
    The TC (2.1) reads sqrt(e)(H - L G) ("F1 removed"); with (B) and F1 -> F this is exactly e u (section 2.1). The
    directional pre-read (4.1) contracts the legs themselves (u, X, P) and needs no substitution.

### 1.7 Truncation tags for the bank

TRUNCATION (hub-separable sources): B_b is a sum over hubs a of tensors with one leg P_a fixed to the hub; this is
exact for the tree-level birth (one neuron's ReLU creates the cumulant) but the repair R and beta absorb what the
pair law `post` says beyond that form, at the (2,1)-slice level only: distinct-index entries kappa3(h_i,h_j,h_k),
i,j,k distinct, of the birth are represented only through the hub form. Kept.
TRUNCATION (first-order gate transport): (1.1). Kept.
TRUNCATION (negative packet rank one, no packet for source 0): removed by 1.6(ii) if adopted; otherwise kept.
TRUNCATION (repair rank 2 + conservation): removed by 1.6(i) if adopted; otherwise kept.
Economy items dropped in the clean version (all SOURCE_MAP 2a/2b): hub admission (gate > 0.01, saturation 0.003,
drop tolerance 1e-3, P-only saturated hubs), COL/S/OUT/row plans and the rank-two fill, 64-padding, shared elder
basis and switch ages, block-Krylov SVD. Saturated hubs (gate >= 1 - 0.003) [corrected 2026-10-28]: in the
completed-square form their H row is F = X/(2 sqrt e), HUGE (e tiny), and its square is removed only at rank one, so
ex_34's P-only rule is a numerical necessity of that form, not an economy; the PK4 dressed leg g' = u + (X - coef P)/(2e)
diverges likewise, so PK4's live mask must exclude them (K:257 does). In the three-leg form the X row is finite
(e Sym(P,u,u) -> 0, (1/2) Sym6(P,u,X) stays) and could be kept in the bank while PK4 still excludes the hub; that is a
measurable variant, not the faithful reading. The clean program keeps ex_34's two hub rules (gate > 0.01; P-only when
saturated) as statistics. Convention difference recorded in REVIEW.md (a): ex_34 subtracts the saturated hub's full H
image from r and beta and then drops the H row; the clean program drops both consistently.

## 2. The trace correction TC (B:1452-1737, 1739-1900, 1902-2035)

### 2.1 What it is

Consumers t = TCP_START..L-1 = 4..15 (B:343, 2418). At consumer t the "newest TC" has source s = t-1 (the source
born at t-1, index k_in = t-1). It is a correction to the kappa4 diagonal q4 of z_t (and, folded, to the (3,1)
slice Q31 of z_t) that the trace core cannot carry: the fourth cumulant produced when the newborn ReLU's
QUADRATIC response (coefficient ev_i/2 on z_i^2) meets the THIRD cumulant the newborn inherited from older sources.
Writing h_(t-1),i ~ dv_i z_i + (ev_i/2)(z_i^2 - ...) and z_t,j = sum_i W_ij h_i, the Leonov-Shiryaev expansion of
kappa(z_i z_i, z_k, z_l, z_m) contains 6 kappa3(z_i,z_k,z_l) C_im-type terms; with the quadratic neuron at any of the
four positions (x4) and ev/2 this gives the coefficient 12 of (2.3) and the 3, 3, 6 of (2.6)-(2.7).

Objects at consumer t (source layer s = t-1, everything in the gauge of layer s for the i index and of layer t for j):

  Q_ij  = kappa3^(old)(z_s,i ; ztilde_t,j ; ztilde_t,j), i = hub (neuron of layer s), j = consumer neuron,      (2.1)
          the mixed third cumulant between one source-layer pre-activation and two copies of the consumer's, from
          the sources older than s (born <= t-2) plus their repairs minus their negative packets; "ztilde" = the
          first-order transport (W o dv). It is the pre-lift query: Q = (qt^T + A B) / dv_s,i (B:1501) with
             qt[j,i]   = sum_(src<=t-2) sum_a [ CU^(t)_aj (dv_s H^(s))_ai + CP^(t)_aj (dv_s P^(s))_ai ] - negpackets  (B:1879-1899)
             (A B)[i,j] = sum_(src<=t-2) sum_r (dv_s RIGHT^(s))_(src,r,i) RREAD^(t)_(src,j,r)                     (B:1498-1501)
          i.e. qt^T/dv = the old-source (2,1) slice b_t with its single index left at layer s. The division by
          dv_s,i un-gates the legs (the stored legs at generation s are gated, 1.3(a)); 0/0 -> 0 (B:742-747).
          NB (B:1767-1776): qt for the TC excludes the newest source's directional-fold change but INCLUDES the
          old sources' temporary directional fold and the PK4 fold (both are in COMB2 during deposit).
  Y_ij  = dv_s,i W_ij                                                                                            (B:1502)
  M     = sqrt(e) (H0 - L G0)  =  e u + sqrt(e) (F - F1)  (the newborn's leg content, birth layer, n x n)
          (H0 = H at birth incl. the repaired diagonal (1.8); F = X/(2 sqrt e); F1 = L G0 its rank-one approximation).
          The exact limit of the packet (F1 -> F) is M = e u and _tc_fdiag = sqrt(e)(F - F1)_aa -> 0: with the exact
          removal 1.6(ii)(B) the TC reads M = e u, cv = e dv - a/2, and the fold acts on the u leg only (2.4).
          [Corrected 2026-10-28; the earlier text "M = e u + X/2" was wrong (Fable A, DISCREPANCIES 1: x2.3 blow-up).]
  cv_i  = M_ii - a_i/2,   a = a1tc = 2 h - 2 m dv  (MF:197; the "isotropic fresh-leg coefficient")             (B:1516)
          (code: cv = e v d + fdiag - a/2 with fdiag = _tc_fdiag = fdiag_new - sqrt(e) (LG)_ii; since
           sqrt(e) H_ii = e v d + fdiag_new, cv = [sqrt(e)(H0 - LG0)]_ii - a/2 = M_ii - a/2.)
  effc_ij = (M W)_ij - cv_i W_ij = sum_(k != i) M_ik W_kj + (a_i/2) W_ij                                        (B:1517)
          (code: sqrt(e) (H^(t)_fresh - L (G0 W)) - cv W, where H^(t)_fresh = H0 W is the untransported-gate fresh
           leg at t, B:1507-1514; so effc is the newborn's hub leg at t with the OWN-hub pairing k = i replaced by
           a_i/2.)
  Zp_ij = W_ij effc_ij - a_i |W_.j|^2 / (2 (n+4))                                                                (B:1518)
          (the last term is the isotropic (trace-core) image of the own-hub term: it is already in the core.)

### 2.2 Repeated-index subtraction

Q_ij counts index triples at layer s in which indices coincide (k = l, i = k, i = k = l); those parts are the
layer-s (2,1) slice B_s and diagonal z3_s, which the engine already uses. They are removed (B:750-768, 846-855):

  Qt_ij = Q_ij - [ sum_k Bo_ki Y_kj^2  +  2 Y_ij sum_l Bo_il Y_lj  +  dd_i Y_ij^2 ],                             (2.2)
  Bo = off-diagonal part of B_s (B_s = b_s before its covariance-star image, stashed at L7),  dd = z3_s diagonal.

In the plain code Bo is replaced by its rank n/256 = 4 truncated SVD U S V^T and dd = diag(B_s) - 3 diag(U S V^T)
(the truncation's spurious diagonal is picked up once by each of the three terms). SOURCE_MAP 2c lists this rank as a
statistics truncation ("lower ranks scored better (adjusted)"). The exact dense reading is (2.2) with the full Bo
(two (n x n) @ (n x n) products per TC layer: Bo^T (Y o Y) and Bo Y) and dd = z3_s. The plain code's reassociation
`_repeated_query_residual` (R such that sum Z o R = the contraction) is exact.
TRUNCATION (TC repeated-index rank 4): keep rank 4 to reproduce ex_34; the dense (2.2) is the exact form of the same
statistic. Recommend: implement (2.2) exactly, measure alone.

### 2.3 The scalar correction and the newest TC

  dq_j = 12 sum_(i in hubs) Zp_ij Qt_ij        (B:1521)                                                           (2.3)

Added to q4 of z_t (L4). At the terminal consumer (t = L-1) the same with W = W_(L-1) ungauged on the terminal
columns (2.6).

### 2.4 The TC slice fold (TC_SLICE_FOLD; B:1522-1529, 1739-1854)

Rationale (B:1780-1785): in the pair law the (3,1) row (1/6) Q31_ij F^(3)_i G'_j and the (2,1) row (1/2) b_ij
F^(2)_i G'_j share G'; at the Gaussian vertices f/e = -mu/v, so (1/6) Q31 f = (1/2)(theta Q31) e with

    theta_j = -mu_j / (3 v_j)     (consumer's gauged mean and prevariance; v = 1 in the gauge)                   (2.4)

Delivering b + diag(theta) Q31 therefore reaches every reader of b (rows 6-9, 14, 15 of the pair law, extra, repair,
TC stash) with one coefficient change. From TCP_START on the engine drops its own (3,1) edge block (5.3.3) for this
reason. TRUNCATION (theta fold): corrected vertices differ from Gaussian ones; the z3 diagonal and the partial trace
do not receive the fold (B:1785; the fold's column sums are not added to rt).

The newest TC's Q31 contribution to z_t, Q31_jj' = kappa4(z_j,z_j,z_j,z_j') from the same mechanism, has three parts
(quadratic neuron at position j' / at a j position pairing with j' / at a j position pairing with j):

  cp_ij  = 3 effc_ij Qt_ij - 3 cv_i W_ij Qt_ij + 6 dv_i mixed_ij        (coefficient on the P leg W_ij')        (2.5)
  ch_ij  = 3 sqrt(e_i) W_ij Qt_ij                                       (coefficient on the H leg H_ij')         (2.6)
  s_j    = -(3/(n+4)) sum_i a_i Qt_ij                                   (isotropic image, on all rows)           (2.7)
  cpf_ij = cp_ij + s_j W_ij

and the delivered slice becomes (B:1804-1849)

  b_t(j,j') += theta_j [ sum_i ch_ij H^(t)_ij' + sum_i cpf_ij W_ij' - (sum_i ch_ij L_i) G^(t)_j' ],              (2.8)

the last term removing the rank-one F1 part of H. With 1.6(ii)(B) (exact removal, F1 -> F): sqrt(e)(H - F) = e u, so
ch multiplies the u leg only, ch_u = 3 e W o Qt against u^(t) = (C dv) W; there is no X-leg fold term and the rank-one
term disappears. [Corrected 2026-10-28; the earlier "ch_X = (3/2) W o Qt" was wrong.] In (2.5), -3 cv W Qt corrects the own-hub
pairing of the H term (ch H contains 3 M_ii W_ij W_ij' Qt_ij; the rule "own-hub pairing C_ii -> a_i/2" applied to
both the P-position and the H-position terms gives the net 3 Qt (M_off W - M_ii W + a W) on the P leg, as coded).
mixed_ij (B:1522, 783-843): the third part needs kappa3^(old)(z_s,i ; ztilde_j ; ztilde_j') which the bank does not
hold. Per consumer column j it is reconstructed from Qt_.j: with x = Y_.j, e = x/|x|, the minimum-Frobenius-norm
symmetric cubic tensor R_j with zero repeated-index entries and R_j(., e, e) = Qt_.j / |x|^2 is found by the
diagonal-plus-rank-two normal equations of `_completion_coefficients` (exact formulas B:797-823), and
mixed_.j = |x| R_j(., e, Zp_.j) (B:826-843). TRUNCATION (distinct-index completion): the mixed query is the
minimum-norm completion, not the true cumulant. Kept as coded; the formulas are in the two functions and should be
transcribed literally (they are per-column vector algebra, O(n^2)).

### 2.5 History TCs (TRACE_HISTORY_DEPTH = 3; B:1651-1737, 1902-2035, 1549-1572)

The TC born at consumer t-g (source b = t-g-1), g = 1 .. TRACE_HISTORY_DEPTH-1 = 1..2, i.e. ages 2 and 3 at t
(B:347-349), is re-read at consumer t with the SAME formulas (2.1)-(2.8), with these substitutions:

  W_ij        -> P^(b->t)_ij  (source b's transported P leg at t)            (legs[1], B:1671, 1730)
  (M W)_ij    -> sqrt(e_b) (H^(b->t)_ij - L_b,i G_b^(t)_j)  i.e. effc = sqrt(e)(H - L g) - cv P   (B:1678, 1732)
  Y_ij        -> dv_b,i P^(b->t)_ij                                            (B:1680)
  Q^(b)_t[i,j] = kappa3^(sources < b)(z_b,i ; ztilde_t,j ; ztilde_t,j) = (qh_b^T + A_b B_t)/dv_b,i               (2.9)
                with A_b = source-b-layer RIGHT rows of the sources < b (stored in the chain) and B_t = RREAD at t
                of the sources < b (B:1669), and qh_b the history query (2.10); the same SVD of B_b (chain) in (2.2).
  theta, a, cv, sqrt(e), dv_b (chain entries sqc, cv, aI, dI) are those of source b's birth.                      (B:1677)

  dq_t += 12 sum_i Zp^(b)_ij Qt^(b)_ij  for each history b                    (B:1564-1566)
  b_t(j,j') += theta_j [ sum_i ch^(b)_ij H^(b->t)_ij' + sum_i cpf^(b)_ij P^(b->t)_ij' - (ch^(b)T L_b)_j G_b^(t)_j' ]  (B:2026-2034)

(The code passes 3 theta and omits the 3's: B:1683-1685, 1693-1695. The lite fold (completion term omitted) applies
only on the row plan, which the clean version does not have: use the full (2.5).)

History query (2.10): for g >= 1, with generation t-g legs (stored gated by dv_(t-g), 1.3(a)),

  qh_(t-g)[j,i] = sum_(src <= t-g-1) sum_a [ CU^(t)_aj Leg^(t-g)_(src,H,a,i) + CP^(t)_aj Leg^(t-g)_(src,P,a,i) ] - negpackets at t-g   (2.10)

(g = 1 is qt of (2.1).) Direct form: one pass per generation. The plain code nests it in Horner form
(B:1921-1927): acc_g = [acc_(g+1) + CP^(t)T of source t-g-1 (P = identity at its birth) + CU^(t)T H^(t-g)_(t-g-1)] @ (W_(t-g) o dv_(t-g)),
starting from acc_A = sum over sources <= t-A-1 of CU^T H^(t-A) + CP^T P^(t-A) with A = 1 + chain length (= 3 at
steady state). With dense sources this is an exact reassociation of (2.10) (transport is linear), and it is
cheaper (CENSUS.md); recommended. "Omit sources whose representation changed after t-g" (B:1916-1918, SOURCE_MAP 2c)
refers to elders switching basis inside the window; with no elders no source ever changes representation, so the
rule is void and the Horner query equals (2.10) exactly. The history fold coefficients cpf^(b), ch^(b) are
computed from acc at generation g = t-b and enter the deposit as (2.8)-type terms with source b's legs at t; the
plain code adds them to the entry generation and lets the Horner lift transport them (B:2026-2032), which is the
same thing by linearity.

### 2.6 Terminal TC (B:1574-1635)

At t = L-1 (consumer of source L-2): no slice; dq by (2.3) and the history terms of 2.5 on the terminal columns
(all columns in the clean version; the plain code crops to |z^2-1| exp(-z^2/2) > 0.3 and the terminal column plan).
The query qt is (2.1) with the terminal W (ungauged rows divided by the last scale, 5.1). dq is added to the
terminal q4 (M:296-298).

### 2.7 TC bookkeeping as a direct formula (what the clean program needs per TC layer t >= 4)

  stash at t-1 (L7): v = diag pc_(t-1) (= 1), dv, ev, a1tc, B_(t-1) (star-free slice), and the newborn's X_aa/2.
  at deposit(t): Q (2.1) from the gated legs at generation t-1 and the consumer coefficients; Qt (2.2); Zp; dq (2.3);
                 fold (2.5)-(2.8) into the fresh source's deposit; for b = t-2, t-3 (if >= 0 and t-b-1 >= 4... see
                 note): Q^(b) (2.9)-(2.10), their dq and folds. Keep for later consumers: Zp needs only legs at t;
                 Q^(b) needs the chain entries (A_b, the B_b SVD or Bo, dv_b, a_b, cv_b, sqrt(e_b), L_b).
  Note on the window: the chain holds the TC states of the last TRACE_HISTORY_DEPTH-1 = 2 consumers, so at t = 4 the
  chain is empty (consumers start at 4), at t = 5 it has the t = 4 state, from t = 6 two states. Sources b < 3 never
  get a TC (their consumer b+1 < TCP_START). tc_pending at t < 4 clears the chain (B:1647-1648).

## 3. PK4: per-neuron fourth-cumulant corrections (K)

### 3.1 Single site (K:377-403), consumer l >= 1

The trace core models kappa4(h_a) of the previous layer as 6 (T_aa + s). The exact unary kappa4(h_a) = k4_a
(MF:183, M:525) replaces it through diagonal powers p fitted jointly with the core:

    r_a = k4_a - 6 (T_aa + s),   p_a = ((n+4)/(n-2)) r_a - (3/((n-2)(n-1))) sum_b r_b                             (3.1)
    dq_j = sum_a p_a W_aj^4 + N_j [ -(6/(n+4)) sum_a p_a W_aj^2 + N_j (3/((n+2)(n+4))) sum_a p_a ],  N_j = |W_.j|^2  (3.2)

(A' = A - diag(p)/(n+4) + sum(p) I/(2(n+2)(n+4)) satisfies 6 A'_aa + p_a = k4_a; (3.2) is the read of
Sym6(A' (x) I) + diag powers minus the read of Sym6(A (x) I) along column j.) Added to q4 of z_l.

### 3.2 Leg families (K:22-49, 206-375), consumer t, every source b < t (dense: every source is "young")

Per source b at birth (K:265-304), hub rows a (natural order in the clean version; `live` = 1 where e_a > 0 and the hub
has an H row, else 0 and all coefficients of that row vanish; with 1.6(ii)(B) the row's X-leg content stays but the
PK4 coefficients still need e_a > 0):

    isq_a = live_a / sqrt(e_a),  de_a = dh_a isq_a  (dh = H_aa at birth, (1.8)),                                 (3.3)
    g'_a = isq_a H_a - de_a P_a   (the dressed Gaussian leg x'_a = H_a/sqrt(e_a) with its self entry removed; at
           birth x'_aj = C_aj dv_j + F_aj/e_a... exactly isq H - de P, and x'_aa = 0)
    6V_a = 12 live_a (d_a (1-d_a) - m_a e_a),   4c31_a = 12 live_a (1-d_a)(Var_a - m_a^2),   4f3_a = 4 f_a live_a  (3.4)
    (d, e, f, m, Var = corrected gate dv, ev, fv, mean m, ov of the birth layer: M:450-459 passes gate = dv etc.)
    cn'_a = |x'_a|^2 (birth, self entry excluded; K:277-278),
    tau6 = 6 V @ (x'^2 with cn' on the self entries) (natural (n,)),  tr6 = sum tau6                            (3.5)
    -p_u,a = tau6_a /(n-2) - tr6 /(2(n-2)(n-1))                                                                   (3.6)
    image rows: I0_a = -(3/(n+4)) (4c31_a + 4f3_a cn'_a),  I1_a = -(6V_a cn'_a)/(n+4) - 6 p_u,a... precisely
                I1_a = 6V_a cn'_a (-1/(n+4)) + (-p_u,a)(-6/(n+4)) + tr6/(2(n-1)(n+4)),  I2_a = -6V_a/(n+4)       (3.7)

Read at consumer t (K:345-372) with the legs at t, A0 = P g', A1 = P^2, A2 = g'^2 (hub x column):

    dq_j = sum_b sum_a [ 4c31 A0 A1 + 4f3 A0 A2 - p_u A1 A1 + 6V A1 A2 ]_aj
         + sum_b N^(b)_j sum_a [ I0 A0 + I1 A1 + I2 A2 ]_aj,     N^(b)_j = sum_a A1_aj = sum_a P_aj^2             (3.8)

Meaning (K:30-45): tree-level families star 4 sum f3 P g'^3, u 6 sum V P^2 g'^2, (3,1) 4 sum c31 P^3 g', each minus
its trace-core image (the harmonic part is already in A). Hidden consumers read the legs at t; the terminal reads the
terminal legs (K:333-338). TRUNCATION (PK4 families): star, u, (3,1) only; the path family (paths through the
covariance) and the (2,2) pair are not read (K:37-38: -1 % MSE for 9.7 GF). "Young" (K:423, 326): in the plain code
the hidden read widens to every young (dense) source and the fold applies to young sources only; with no elders every
source is young at every consumer, so read and fold cover every source b < t. This is the faithful dense reading:
the exclusion of elders was a representation issue (K:423, rationale not recorded), not a statistic.

### 3.3 The persistent response fold (K:405-498), inside deposit(t) before the slice (B:2134-2136)

With theta_j = -mu_j/(3 v_j) of the consumer (2.4) and, per source, the represented star + u residual quartic
F(p, g) = sum_a [ 4f3 p g^3 - p_u p^4 + 6V p^2 g^2 ] + N sum_a [ u p g + v p^2 + w g^2 ],  N = sum_a p_a^2,
u = I0 + (3/(n+4)) 4c31 (the (3,1) image removed, K:449), v = I1, w = I2, evaluated at p = P_.j, g = g'_.j:

    A = (1/4) dF/dp,   B = (1/4) dF/dg   (elementwise in hub a and column j; K:469-484)                          (3.9)
    CU_aj += theta_j isq_a B_aj,    CP_aj += theta_j (A_aj - de_a B_aj)       (chain rule g' = isq H - de P)      (3.10)

so that the slice carries b + diag(theta) Q31^(star+u) with Q31_ij = sum_a A_ai P_aj + B_ai g'_aj (K:57-61). Not
undone by PK4 itself, but the deposit's restore of the directional-fold copies (1.4) removes it from every row except
the newest source's CU row; that row keeps it until accumulate(t) rebuilds COMB2, and the directional pre-read (4.1)
sees it there. TRUNCATION (theta fold, PK4 variant): same as 2.4; the (3,1) family is excluded from the fold.

## 4. The directional response E6 (B:2266-2380, 2060-2120; context M:429-439)

### 4.1 Pre-read at accumulate(l) (B:2266-2332), before gating

u = ga1_l = 2 h (1 - d) (MF:186; the variance response dVar_G(relu)/dmu, Gaussian vertices). For every source
j < l with legs at layer l (before this accumulate's gating) and R_j = LEFT RIGHT at l:

    xH_ja = H_ja . u,   xP_ja = P_ja . u,   xR_ja = R_ja . u    (hub contractions; (B:2287-2294))               (4.1)
    chained (E_PROP, B:2303-2314): x2 += rho x2_prev (previous level's xH, xP of sources < l-1),
       xr += rho [xr_prev; -dl] with dl = previous dvec (net of its radial image, 4.2) on the newest source's row,
       ku += rho ku_prev (packets), rho = 2 kappa/(n+4), kappa = core_propagation (5.3.2).                         (4.2)
    T(u)_aa = sum_j sum_i [ xH_ji CU_ia + xP_ji CP_ia + xR_ji P_ia^2 ] - sum_packets ku_p a_pa      (B:2317-2328)   (4.3)
    (CU, CP, PP of layer l from accumulate(l-1), with the PK4 fold of deposit(l) and without the directional fold)
    dvec_a = dv_a^2 T(u)_aa                                                                                       (4.4)

T(u)_ab = sum_c kappa3(z_l)_abc u_c; the engine's core carries only its radial image Sym(v, pc)(., ., u) (B:2274-2277).

### 4.2 Read at the next layer (B:2334-2380), delivered as E_GA to layer l+1

    read_j = sum_j' sum_i [ xH CU^(l+1) + xP CP^(l+1) + xR PP^(l+1) ]_ij - packets   (the same contractions against
             the NEXT layer's coefficients = sum_(a,b) W_aj W_bj (D T(u) D)_ab)                                    (4.5)
    radial image of the bank history: vb = v_rad - pv,  ca = pc u,  sb = vb . u  (rad = (v_rad, pc u, u.v_rad, u) of
             the radial solve 5.3.1 at layer l; pv the transported star vector)                                   (4.6)
    rad_d_a = 2 dv_a^2 vb_a ca_a + sb ov_a;   tv = [vb dv; ca dv] W;   read -= 2 tv_0 tv_1 + sb vnext              (4.7)
    dga_j = -[ (dvec - rad_d) @ (W o W) - read ]_j / (n+4)                                                        (4.8)

vnext = diag pc_(l+1) (terminal: the terminal prevariance). dga is added to ga of layer l+1 (M:793-795; terminal
M:284-286), hence q4 += 6 gv dga and the Q22, Q31 diagonal parts. Side effects for 4.3: _response_state[4] =
dvec_g - rad_d and _response_radial_fold = (vb dv, ca dv, tv, sb). TRUNCATION (E6): first Mehler order for the
radial image (vnext stands in for the gated covariance), E_PROP's least-squares kappa; the read is of (exact -
radial) along u only.

### 4.3 Temporary directional fold in deposit(l+1) (B:2060-2120)

s_j = 3 theta_j gv_j/(n+4) = -mu_j gv_j/((n+4) v_j) (consumer l+1; gv = diag W^T W of its weights). For the sources
j < l (the "old" ones, i.e. all but the newest l):

    CU_aj += s_j (xH_a P_aj + xP_a H_aj),  CP_aj += s_j (xP_a R_aj + (beta_a xP_a + xR_a) P_aj + xH_a H_aj)          (4.9)
    RREAD_a. += s (xP_a P_a)^T LEFT  (the repair's slice read; and the pending side factor with it)              (4.10)
    newest source l: CP_aj -= dcf_aj,  dcf = s_j (dl_a P_aj + cag_a tv_0,j + vbg_a tv_1,j),  dl = dvec_g - rad_d   (4.11)
    whole slice: b(i, j) -= (sb s_i) pc_ij                                                                       (4.12)

(4.9)-(4.12) add theta x the (3,1) image of T(u) minus its radial image to the delivered slice; (4.9)-(4.11) are
undone after the deposit (B:2056-2058). The TC query excludes (4.11) but includes (4.9) (B:1767-1776).
TRUNCATION: as 2.4 (theta fold).

## 5. The engine (M, MF, MM)

### 5.1 Input law, radial factor and gauge

Spherical-radial factorization (MM:226-244, M:533, README 51-57): x = (r/sqrt n) u, r ~ chi_n independent of u uniform
on the sphere of radius sqrt n; E[h_l(x)] = E[r/sqrt n] E[h_l(u)] by positive homogeneity (no biases). The engine
propagates the spherical law and multiplies every layer mean at the end by

    radius_mean(n) = E[chi_n]/sqrt n = base prod_(j = first, first+2, ..., n-2) (1 + 1/j) / sqrt n,
    first = 1, base = sqrt(2/pi) for odd n; first = 2, base = sqrt(pi/2) for even n   (exp of a sum of log1p).   (5.1)

Layer 0 (M:226-237, 582-584): the sphere has kappa3 = 0 and the isotropic kappa4 = Sym6(A (x) I) with A = -I/(n+2),
so gm = -pc/(n+2) and gram = pc (the metric W^T W IS pc at layer 0); z3_0 = 0, b_0 = 0 (M:235-237); the stf scale is
2 - 4/(n+2) instead of 2 (M:229, MF:178).

Unit-prevariance gauge (M:1104-1145; terminal M:395-403): for hidden layer l with previous scale sigma_(l-1) ((n,), 1 at
layer 0) and Cov(h_(l-1)) (W^T W for layer 0):

    W' = W_l / sigma_(l-1)[:, None]                  (rows divided by the previous scale)                        (5.2)
    mu' = m_(l-1) W',   pc' = W'^T Cov(h_(l-1)) W'    (full products; the clean version has no crops)
    sigma_l = 1 / sqrt(diag pc')
    W~_l = W' sigma_l[None, :],  mu_l = mu' sigma_l,  pc_l = sigma_l pc' sigma_l      (diag pc_l = 1)             (5.3)
    stored mean of layer l: m_l / sigma_l (M:523)                                                                 (5.4)

The terminal weight is only W' (rows divided), no column rescaling; its prevariance v_T = diag(W'^T Cov(h_(L-2)) W')
and terminal_z = (m_(L-2) W') / sqrt(v_T) (M:396-402). Exact reparameterization (homogeneity); kept because the trace
core's isotropic I and all thresholds are gauge dependent (SOURCE_MAP 1a). The Gaussian gate d = Phi(mu_l) (sd = 1).

### 5.2 Per-layer closure

5.2.1 Gaussian ReLU basis (MF:88-106), X ~ N(mu, sd^2), zeta = mu/sd, phi = phi(zeta), d = Phi(zeta):
    h = sd (zeta d + phi);  raw2 = sd^2 ((1+zeta^2) d + zeta phi);  raw3 = sd^3 ((3+zeta^2) zeta d + (2+zeta^2) phi);
    raw4 = sd^4 ((3 + (6+zeta^2) zeta^2) d + (5+zeta^2) zeta phi);
    e = phi/sd;  f = -zeta e/sd;  ds4 = (zeta^2-1) e/sd^2;  ds5 = (3-zeta^2) zeta e/sd^3;
    ds6 = (3 + (zeta^2-6) zeta^2) e/sd^4;  ds7 = (-15 + (10-zeta^2) zeta^2) zeta e/sd^5.                        (5.5)
    (d^k h/dmu^k = (-1)^k He_(k-2)(zeta) phi / sd^(k-1) for k >= 2.)

5.2.2 Edgeworth closure (MF:109-127), with the layer's z3 and q4 = 6 ga gv + extra_fourth (M:799-801):
    m  = h + (z3/6) f + (q4/24) ds4;   m2 = raw2 + z3 e/3 + q4 f/12;   m3 = raw3 + z3 d + q4 e/4;
    m4 = raw4 + 4 z3 h + q4 d;
    dv = clip(d + (z3/6) ds4 + (q4/24) ds5, 0, 1);  ev = max(e + (z3/6) ds5 + (q4/24) ds6, 0);
    fv = f + (z3/6) ds6 + (q4/24) ds7.                                                                           (5.6)
    TRUNCATION (Edgeworth order): first kappa3 and kappa4 terms only; the clips are statistics when they bind.

5.2.3 Unary statistics (MF:134-209). D1_k = k-th mu-derivative of E relu = (h, d, e, f, ds4, ds5, ds6, ds7)_k;
    D2 = (raw2, 2 D1_0, ..., 2 D1_4);  D3 = (raw3, 3 raw2, 6 D1_0, ..., 6 D1_3)  (orders 0..5).
    Centred families about the corrected mean m (orders k = 0..5):
      f_k = D1_k - m [k=0];  g_k = D2_k - 2 m D1_k + m^2 [k=0];  c_k = D3_k - 3 m D2_k + 3 m^2 D1_k - m^3 [k=0]   (5.7)
    ov = m2 - m^2;  k3 = m3 - 3 m m2 + 2 m^3;  k4 = m4 - 4 m m3 + 12 m^2 m2 - 6 m^4 - 3 m2^2;
    a1 = g_1 = 2h - 2 m d;  a2 = g_2 = 2d - 2 m e;  ga1 = 2 h (1-d);  ga2 = 2 d - 2 h e;
    u2 = 2d - 2 h e - 2 d^2;  u3 = e - h f - d e;  stf = S d - S m e - 2 d^2 (S = stf scale);  dh = d/2;
    gah = ga/2, gvh = gv/2;  z36 = z3/6;  q424 = q4/24;  astar = ev gv/2;  ddv = dv^2;  fv2 = 2 fv;
    a1tc = 2 h - 2 m dv;  dmv = dv^2 ga;  dz = dv^2 z3;  m7 = -3 (a2 d + a1 e);
    key = [stf, u2, sd u3];  lc = [2e, 2e, e, 2d];  lr = [u3, u3, u2, u3];  oqc = [gah, gvh];  oqr = [gvh, gah].  (5.8)
    (The archive notes that stf and the edge rows key/lc/lr were not derived; they are the submitted closure, MF:174-177.)

5.2.4 Mehler pair law (MF:27-67, 212-248; M:843-899). For i != j, x = pc with zero diagonal,
    E[F(z_i) G(z_j)] ~ sum_(r,s,k) K_rs[i,j] (x_ij^k/k!) F^(r+k)_i G^(s+k)_j,                                  (5.9)
    K_00 = 1, K_30 = z3_i/6, K_03 = z3_j/6, K_40 = q4_i/24, K_04 = q4_j/24, K_21 = b_ij/2, K_12 = b_ji/2,
    K_22 = Q22_ij/4, K_31 = Q31_ij/6, K_13 = Q31_ji/6, with
    Q22/4 = (ga_i gv_j + gv_i ga_j)/4 + gm o gram  (layer 0: gm o pc; later: diagonal only, ga gv)   (M:855-862)
    Q31/6 = gvh_i gm_ij  (+ gah_i gram_ij at layer 0)                                              (M:867-872)
    The 21 coefficient matrices M_q (rows q = 0..20, PAIR_ORDERS (r,s,k)):
      0 (0,0,0) 1; 1 (0,0,1) x; 2 (0,0,2) x^2/2; 3 (0,0,3) x^3/6; 4 (2,2,0) Q22/4; 5 (2,2,1) x Q22/4;
      6 (2,1,0) b/2; 7 (2,1,1) x b/2; 8 (1,2,0) b^T/2; 9 (1,2,1) x b^T/2; 10 (3,1,0) Q31/6; 11 (3,1,1) x Q31/6;
      12 (1,3,0) Q31^T/6; 13 (1,3,1) x Q31^T/6; 14 (2,1,2) (x^2/2) b/2; 15 (1,2,2) (x^2/2) b^T/2;
      16 (0,3,0) z3_j/6; 17 (0,3,1) x z3_j/6; 18 (0,3,2) (x^2/2) z3_j/6; 19 (0,4,0) q4_j/24; 20 (0,4,1) x q4_j/24.
    All with zero (or irrelevant) diagonal: b/2 and Q22/4 have their diagonals zeroed (M:864-866).
    Rows 21-25 are (3,0,0..2), (4,0,0..1): folded (ROW_FOLD, M:912-921) into the left vectors of rows 0..2:
      left_m <- left_m + z36 left_(21+m) + q424 left_(24+m) [m < 2]   for m = 0, 1, 2, with left_q = family_(r+k).  (5.10)
    STATISTIC (not a pure fold, M:917-920): rows 17, 20 and 18 reuse the FOLDED left vectors of rows 1, 1 and 2:
      left_17 = left_20 = folded left_1 = fam_1 + z36 fam_4 + q424 fam_5,  left_18 = folded left_2 = fam_2 + z36 fam_5,
    so the pair law contains the extra cross products
      row 17: (z3_i/6)(z3_j/6) x fam_4,i fam_1,j + (q4_i/24)(z3_j/6) x fam_5,i fam_1,j     [(3,3) and (4,3) at k = 1]
      row 20: (z3_i/6)(q4_j/24) x fam_4,i fam_1,j + (q4_i/24)(q4_j/24) x fam_5,i fam_1,j   [(3,4) and (4,4) at k = 1]
      row 18: (z3_i/6)(z3_j/6)(x^2/2) fam_5,i fam_2,j                                       [(3,3) at k = 2]
    while rows 16 and 19 (k = 0) keep the plain left vectors fam_0 (M:919, XXX: not recorded). Reproduce exactly.
    Pair powers (M:923-934), each = sum_q M_q[i,j] left_q[i] right_q[j], right_q = f_(s+k) always:
      Cov(h) (f, f) with diagonal <- ov;  post (g, f) with diagonal <- k3;  pv3 = (3,1) pair power (c, f).      (5.11)
    cubic_self_coefficient = folded left_g,1 = g_1 + z36 g_4 + q424 g_5 (M:927) -> (1.8).
5.2.5 (2,2) row sums (U_ROWSUM, M:935-952). u = M - ov ov^T - 2 Cov o Cov with M = sum_q M_q o (g_q h_q^T),
    g = left_g, h = right_g = g_(s+k); read only through
      urow_i = sum_(j != i) [sum_(q>=1) M_q,ij g_q,i h_q,j] + k4_i + g_0,i sum_j (h_0,j - ov_j) + (g_0,i - ov_i) sum_j ov_j
               - (g_0,i h_0,i - ov_i^2) - 2 (sum_j Cov_ij^2 - ov_i^2)                                           (5.12)
    (row 0 taken analytically against ov ov^T to avoid float32 cancellation; exact). Use float64 in the clean version.

### 5.3 Fourth-cumulant trace core

5.3.1 Model (README 93-101, M:1006-1089): kappa4(h) ~ Sym6(A (x) I) with the exact partial trace t_ij = sum_k
    kappa4(h_i,h_j,h_k,h_k):  A = T + s I,  T = t/(n+4),  s = -tr(t)/(2(n+2)(n+4))  (M:1088-1089, 227).        (5.13)
    Pushed through W: gm = W^T A W, gram = W^T W; q4 = 6 ga gv; Q22, Q31 as in 5.2.4. TRUNCATION (trace-core ansatz):
    only the harmonic/trace part of kappa4 is carried; PK4 and the TC restore pieces of the rest.
    Core push (M:562-615, dense): gm_offdiag = W^T T W (from T; the s I part only on the diagonal because gram is kept
    diagonal after layer 0: DEAD_DIAG), ga = diag(W^T T W) + s gv. Metric gram = diag(gv) (M:574-577: full W^T W
    measured x1.002 MSE for -5.5 GF; the clean version may form it: SOURCE_MAP 2b; Fable A's plan does).
    Radial solve (MM:96-151, M:1032): v_rad solves (2 pc + tr(pc) I) v = rt (the layer's partial trace) -> the
    covariance-star vector whose partial trace is rt; radial_image = (v_rad, pc ga1, ga1 . v_rad, ga1). The Chebyshev
    series equals linalg.solve to eps32; the clean version may use solve (float64) directly.
5.3.2 Core update (M:1052-1089, MF:251-315). Vectors: ca1 = pc a1, gma = gm a1, gra = gv a1 (layer 0: gram a1),
    p = Cov(h) ga1 (cross_covariance = Cov(h) output covariance, M:1040 uses pairs.cross_covariance = mehler.x: NB
    PairClosure(mehler.x, ...) so p = x ga1 with x = pc zero-diag; see Discrepancies D3), q2 = (x^2/2) ga2,
    bg = (b/2) ga1, btg = (b/2)^T ga2, qg = (Q22/4) ga2, vv = v_rad, ca = pc ga1, sr = ga1 . v_rad, mk = ones.
    Weights (MF:274-315): a1d = a1 d; fp = f p - 2 a1d; rho = d ga2 + e ga1; w3 = q2 + bg + qg;
      g = [vv d, f w3 + e btg, fp + sr e, -3 f, fp, -f, fp, -3 rho, e p, d ga1, m7]  (rows 0..10)
      r = [ca d, d, e, rho, e, rho, d, e, e, -2 f, d]
      cc = e ca1 - v d (a2 - 2 d^2) - v a1 e + (sr/2) d;   oc = [e ga/2, e gv/2];  or = [d gra, d gma]
      gmc = (a2 . gv / 4) d + e gra (+ a2 gv d past layer 0);   grc = (a2 . ga / 4) d + e gma (layer 0 only)
      xc = vv e;  xr = ca e;  hd = d;  hv = v mk (= v = diag pc = 1).                                            (5.14)
    kappa (E_PROP, M:1054-1055): effective = gmc + gvh m7;  kappa = (effective . d)/(d . d).
    Half trace (M:1056-1067):
      Z = sum_(q=0..10) M_q o (g_q r_q^T) + pc o (cc d^T) + gm o (gmc d^T) + x o (xc xr^T) + oc^T or + pv3
          - 3 ov_i Cov(h)_ij   (+ gram o (grc d^T) at layer 0)                                                   (5.15)
      + edge block (5.3.3) for l < TCP_START.
      t = Z + Z^T with diagonal <- urow (5.12);  s' = tr(t) (-1/(2(n+2)(n+4)));  T' = t/(n+4).                   (5.16)
5.3.3 Edge block (M:955-1004, 1068-1084), layers l = 0..3 only (from TCP_START on the TC fold delivers it, 2.4):
      left = sum_(k=2,4,6,8) M_k o (lc_k lr_k^T)  (rows x^2/2, Q22/4, b/2, b^T/2 with lc = [2e, 2e, e, 2d], lr = [u3, u3, u2, u3])
      edge_left = left + pc o (dh stf^T);  edge_trace_ij = (edge_left pc - left o (1 v^T))_ij d_j;  Z += edge_trace. (5.17)
      (Dense: no output/contraction crops; hv = v = diag pc.)
5.3.4 Terminal core diagonal (M:630-659): ga_T = diag(W'^T T W') + s gv on every column (dense), gv = diag W'^T W'.

### 5.4 Covariance star (MM:12-28, 154-199; M:480, 495-498, 367)

Sym3(v (x) C)_ijk = v_i C_jk + v_j C_ik + v_k C_ij, carried as v next to C; exact affine transport v -> v W with
C -> W^T C W; first-order gate v -> dv v, C -> D C D. Images: slice S_ij = C_ii v_j + 2 v_i C_ij; diagonal 3 v_j C_jj;
trace tr(C) v + 2 C v. At hidden layer l >= 1 (M:495-498): b_l += S(pv, pc), z3_l += 3 pv diag(pc), rt_l += trace.
Birth (MM:172-199, M:349-351, 367): fresh core star (a, Mstar) = (astar = ev gv/2, D gm D) and inherited star
(dv pv, D pc D); their total partial trace tr(Mstar) a + 2 Mstar a + tr(D pc D)(dv pv) + 2 (D pc D)(dv pv) is matched by
one star Sym3(v' (x) Cov(h)): (2 Cov(h) + tr Cov(h) I) v' = that trace; stardelta = S(v', Cov(h)) - S(dv pv, D pc D)
goes into r and beta (1.10)-(1.11); v' becomes cubic_vector. TRUNCATION (star): distinct-index entries of the replaced
stars are approximate (only the partial trace is preserved).

### 5.5 Terminal readout (M:268-300)

premean = m_(L-2) W' (W' = terminal weight, rows divided by sigma_(L-2)), sd = sqrt(v_T), zeta = premean/sd, basis (5.5);
    z3_T = terminal_cubic_diagonal (1.13 on the terminal legs) + 3 pv_T v_T  (pv_T = v W')
    q4_T = 6 (ga_T + dga_T) gv_T + PK4_T (3.1 + 3.8 on the terminal legs) + TC_T (2.6)
    mean_T = h + z3_T f/6 + q4_T ds4/24 + (z3_T^2/72) ds6                                                        (5.18)
(the z3^2/72 ds6 term is the second-order Edgeworth kappa3^2 term; it appears at the terminal only). Output rows are
clipped at 0 (E:331) and multiplied by radius_mean (5.1).

## 6. Truncation summary and the faithful dense reading of unexplained choices

Kept (statistics): first-order gate transport (1.1); hub-separable births (1.7); Edgeworth order (5.6); Mehler row
set and the row 17/18/20 dressing (5.10); trace-core ansatz (5.13); TC depth 3, TCP_START = 4, theta folds (2.4,
3.3, 4.3), distinct-index completion (2.4); PK4 families star/u/(3,1) (3.2); E6 at first Mehler order (4.2);
covariance-star distinct entries (5.4); the terminal z3^2/72 term; the asymmetry "directional fold undone, PK4 fold
kept" (B:2049-2050); the TC query includes the old sources' directional fold (B:1767).
Removed or exact in the clean version: all crops and plans, elders, block-Krylov SVD, Horner-form approximation
(void), lite history fold (void), rank-4 repeated index (exact (2.2) recommended), and, if adopted, 1.6(i)/(ii).
Unexplained choices (SOURCE_MAP 2c list) and their dense reading:
- double-gated elder repair rows (B:1490-1497): void without elders (every RIGHT row is gated once and un-gated once).
- rows 16/19 undressed (M:919): keep as coded (statistic).
- switch cutoff layer <= depth-3 (B:2680): void (no switches).
- condition limit (B:378): void (no shared basis).
- completion tolerance 1e-10 (B:796, 803, 814): keep the checks; on failure the plain code raises DegenerateMoments
  (newest TC) or keeps the lite fold (history TC, B:1689-1692). Dense reading: same behaviour.
- gram kept diagonal after layer 0 (M:574-577): the pre-economy form would change, at every layer as at layer 0,
  Q22/4 (gm o gram), Q31/6 (ga_i gram_ij / 2), gra = gram a1, the gram o (grc d^T) half-trace term, gmc (MF:297-299:
  the a2 gv d term is the diagonal-gram specialisation) AND gm itself (M:612-613: the s I part of A pushes to s W^T W,
  off-diagonal when gram is dense). Fable A's dense-gram test wired the first four and measured x1.343, so it measured
  a hybrid; the recorded x1.002 (M:577) refers to a wiring not recoverable from the plain source. The diagonal gram is
  ex_34's statistic and is what ships.
- e = 0 guard in the birth (M:325-328) and PK4 live mask (K:257, 266): with 1.6(ii)(B) the guard disappears; the PK4
  coefficients of a hub with e = 0 remain zero (3.3).

## 7. Discrepancies (code vs README / docstrings), recorded while reading

D1. B:869 / README 84: the README's model removes Sym(K,G,G) for every source; the code registers no packet for
    source 0 (`self.fresh = ... if self.count else None`), so source 0 keeps its spurious F (x) F term.
D2. K:30-35 states the PK4 vertices use "Edgeworth-corrected vertices"; M:450-459 passes gate = dv, curvature = ev,
    third = fv, mean = m, variance = ov: consistent. But K:241-249 zeroes the self-entry debit de for hubs outside the
    birth COL plan; dense: every hub is in COL, de = dh isq everywhere.
D3. M:1040 `p = cross_covariance @ ga1` where PairClosure's first field is `mehler.x` (pc with zero diagonal), while
    MF:268 documents p as "x ga1" and the dataclass field is named cross_covariance; the half-trace term
    `cross_covariance o (xc xr^T)` (M:1060) is likewise x, not Cov(h). The code is the truth: p = x ga1.
D4. B:1683 passes 3 theta to the history fold and writes cp = Qt (effc - cv P) (3 theta) without the explicit 3 of the
    newest fold (B:1524): same statistic, documented in 2.5.
D5. README 234 says LEG2 generations are 120 MiB each "TRACE_HISTORY_DEPTH + 1 = 4": consistent with B:1162.
D6. SOURCE_MAP 1a says the (2,2) row sums "diagonal is set to ov and k3" for cov/post; the u row sums use k4 on the
    diagonal (M:947): consistent with M:935-939, documented in (5.12).

## 8. Checks performed

- (1.17) and (1.18) verified numerically at width 7 on random (C, e, d, extra, coef) with the hub-diagonal repair
  applied: max error 8e-16 (scratch check117.py). The identities use the 3-placement Sym for Sym(P,H,H) and the
  6-placement Sym6 for Sym6(P,u,X), as defined in section 0.
- clean/k3_reference.py (deliverable 3, chunked preallocated bank, faces read once per layer) validated against
  k3_original/dense_simple.py at widths 8, 16, 32, 64 for 3-5 layers: mean, covariance, full cubic tensor and q agree
  to <= 8e-15 at every layer. Peak memory at depth 16: tracemalloc 18 / 64 / 250 MiB and RSS 100 / 226 / 612 MiB at
  widths 128 / 256 / 512 (bank 11 / 45 / 180 MiB), so ~1 GiB tracked and an expected RSS below 2.5 GiB at 1024.
- Not checked numerically: the TC coefficient derivation of 2.1/2.4 (12, 3, 3, 6 and the own-hub a/2 rule) was
  read off the code and matched against the Leonov-Shiryaev count; the code is the statistic either way.
