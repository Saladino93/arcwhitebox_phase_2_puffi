# Shared-basis Argon recurrence on the qualified 337428 backend (submission 337479)

This is the exact artifact uploaded as [submission 337479](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337479)
on 2026-10-21. **Graded successfully: official score 1.8840699578e-9**, final-layer MSE 1.1741755888e-8 at 16.0351 % mean
public FLOP budget use. All 100 networks completed with zero failures. It is the repository's best graded submission:

- **4.99 % lower** than the previous team best, submission 337428 (1.9829771968e-9). 337428 came from the arc4 experiment
  `e6_history_lite_final_combine`, which is not archived as an example here.
- **6.24 % lower** than [submission 336552](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336552) (336552, 2.0093582727e-9).

The leaderboard leader when grading finished, submission 337381 (1.3453767043e-9), remains 40 % lower.

The statistical source is candidate F0 from the separate submission 337479 campaign on branch `claude/gifted-einstein-44qo48`
(`research/submission 337479/candidate_F0`, commit `00e5e306`), which was never submitted. That branch is not merged into main. This
directory is the graded descendant of F0.

## What it is

submission 337479 combines two exact parents and adds no fitted parameters, ranks, thresholds or catalogue changes:

- **Statistics** (`bank.py`, `pk4.py` and the `estimator.py` numerics): F0 plus three safety changes. F0 is the Argon
  chassis of [submission 336299](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/336299) plus two lanes from the F0 campaign.
- **Matrix backend** (`fmm.py`): byte-identical to the officially graded 337428 package.

### Statistical changes relative to 337428

1. **Shared elder basis** (`SB`, `SB_DENSE_REPAIR`). Elder cubic-history sources are rebuilt as `Y @ G` from one right basis
   `G` shared by every elder, instead of one basis per source. At the first switch, `G` is the entrant's own top-d right
   singular subspace; later entrants are expressed in it with a dense repair. One transport of `G` per layer serves every elder.
2. **Lane L1, slice row repair** (`SLICE_ROWS`, `SLICE_UROW = "fill"`, `HF_LITE_ROWS`). The delayed (2,1) slice drops its
   saturated repeated-index rows. The next core diagonal (urow) reads those rows with O(1) weight, so dropping them alone
   roughly doubles MSE. L1 refills them with the minimum-norm rank-2 matrix that reproduces urow's two functionals, computed
   exactly from the factored bank. On the row plan, the history fold runs without its distinct-completion term.
3. **Lane L2, exact switch restructuring** (`SB_E1`, `SB_E2`). The switch reuses `U_d S_d` from the SVD it already computes,
   and the elder rebuild reads the needed basis columns directly from the transport product. This is exact algebra.
4. **Readout and core.** The terminal readout adds the `z3^2 / 72` Edgeworth term. The off-diagonal edge block of the core
   trace is omitted from layer 4 on.

### Safety changes to F0

These were registered and made before any score measurement:

- **L1 contractions.** L1's three-operand contractions are evaluated as an explicit factored sequence of binary public
  operations. An empty or non-finite fill falls back to the original full-row path.
- **Shared-basis QR and solve.** Before installing any factor state, the code checks that `R` is square, that the factors
  are finite and that every pivot is nonzero. On failure, it falls back to the whole-Gaussian path through
  `DegenerateMoments`.
- **Server-side linear-algebra errors.** LinAlgError reasons forwarded by the official client as
  `internal server error: LinAlgError: ...` are recognized like local ones. Budget, time, memory and unknown errors still
  propagate.

### Backend and lifecycle

The backend comes from the officially qualified submission 336299 static-setup lineage behind 337428:

- Shape-only Strassen views and arenas are built once, in `Estimator.setup`, by `fmm.prepare_static_views(width)`.
- `fmm.configure()` invalidates data-dependent plans and operand transforms at each prediction.
- The final combine stage is one ordinary matmul with a coefficient matrix.
- Products are routed by call-priced depth.

Relative to F0 plus the safety changes, `estimator.py` adds `setup()` and removes the two outer `fmm.cold_reset()` calls.
Nothing else in it changes.

**Why the backend swap was needed.** F0's backend reset at prediction entry and exit, so every prediction rebuilt its views.
Running the same statistics on F0's own backend (arc4 `e6_shared_basis_replacement`) gave these results:

- Every functional check, the SDK API check and eight smoke cases passed.
- The quiet cold public-500 residual was 0.4457 s, which fails the 0.4 s cap.
- That run made 19,022 more client requests during prediction, and 15,861 fewer during setup, than 337428.

submission 337479 keeps every statistical term and swaps only the backend and lifecycle; its cold residual is 0.3814 s. Products are
unchanged in exact arithmetic, but FMM routing, rounding and billed FLOPs differ from F0's.

All prediction-time numerical work uses public FlopScope 0.12.3 operations, one value per scalar, without packing.

## Official grade

Graded 2026-10-21 06:07–06:39 UTC.

| Official measurement | 337479 (submission 337479) | 337428 |
| --- | ---: | ---: |
| Public-50 mean per-network adjusted score | **1.8840699578e-9** | 1.9829771968e-9 |
| Public-50 mean final-layer MSE | 1.1741755888e-8 | 1.1554468262e-8 |
| Public-50 mean FLOP budget use | 16.0351 % | 17.1543 % |
| Maximum FLOP budget use, all 100 | 16.9072 % | |
| Residual median / maximum | 0.3436 / 0.3736 s | — / 0.3822 s |
| Prediction wall median / maximum | 91.00 / 98.02 s | — / 104.36 s |
| Maximum controller round trip | 98.41 s | |
| Maximum setup (5 executions) | 2.703 s | 2.724 s |

All 100 networks completed with zero failures, and no budget, time or residual flags were set. The smoke test passed with 5
worker passes and 0 failures. The grader reports whestbench 0.16.1 and FlopScope client/server 0.12.3 on Python 3.10.21.

Accuracy is slightly worse than 337428 (MSE ×1.016), while the bill fell by 1.12 budget points. The score gain comes from the
lower bill. The top-level official score is authoritative. Per-network rows in the API are rounded and do not reproduce it.

Records:

- [graded/platform_grading_337479_complete.json](graded/platform_grading_337479_complete.json): the full final API response.
- [graded/terminal_receipt.json](graded/terminal_receipt.json): the terminal receipt.
- [graded/official_terminal_review.json](graded/official_terminal_review.json): an 86-check independent audit of the grade.
- [platform_receipt.json](platform_receipt.json): the upload receipt.

## How it was chosen

Scores are the mean of each network's `final_layer_MSE * max(0.1, F / 2^41)`, using actual public FlopScope 0.12.3 metering.
No training, sweep or tuning was done. Each run below compares submission 337479 against the exact 337428 package.

- **Development panel, public 560–567.** These networks had been used before, so this is exposed development evidence. Mean
  adjusted score was 1.8648954074e-9 against 1.9181084393e-9 (−2.77 %), with final MSE 1.1514e-8 at 16.21 % actual FLOPs.
  The first four networks (560–563) gave 1.7621e-9 against 1.8108e-9.
- **Local qualification.** Linux host, FlopScope client/server 0.12.3 and a handwritten SDK contract:
  - Cold public 500: startup plus setup 4.652 s, prediction 57.88 s, residual 0.3814 s.
  - One worker running 501, 502 and 501: residuals 0.3969, 0.3936 and 0.3873 s (minimum margin 3.1 ms), with repeated
    predictions byte-identical.
  - The genuine SDK 0.16.2 CLI smoke and eight off-shape cases passed, and so did the 0.16.2 package validator.
  - The first cold run and the first worker-sequence run were interrupted by host guards: GPU activity from desktop apps,
    and a monitoring-service process. Each was repeated once under a registered environmental fix, with the candidate
    unchanged.
- **Independent confirmation.** Twenty-four fresh oracle networks were reserved before selection, each with 16 reference
  blocks of 2^20 samples. Mean adjusted score was 2.2069146978e-9 against 2.3143372587e-9 (−4.64 %), and the one-sided 95 %
  upper bound on the paired difference was −6.81e-11. These are finite-sample reference estimates, not official scores.
- **Transfer.** The gain grew from −2.77 % on the exposed panel to −4.64 % on the fresh panel and −4.99 % officially.

The exact frozen archive was uploaded once, under the user's standing submission authorization.

## Files

- `estimator.py`, `bank.py`, `pk4.py`, `fmm.py`, `LICENSE`: the five submitted payloads. They are byte-identical to the
  members of `submitted.tar.gz`.
- `submitted.tar.gz`: the uploaded archive, 93,678 bytes, sha256
  `a7d47a1e0314211e6d315907b665008e7be432f8aa5c57d8130d34cce8dbaba1`. It includes the SDK [manifest](manifest.json)
  (`min_version` 3.10). [SHA256.txt](SHA256.txt) lists every hash.
- `validation/`: compact copies of the source receipts.
  - [protocol.json](validation/protocol.json): the registered hypothesis and gates.
  - [experiment_receipt.json](validation/experiment_receipt.json): the experiment receipt.
  - [local_qualification_boundary.json](validation/local_qualification_boundary.json): the local qualification boundary.
  - [fresh24_confirmation_boundary.json](validation/fresh24_confirmation_boundary.json): the fresh-24 confirmation.
  - [smoke_audit.json](validation/smoke_audit.json): the smoke audit.
  - [submission_validation.json](validation/submission_validation.json): the package validator result.

The full ledgers, client audits and reviews are in the arc4 repo under
`research/e6_shared_basis_qualified_backend_20261021`.
