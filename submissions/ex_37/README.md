# Pointwise final combine with the 256 shared elder basis (ex_37)

Exact code uploaded as [submission 337631](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337631)
on 2026-10-21 at 21:44 UTC, in the competition's final hours.
**Graded successfully: official adjusted score 1.8607855056e-9**, final-layer MSE **1.2105699785e-8**,
at **15.3601%** mean public FLOP budget use. All 100 networks completed with zero failures.
This is the best official score archived in this repository: 0.3358% below ex_36, 0.9310% below ex_35 and 1.2358% below ex_34.

## Code and mechanism

This is the two graded one-block changes to [ex_34](../ex_34) combined. From [ex_36](../ex_36) it takes `fmm.py`,
whose final FMM combine step runs the compiled pointwise add/subtract program instead of a dense 7-to-4 matmul
(2 instead of 13 FLOPs per output element, three more public calls per fused product). From [ex_35](../ex_35) it
takes `bank.py`, whose shared elder-basis SVD cap is `n // 4` instead of `5 * n // 16` (256 instead of 320 at width
1024). `estimator.py`, `pk4.py` and `LICENSE` are byte-identical to ex_34, ex_35 and ex_36. The package differs from
ex_36 by exactly one line (`bank.py:649`) and from ex_35 by exactly the deleted ten-line shortcut in `fmm.py`.
The two changes act on different mechanisms (bank transport rank versus FMM combine arithmetic), so their effects
were expected to add. All prediction arithmetic uses public FlopScope operations, with one value per scalar.

## Official result

The official response fetched on 2026-10-21 at 22:37 UTC reports `GRADED` with all observable audit checks passing.
Five smoke-test workers passed with zero failures. FlopScope client/server were both 0.12.3; the grader used
whestbench 0.16.1 on CPython 3.10.21.

| Measurement | Official result |
| --- | ---: |
| Mean per-network adjusted score | 1.8607855056e-9 |
| Mean final-layer MSE | 1.2105699785e-8 |
| Mean public FLOP budget use | 15.3601% |
| Maximum FLOP budget use, all 100 | 16.2005% |
| Maximum prediction wall time | 96.8227 s |
| Maximum residual time | 0.375447 s |
| Maximum setup time, five executions | 2.76851 s |

Both parent effects transferred as they had officially, not as they had on the development panel: against ex_35
(identical `bank.py`, hence the identical final-layer MSE 1.2106e-8) the pointwise combine saves 0.931% through cost
alone, and against ex_36 the 256 basis nets 0.336%, matching ex_35's own 0.308% official gain over ex_34. The
development-panel projection of 1.16% overstated the basis cut, exactly as it had for ex_35. No budget, time,
residual or combined-budget exhaustion flags were set. Direct official RSS is not exposed in the response. The
full-precision top-level score is authoritative; rounded per-network rows are not used to reconstruct it.
No current leaderboard rank is claimed; the leader at the time of upload was 337567 at 1.2952e-9.

See [the complete response](graded/platform_grading_337631_complete.json), [audit](graded/official_audit.json)
and [fetch receipt](graded/fetch_receipt.json).

## Historical local evidence and limits

On exposed development networks public 560–567, with public FlopScope 0.12.3, mean per-network adjusted score was
**1.8260e-9**, final-layer MSE **1.1775e-8** and actual public FLOPs **15.521%** of budget. On the same networks
ex_36 scored 1.8475e-9, ex_35 1.8423e-9 and ex_34 1.8649e-9: 1.164% below ex_36 (better on 4 of 8), 0.883% below
ex_35 (better on 8 of 8) and 2.086% below ex_34, the additive prediction. The final-layer MSE equals ex_35's, as
the combine change is arithmetic only. This panel had been exposed to both parents; the comparison is descriptive.
No independent fresh-network confirmation was performed.

The whestbench 0.16.2 API validation and eight shape/budget smoke cases passed. One quiet cold run on public 500
through the Linux official FlopScope client (CPython 3.10.21, idle host) passed every cap: residual proxy
**0.381544 s** (ex_36 measured 0.391348 s on the identical setup and then graded at 0.380376 s), prediction
57.816 s, setup 4.783 s, 57,620 requests (ex_36: 57,812). **The continuous lifecycle sequence was not run.**
The user approved this exact artifact with these limitations under the submission deadline.

## Files and provenance

- `estimator.py`, `bank.py`, `pk4.py`, `fmm.py`, `LICENSE`, `manifest.json`: exact extracted archive members.
- `submitted.tar.gz`: the uploaded archive, SHA-256 `9afc64478602acd060ab058a68dea6942db587e5fe31bee2e3c112a4275fbc85`.
- `platform_receipt.json`: upload-ID binding, preserved with its then-unknown official status.
- `graded/`: fetched official response, eligibility audit and fetch provenance.
- `validation/`: protocol, source freeze, local boundary (the experiment receipt), archive content receipt, the
  560–567 selection summary and the Linux cold-run result.
- `SHA256.txt`: hashes of the archived payload and evidence files.

Source records: arc4 `research/e6_combined_rank256_pointwise_20261021`. Packaging verified archive hash and exact
member bytes; it did not rerun inference.

## Readable implementation

[`estimator_refactored/`](estimator_refactored/) contains the seven-module refactor ported
from ex_36 with the ex_37 elder-basis cap, followed by an organizer-facing clarity pass.
The clarity pass preserves exact predictions and FLOPs relative to the frozen initial refactor. Its evidence and limits are
in [`estimator_refactored/VALIDATION.md`](estimator_refactored/VALIDATION.md).
The refactor is unsubmitted; the official result above belongs to the original archived package.

## Interactive local explainer

[`explainer/index.html`](explainer/index.html) is a numeric debugger for the readable estimator.
Clone or download the whole `explainer/` folder, including its large `data/` packets, then open `index.html`
directly in a recent Chromium browser; no server or hosting is required. It displays observed per-layer
estimator state and actual activation samples from public networks 500 and 501, with zero-based layers 0–15.
See [`explainer/README.md`](explainer/README.md) for capture provenance, limits, and reproduction instructions.
