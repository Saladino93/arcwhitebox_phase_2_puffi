# Pointwise final combine with the 256 shared elder basis (submission 337631)

Exact code uploaded as [submission 337631](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337631)
on 2026-10-21 at 21:44 UTC, in the competition's final hours.
**Graded successfully: official adjusted score 1.8607855056e-9**, final-layer MSE **1.2105699785e-8**,
at **15.3601%** mean public FLOP budget use. All 100 networks completed with zero failures.
This is the best official score archived in this repository: 0.3358% below submission 337577, 0.9310% below submission 337563 and 1.2358% below submission 337479.

## Code and mechanism

This is the two graded one-block changes to [submission 337479](../337479) combined. From [submission 337577](../337577) it takes `fmm.py`,
whose final FMM combine step runs the compiled pointwise add/subtract program instead of a dense 7-to-4 matmul
(2 instead of 13 FLOPs per output element, three more public calls per fused product). From [submission 337563](../337563) it
takes `bank.py`, whose shared elder-basis SVD cap is `n // 4` instead of `5 * n // 16` (256 instead of 320 at width
1024). `estimator.py`, `pk4.py` and `LICENSE` are byte-identical to submission 337479, submission 337563 and submission 337577. The package differs from
submission 337577 by exactly one line (`bank.py:649`) and from submission 337563 by exactly the deleted ten-line shortcut in `fmm.py`.
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

Both parent effects transferred as they had officially, not as they had on the development panel: against submission 337563
(identical `bank.py`, hence the identical final-layer MSE 1.2106e-8) the pointwise combine saves 0.931% through cost
alone, and against submission 337577 the 256 basis nets 0.336%, matching submission 337563's own 0.308% official gain over submission 337479. The
development-panel projection of 1.16% overstated the basis cut, exactly as it had for submission 337563. No budget, time,
residual or combined-budget exhaustion flags were set. Direct official RSS is not exposed in the response. The
full-precision top-level score is authoritative; rounded per-network rows are not used to reconstruct it.
No current leaderboard rank is claimed; the leader at the time of upload was 337567 at 1.2952e-9.

See the [official audit](graded/official_audit.json). The full per-network API response stays in the team's private
research archive.

## Historical local evidence and limits

On exposed development networks public 560–567, with public FlopScope 0.12.3, mean per-network adjusted score was
**1.8260e-9**, final-layer MSE **1.1775e-8** and actual public FLOPs **15.521%** of budget. On the same networks
submission 337577 scored 1.8475e-9, submission 337563 1.8423e-9 and submission 337479 1.8649e-9: 1.164% below submission 337577 (better on 4 of 8), 0.883% below
submission 337563 (better on 8 of 8) and 2.086% below submission 337479, the additive prediction. The final-layer MSE equals submission 337563's, as
the combine change is arithmetic only. This panel had been exposed to both parents; the comparison is descriptive.
No independent fresh-network confirmation was performed.

The whestbench 0.16.2 API validation and eight shape/budget smoke cases passed. One quiet cold run on public 500
through the Linux official FlopScope client (CPython 3.10.21, idle host) passed every cap: residual proxy
**0.381544 s** (submission 337577 measured 0.391348 s on the identical setup and then graded at 0.380376 s), prediction
57.816 s, setup 4.783 s, 57,620 requests (submission 337577: 57,812). **The continuous lifecycle sequence was not run.**
The user approved this exact artifact with these limitations under the submission deadline.

## Files and provenance

- `estimator.py`, `bank.py`, `pk4.py`, `fmm.py`, `LICENSE`, `manifest.json`: exact extracted archive members. The
  uploaded archive has SHA-256 `9afc64478602acd060ab058a68dea6942db587e5fe31bee2e3c112a4275fbc85`, recorded in
  `platform_receipt.json`.
- `platform_receipt.json`: upload-ID binding, preserved with its then-unknown official status.
- [graded/official_audit.json](graded/official_audit.json): the official eligibility audit (aggregates, maxima and checks).
- `SHA256.txt`: hashes of every file shipped in this folder.

Packaging verified the archive hash and exact member bytes; it did not rerun inference. The protocol, source freeze,
local boundary, archive content receipt, 560–567 selection summary and Linux cold-run result stay in the team's
private research archive.

## Readable implementation

A trick-free numpy rewrite of this estimator family, with the organizers' K=3 estimator for comparison, is in
[`../../distilled_submission/`](../../distilled_submission/).
