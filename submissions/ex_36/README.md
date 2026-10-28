# Pointwise final FMM combine (ex_36)

Exact code uploaded as [submission 337577](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337577) on 2026-10-21.
**Graded successfully: official adjusted score 1.8670546715e-9**, final-layer MSE **1.1740799639e-8**,
at **15.8916%** mean public FLOP budget use. All 100 networks completed with zero failures.
This is the best official score archived in this repository: 0.5973% below ex_35 and 0.9031% below ex_34.

## Code and mechanism

This branches directly from [ex_34](../ex_34), not ex_35: it retains the shared elder-basis cap of 320.
Only `fmm.py` changes. Removing the final-step dense 7-to-4 matmul shortcut in `_out_batch` restores the
existing compiled pointwise add/subtract combine. The exact-real linear map is unchanged, but floating-point
association, routing and billed work can differ. The final combine's ordinary arithmetic drops from 13 to 2
FLOPs per output at the cost of three additional public arithmetic calls per FMM product.
`estimator.py`, `bank.py`, `pk4.py` and `LICENSE` are byte-identical to ex_34. Setup-prepared views remain intact.
All prediction arithmetic uses public FlopScope operations, with one value per scalar.

## Official result

The official response fetched on 2026-10-21 reports `GRADED` with all observable audit checks passing.
Five smoke-test workers passed with zero failures. FlopScope client/server were both 0.12.3;
the grader used whestbench 0.16.1.

| Measurement | Official result |
| --- | ---: |
| Mean per-network adjusted score | 1.8670546715e-9 |
| Mean final-layer MSE | 1.1740799639e-8 |
| Mean public FLOP budget use | 15.8916% |
| Maximum FLOP budget use, all 100 | 16.7578% |
| Maximum prediction wall time | 99.4356 s |
| Maximum residual time | 0.380376 s |
| Maximum setup time, five executions | 2.76026 s |

No budget, time, residual or combined-budget exhaustion flags were set. Direct official RSS is not exposed
in the response; successful grading reflects the enforced limits without supplying an independent RSS measurement.
The full-precision top-level score is authoritative; rounded per-network score rows are not used to reconstruct it.
This establishes the official result for this artifact, without retroactively completing the interrupted local run
or the unperformed independent confirmation. No current leaderboard rank is claimed.

See [the complete response](graded/platform_grading_337577_complete.json), [audit](graded/official_audit.json)
and [fetch receipt](graded/fetch_receipt.json).

## Historical local evidence and limits

On exposed development networks public 560–567, mean per-network adjusted score was **1.8475057890e-9**,
final-layer MSE **1.1509746118e-8**, and actual public FLOPs **16.0664%** of budget (FlopScope 0.12.3).
The matched ex_34 adjusted score was 1.8648954074e-9, a 0.9325% development gain. This is not an official score
or independent confirmation; the reserved fresh confirmation was not performed.

Source controls, SDK/API and eight smoke cases passed. The isolated cold case passed all 939 checks:
startup plus setup 4.86225 s, prediction 59.0181 s, residual 0.391348 s.
The continuous sequence completed two calls with residuals 0.387504 s and 0.395171 s; the third was interrupted
by a host guard after lightweight metadata SSH activity. **The local continuous lifecycle qualification remained incomplete.**
The user approved this exact artifact with these limitations. No complete lifecycle pass is asserted.

## Files and provenance

- `estimator.py`, `bank.py`, `pk4.py`, `fmm.py`, `LICENSE`, `manifest.json`: exact extracted archive members.
- `submitted.tar.gz`: original uploaded archive, SHA-256 `9a01aa8d03f3382ff0cc0e351af7b3054e68af772922c2b39eec0522bfb92aad`.
- `platform_receipt.json`: original upload-ID binding, preserved with its then-unknown official status.
- `graded/`: fetched official response, eligibility audit and fetch provenance.
- `validation/`: original protocol, source freeze, final local boundary and archive content audit.
- `SHA256.txt`: hashes of the archived payload and evidence files.

Source records: arc4 `research/e6_w_pointwise_final_20261021`. Pre-upload protocol gates and audit statements
are historical records; the final boundary and upload receipt document the actual incomplete qualification and
specific approval. The later official response supersedes the upload receipt’s unknown official status.
Packaging verified archive hash, exact member bytes and Python syntax. This update fetched and audited the official
result without changing submitted code or rerunning inference.
