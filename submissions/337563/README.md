# Shared elder basis rank 256 (submission 337563)

Exact code uploaded as [submission 337563](https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/337563) on 2026-10-21.
The saved official response reports **GRADED**, adjusted score **1.8782730189e-9**, final-layer MSE **1.2106318454e-8**,
and 15.5037% mean public FLOP budget use. All 100 networks completed with zero failures.
This is approximately 0.308% below submission 337479's official score.

## Code and mechanism

This is [submission 337479](../337479) with one line changed in `bank.py`: the shared elder-basis SVD cap is `n // 4`
instead of `5 * n // 16` (256 instead of 320 at width 1024). The smaller shared basis reduces transport and
contraction work while omitting additional singular directions. `estimator.py`, `pk4.py`, `fmm.py` and `LICENSE`
are byte-identical to submission 337479. The dense final FMM combine and setup-prepared views remain intact.
All prediction arithmetic uses public FlopScope operations, with one value per scalar.

## Evidence and limits

On exposed development networks public 560–567, mean per-network adjusted score was 1.8422624077e-9,
final-layer MSE 1.1770649937e-8, and actual public FLOPs 15.6653% of budget (FlopScope 0.12.3).
The matched submission 337479 score was 1.8648954074e-9, a 1.214% development gain.

The independent 24-network confirmation **failed its registered superiority gate**: estimated adjusted score
2.0542979085e-9 versus 2.0547264984e-9, only 0.0209% lower, with the one-sided 95% upper difference positive.
The user subsequently approved this exact uncertain artifact for submission. The historical confirmation failure
is preserved in the team's private research archive; the later official result does not rewrite it.

Local SDK/API, eight smoke cases, cold and continuous lifecycle checks passed. The saved official audit reports
maximum prediction wall 101.8783 s, residual 0.371910 s, setup 2.81521 s, and maximum FLOPs 16.3506% of budget.
The official environment used FlopScope client/server 0.12.3. Direct official RSS was not reported.

## Files and provenance

- `estimator.py`, `bank.py`, `pk4.py`, `fmm.py`, `LICENSE`, `manifest.json`: exact extracted archive members. The
  uploaded archive has SHA-256 `7bf83c55643c39497c4bfdd075ea55ff8ce479418a5a838fb0627ed80cca6818`, recorded in
  `platform_receipt.json`.
- `platform_receipt.json`: upload-ID binding.
- [graded/official_audit.json](graded/official_audit.json): the saved official audit (aggregates, maxima and checks).
- `SHA256.txt`: hashes of every file shipped in this folder.

The original protocol, source freeze, local and confirmation boundaries and SDK validation receipts stay in the
team's private research archive.

Source records: arc4 `research/e6_w_shared_rank256_20261021`. Historical receipts describe their status at creation;
the saved official response above supersedes earlier pending/no-submission statuses. Packaging verified the archive
hash, exact member bytes and Python syntax; it did not rerun inference or refresh official grading.
