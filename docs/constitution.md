# Constitution

The **deal-breaker behaviors** for the DiLoCo MVP (Foundations + Phase 0–2). If
any of these is broken, revert the PR without reading further. These are success
criteria the implementing agent did **not** author and cannot narrow — they beat
the gaming problem (agent writes both code and tests). Derived from
[`DESIGN.md`](DESIGN.md) §Invariants and [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) P1–P8.

The single law above all: **earn each layer by proving the previous one.** A
change that makes a gate *pass by weakening the gate* (loosening a tolerance,
editing the baseline, changing the eval set) is a constitutional violation even
if CI is green.

---

## Correctness invariants

- **Pseudo-gradient is `master − local`, in fp32, never scaled.** No loss scaler,
  no autocast, no bf16 in the pseudo-gradient path. A sign flip or a scaled
  pseudo-grad is a deal-breaker. *(P2)*
- **The outer momentum buffer persists for the whole run.** Created once, never
  zeroed between blocks, saved and restored on checkpoint/resume. Re-instantiating
  the outer optimizer per block is forbidden. *(P3)*
- **After every outer step, master weights are identical across all workers and
  ranks** — including across the two hosts on the 5 ms link. Per-rank drift is a
  deal-breaker. *(P4)*

## Data & evaluation integrity

- **Data shards are disjoint and complete.** No token index appears in two ranks;
  the union covers the corpus. Shard *assignment* uses no RNG. Overlapping shards
  silently inflate quality and void every comparison — this is the most dangerous
  failure and is non-negotiable. *(P1)*
- **The eval harness, eval set, and tokenizer are frozen after Phase 0.** They may
  change only with an explicit `EVAL_HARNESS_VERSION` bump *and* a re-stamped
  baseline. Comparing against a baseline produced by a different harness is
  meaningless and forbidden. *(P5)*
- **Every run logs its resolved config, config hash, and seed.** An un-logged run
  is not evidence and cannot satisfy a gate. *(P5)*

## The gates that define "it works"

- **M=1 DiLoCo eval loss ≤ the data-parallel baseline** (within the stated
  tolerance) at equal token budget. M=1 is Lookahead; failing this means the outer
  step is wrong. *(P7)*
- **M=2 DiLoCo eval loss is within +1% of the baseline** at equal *total* token
  budget. The whole point of DiLoCo is matching DP quality at a fraction of the
  communication — missing quality is missing the point. *(P7)*
- **Communication is genuinely reduced:** measured bytes-per-sync ≈ DP / H within
  5%. A "DiLoCo" that doesn't cut bandwidth is not DiLoCo. *(P6)*
- **Simulated and real-collective runs agree** (within fp32 reduction-order
  tolerance) given the same seed and shards — verified both locally and across the
  two hosts. The networking must reproduce the math, not a different number. *(P4/equiv)*

## Honesty & resources

- **Failures are visible, not swallowed.** NaN loss, shard overlap, master drift,
  and momentum reset each fail loudly with a named diagnostic — never a silent
  fallback that lets a broken run report a number. *(Step 5 failure-injection)*
- **Memory does not grow with M** in the simulated path: at most master(fp32) +
  one local replica + its optimizer state resident at once. *(P8)*
- **No gate is satisfied by editing the gate.** Tolerances, baselines, and golden
  files are changed only in a PR whose explicit purpose is to change them, with
  the reasoning stated. Quietly loosening a threshold to go green is the canonical
  gaming move and is a revert-on-sight violation.
