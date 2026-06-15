# IMPLEMENTATION_PLAN.md — execution checklist

> Companion to [`DESIGN.md`](./DESIGN.md). That document is **what** the
> system is (DiLoCo MVP: Foundations + Phase 0–2, single-GPU compute + a two-host
> 5 ms CPU testbed). This one is **how to land it** — fine-grained vertical-slice
> steps, the tests that gate each, the loop when something fails, and binary
> acceptance.
>
> **Pre-flight:** greenfield repo (only harness wiring + docs exist). This is a
> **build-from-scratch** sequence — nothing to migrate or excise. The model is
> **not** written from scratch: it is vendored from
> [nanoGPT](https://github.com/karpathy/nanoGPT) (see Step 6).
>
> **Status: ready to execute.**

---

## The steps at a glance

**21 steps (+1 optional)** across **5 build-phases**. Each step is one PR, one
capability end-to-end, ending with a binary acceptance check and the golden-path
test (§A) green (from Step 9 on). Steps are deliberately small so a coding agent
can land one per batch under the elves harness.

**Build-phase 0 — Foundations** (substrate everything sits on) ✅ **COMPLETE**
- [x] **Step 0** — Repo + pytest + CI + metrics logger skeleton
- [x] **Step 1** — Config system (`RunCfg` tree) + stable `config_hash` (**P5**)
- [x] **Step 2** — Data prep: vendor nanoGPT, tokenize corpus → `{train,val}.bin`
- [x] **Step 3** — Deterministic rank-sharded memmap loader (**P1**)
- [x] **Step 4** — Checkpoint format: save/load model+inner+outer+step+RNG
- [x] **Step 5** — `Comm` protocol + `SimComm` stub (exact-average)

**Build-phase 1 — Phase 0 baseline** (the inner loop + the anchor) ✅ **COMPLETE**
- [x] **Step 6** — Vendor & adapt nanoGPT model → `swarm/model/gpt.py` (guardrails)
- [x] **Step 7** — AdamW inner loop: bf16, grad-clip, warmup+cosine
- [x] **Step 8** — Frozen eval harness (loss/bpb) + resume equality (**P5**)
- [x] **Step 9** — Baseline run → **stamp the baseline** + **golden path born** (§A)

**Build-phase 2 — Phase 1 DiLoCo (simulated M)** (the algorithm)
- [ ] **Step 10** — Pseudo-gradient `master−local`, fp32, unscaled (**P2**)
- [ ] **Step 11** — Nesterov outer optimizer + momentum persistence (**P3**)
- [ ] **Step 12** — `train_diloco` driver over `SimComm` → **M=1 gate** (**P2/P7**)
- [ ] **Step 13** — **M=2 gate** + bounded-memory + DiLoCo golden (**P7/P8**, §A ext.)

**Build-phase 3 — Phase 2 real collectives** (the plumbing + measurements)
- [ ] **Step 14** — `GlooComm` + process group + barrier outer step (local)
- [ ] **Step 15** — Master-agreement + **equivalence gate** sim≈gloo, local (**P4**)
- [ ] **Step 16** — Bandwidth accounting → **bytes/sync ≈ DP/H** (**P6**)
- [ ] **Step 17** — Two-host launch over 5 ms + cross-host equivalence + latency capture
- [ ] **Step 18** — Rank-aware checkpoint/resume across processes

**Build-phase 4 — Hardening** (mechanize the invariants, see the numbers)
- [ ] **Step 19** — CI guards: grep every forbidden pattern (**P1/P2/P5**)
- [ ] **Step 20** — Failure injection: overlap, NaN, momentum-reset, master-drift
- [ ] **Step 21** — Observability: throughput/bandwidth/latency table + `report.py`
- [ ] **Step 22** — (optional) fp16+scaler · 150M confirm · downstream-eval hook

Critical path: **0→1→2→3→4→5 → 6→7→8→9 → 10→11→12→13 → 14→15→16→17→18 → 19→20**.
**Step 21** can start once Step 16 lands; **Step 22** is optional.

```
Foundations          Phase0 baseline       Phase1 DiLoCo-sim
0─1─2─3─4─5 ─────────▶ 6─7─8─9 ───────────▶ 10─11─12─13 ──┐
                                                          │
Phase2 real-collectives          Hardening                │
14─15─16─17─18 ◀──────────────────────────────────────────┘
   │      └────────▶ 21 (observability, parallel)
   └─▶ 19─20 (guards+failure, after violations gone)
                     ⋯▶ 22 (optional)
```

---

## Build log — actuals (Steps 0–9 done, 2026-06-14)

Environment: **uv-managed Python 3.12 venv**, **torch 2.12.0+cu130**, verified on
an **NVIDIA RTX PRO 6000 Blackwell (sm_120, 96 GB)** single GPU. CI is CPU-only.
**43 tests passing.** Each step landed test-first, green before the next, one
commit per step.

| Step | Landed | Acceptance evidence |
|---|---|---|
| 0 | ✅ | `swarm` pkg + `metrics/logger.py` (fixed-schema TSV, fail-fast); CPU CI; 6 tests |
| 1 | ✅ | `RunCfg` tree + SHA-256 `config_hash` (process-stable); 3 configs hash distinctly |
| 2 | ✅ | `prepare.py` tiktoken→uint16 bins + `meta.json`, idempotent; nanoGPT pinned `3adf61e` |
| 3 (**P1**) | ✅ | `shard_bounds` balanced contiguous partition; disjoint+complete over ws 1–7, no RNG in assignment |
| 4 | ✅ | checkpoint round-trip + RNG restore; outer-momentum slot forward-declared |
| 5 | ✅ | `Comm` protocol + `SimComm` exact fp32 mean; `GlooComm` guarded stub |
| 6 | ✅ | nanoGPT vendored w/ attribution+SHA; `attn_impl` flag; tied weights; **51.1M** params on baseline.yaml; guardrail-#1 grep clean |
| 7 (**P2**) | ✅ | `inner_step` bf16/no-scaler; LR warmup+cosine; overfit↓, grad-clip bounds update |
| 8 (**P5**) | ✅ | frozen `estimate_loss` (reproducible, versioned); resume == straight 2N within 1e-4 |
| 9 (**§A**) | ✅ | golden path stamped `eval_loss=4.872628` (CPU/fp32, repro 1e-9); **GPU run: TinyStories 51M, loss 10.9→3.03 / bpb 1.11 in 300 steps** |

**Not yet done (deliberately):** the *headline* frozen baseline is a stamped
short run, not a converged 100M-token run — that long run is a launch, not
implementation, and is the first thing to kick off (`scripts/train_baseline.py
--config configs/baseline.yaml --device cuda`) before the M=1 gate in Step 12.

**Next:** Build-phase 2 (Steps 10–13) — the DiLoCo algorithm itself.

---

## Properties the system must preserve

Invariants every step is judged against (from `DESIGN.md` §Invariants). A step
that weakens any **cannot merge** without explicit reviewed justification.

### P1 — Disjoint, complete sharding
**Invariant:** for a given `seed`, token shards across ranks are pairwise disjoint; their union covers the corpus.
**Forbids:** RNG in shard *assignment*; a token index in two ranks; gaps.
**Allowed:** RNG in *batch order within* a shard; a deterministic uneven last shard.
**Proved by:** Step 3 — `tests/test_sharding.py::test_disjoint_and_complete`.

### P2 — Pseudo-gradient correctness
**Invariant:** `pseudo_grad = master − local`, computed/reduced in fp32, never scaled.
**Forbids:** `local − master`; pseudo-grad under autocast/bf16; a `GradScaler` in the path.
**Allowed:** bf16 inside the *inner* AdamW step; fp16+scaler confined to inner (Step 22).
**Proved by:** Step 10 (`tests/test_pseudograd.py`) + the **M=1 ≤ baseline** gate (Step 12).

### P3 — Outer momentum persistence
**Invariant:** the outer Nesterov buffer is created once, persists across every outer step, survives checkpoint/resume; never zeroed mid-run.
**Forbids:** re-instantiating the outer optimizer per block; dropping momentum from the checkpoint.
**Proved by:** Step 11 — `tests/test_outer_resume.py`.

### P4 — Master agreement
**Invariant:** after every outer step, master weights are bit-identical across all workers/ranks (incl. across the two hosts).
**Forbids:** per-rank outer updates that drift; trusting rank-0 without broadcast/verify.
**Proved by:** Step 15 — driver hash assertion + `tests/test_master_agree.py`.

### P5 — Reproducibility & frozen eval
**Invariant:** every run logs resolved config + hash + seed; eval harness/set/tokenizer frozen after Step 9; same config+seed → same eval numbers.
**Forbids:** un-logged runs; editing the eval set/tokenizer without an `EVAL_HARNESS_VERSION` bump + re-stamped baseline.
**Proved by:** Step 1 (hash stability) + Step 8 (`EVAL_HARNESS_VERSION`, eval reproducible) + Step 9 (two-seed-equal) + Step 19 guards.

### P6 — Bandwidth reduction
**Invariant:** `bytes_per_sync(DiLoCo) ≈ bytes_per_sync(DP) / H` within 5%; tied weights counted once.
**Proved by:** Step 16 — `tests/test_bandwidth.py`.

### P7 — Quality gates
**Invariant:** M=1 eval loss ≤ baseline (within tolerance); M=2 within +1% of baseline at equal total tokens.
**Proved by:** Step 12 (M=1) + Step 13 (M=2) — `scripts/gate_check.py` appends a PASS/FAIL verdict to `results.tsv`.

### P8 — Bounded memory
**Invariant:** the simulated path keeps ≤ master(fp32) + one local replica + its AdamW state resident; memory does not grow with M.
**Forbids:** M live replicas at once.
**Proved by:** Step 13 — sequential worker loop + `tests/test_mem_bounded.py`.

---

## How to execute (read once before starting)

- **Steps are vertical slices.** Each lands one capability end-to-end and leaves the trainer runnable. Never horizontal-slice.
- **Tests first.** Complete each step's "Tests first" block before implementation.
- **Acceptance is binary.** A grep returns empty, a test is green, a stamped number matches, a file is deleted. No "looks right."
- **Golden path (§A) runs after every step from Step 9 on.** Red ⇒ the most recent step caused it.
- **Rewrite from scratch when easier.** The gate is the test passing, not whether code was ported. Note the chosen path in the PR body. (The *model* is the one exception — it is vendored, not rewritten; see Step 6.)
- **No scope creep to escape a stuck step.** Use the loop in §B.

---

# Build-phase 0 — Foundations

## Step 0 — Repo + pytest + CI + metrics logger skeleton
**Goal:** package scaffold, a running test harness, CPU CI, and the append-only logger.
**Why now:** without a running harness every later test is hope-not-prove.
**Note:** infrastructure only — no property tests (their abstractions don't exist yet).
### Tests first
- [ ] `tests/test_smoke.py` — import `swarm`, assert version; proves pytest + import path.
- [ ] `tests/test_logger.py` — `metrics.logger.append_row(dict)` writes a TSV row; re-read equals; header written once.
### Implementation
- [ ] `pyproject.toml` / `requirements.txt` — pinned `torch`, `numpy`, `tiktoken`, `pyyaml`, `pytest`.
- [ ] `swarm/__init__.py` + empty submodule packages (`config`, `data`, `model`, `train`, `dist`, `eval`, `metrics`, `checkpoint`).
- [ ] `swarm/metrics/logger.py` — `results/results.tsv` writer (cols: config_hash, seed, phase, M, H, eval_loss, eval_bpb, tok_per_s, bytes_per_sync, comm_ms, compute_ms, verdict) + per-step JSONL.
- [ ] `.github/workflows/ci.yml` — `pytest -q` on CPU.
- [ ] `.gitignore` — `third_party/`, `data/**/*.bin`, `results/checkpoints/`, `__pycache__`.
### Integration check
- [ ] CI green on the scaffold PR; `pytest -q` green locally.
### Acceptance
- [ ] Smoke + logger tests pass; `python -c "import swarm"` exits 0.
**Depends on:** nothing.

## Step 1 — Config system + stable hash
**Goal:** one `RunCfg` dataclass tree is the single source of truth; `config_hash` is stable.
**Why now:** every later run logs its config+hash; sweeps depend on it (P5).
### Tests first
- [ ] `tests/test_config_hash.py` (**P5**) — same `RunCfg` → same hash; any field change → different hash; stable across process runs; YAML round-trips.
### Implementation
- [ ] `swarm/config.py` — `ModelCfg`/`OptimCfg`/`DiLoCoCfg`/`RunCfg`; `load_yaml`, `resolve(defaults+overrides)`, `config_hash` (hash of sorted resolved fields).
- [ ] `configs/{baseline,diloco-sim,diloco-dist}.yaml` — starter configs (H, M, inner/outer LR+momentum, seed, token_budget, eval_every, precision, backend).
### Acceptance
- [ ] `test_config_hash` green; a sample config resolves and hashes deterministically.
**Depends on:** Step 0.

## Step 2 — Data prep: vendor nanoGPT, tokenize corpus
**Goal:** produce `data/<name>/{train,val}.bin` (uint16, tiktoken gpt2) + `meta.json`.
**Why now:** the loader (Step 3) and every run need real tokens.
### Tests first
- [ ] `tests/test_prepare.py` — on a tiny fixture corpus, prepare writes `train.bin`/`val.bin`/`meta.json`; token count in `meta.json` matches the `.bin` length; re-run is idempotent (skips if up to date).
### Implementation
- [ ] `scripts/clone_nanogpt.sh` — clone `karpathy/nanoGPT` at a **pinned commit** into gitignored `third_party/nanoGPT/`.
- [ ] `swarm/data/prepare.py` — adapt nanoGPT's `data/*/prepare.py`: tokenize a corpus (TinyStories for smoke; FineWeb-Edu/OpenWebText sample for headline) to a flat `uint16` `.bin` + `meta.json`.
- [ ] `scripts/prepare_data.py` — CLI wrapper, config-driven, idempotent.
### Acceptance
- [ ] `python scripts/prepare_data.py --config configs/baseline.yaml` produces the three files; `test_prepare` green.
- [ ] Pinned nanoGPT commit SHA recorded in `scripts/clone_nanogpt.sh`.
**Depends on:** Step 1.

## Step 3 — Deterministic rank-sharded memmap loader
**Goal:** `get_batch` draws each rank's batches only from its disjoint shard; sharding is pure index arithmetic.
**Why now:** overlapping shards silently inflate quality and void every later comparison — the most dangerous failure (P1).
### Tests first
- [ ] `tests/test_sharding.py::test_disjoint_and_complete` (**P1**) — for `world_size ∈ {1,2,4}` × several seeds: each rank's token indices are pairwise-disjoint and union == corpus index set.
- [ ] `tests/test_loader_shapes.py` — `get_batch` returns `(B,T)` int64 `x,y` with `y` shifted by one; indices stay within the rank's shard.
### Implementation
- [ ] `swarm/data/loader.py` — `np.memmap` reader; `iter_shard_indices(split, rank, world_size)` (no RNG in assignment); `get_batch(split, rank, world_size, seed, step)` (RNG only for in-shard batch order).
### Acceptance
- [ ] Both tests green.
- [ ] Reviewer-confirmed: no `random/shuffle/randint` in the shard-*assignment* region (will become a CI guard in Step 19).
**Depends on:** Step 2.

## Step 4 — Checkpoint format
**Goal:** save/load the full run state: model + inner-optim + **outer-momentum** + step + RNG (torch+numpy).
**Why now:** resume tests (Step 8/11) and rank-aware checkpoint (Step 18) build on one format.
### Tests first
- [ ] `tests/test_checkpoint.py` — save a state dict, load it: all tensors `allclose` (fp32 exact), step + RNG restored.
### Implementation
- [ ] `swarm/checkpoint.py` — `save(path, state)` / `load(path)`; explicit slots for model, inner optim, **outer momentum**, step, RNG.
### Acceptance
- [ ] `test_checkpoint` green; the outer-momentum slot exists even before the outer optimizer does (forward-declared).
**Depends on:** Step 1.

## Step 5 — `Comm` protocol + `SimComm`
**Goal:** the load-bearing seam — one interface, with the in-process simulated backend.
**Why now:** Phase 1 and Phase 2 share one driver behind this seam; locking it now is what makes the later equivalence gate a real regression.
### Tests first
- [ ] `tests/test_comm_protocol.py` — `SimComm(world_size=4)`: `all_reduce_mean` of known tensors == exact fp32 mean; `broadcast_master` makes replicas equal; `barrier` returns.
### Implementation
- [ ] `swarm/dist/comm.py` — `Comm` protocol (`rank`, `world_size`, `all_reduce_mean`, `broadcast_master`, `barrier`); `SimComm` (in-process exact average over M replicas); `GlooComm` declared, raises `NotImplementedError` until Step 14.
### Acceptance
- [ ] `test_comm_protocol` green; `GlooComm` raises (placeholder, not silently wrong).
**Depends on:** Step 0.

---

# Build-phase 1 — Phase 0 baseline

## Step 6 — Vendor & adapt nanoGPT model
**Goal:** `swarm/model/gpt.py` is nanoGPT's `model.py`, vendored with attribution and the guardrails applied.
**Why now:** every training step needs the model; vendoring (not rewriting) removes model-bug risk and matches the DiLoCo literature.
**Note:** this is the deliberate exception to "rewrite from scratch" — we vendor, then minimally adapt.
### Tests first
- [ ] `tests/test_model_shapes.py` — forward on a toy `ModelCfg` returns `(B,T,vocab)`; loss finite; param count in the expected band; **tied `wte`↔`lm_head` verified** (same storage).
- [ ] `tests/test_model_determinism.py` — with `dropout=0` + math attention + fixed seed, two forwards on CPU are bit-identical (golden-path precondition).
### Implementation
- [ ] `scripts/clone_nanogpt.sh` already pinned (Step 2); copy `model.py` → `swarm/model/gpt.py` with a header recording the **upstream commit SHA** + MIT attribution.
- [ ] Adapt: drive sizes from `ModelCfg`; expose `dropout` (**default 0**); add an `attn_impl` flag (`"math"` for golden path, `"flash"` for GPU speed). **Do not** import anything from `modded-nanoGPT`.
### Acceptance
- [ ] Both tests green; header cites the pinned SHA.
- [ ] `grep -rni "muon\|modded" swarm/model/gpt.py` → empty (**guardrail #1**).
**Depends on:** Step 1.

## Step 7 — AdamW inner loop
**Goal:** the single-GPU inner training step — reused by Phase 0 *and* as DiLoCo's inner block.
**Why now:** it is DiLoCo's inner optimizer; get it correct in isolation.
### Tests first
- [ ] `tests/test_inner_step.py` — one step decreases loss on an overfit batch; grad-clip caps grad norm; LR follows warmup+cosine at sampled steps.
- [ ] `tests/test_no_scaler.py` (**P2 pre-guard**) — bf16 autocast present, `GradScaler` absent.
### Implementation
- [ ] `swarm/train/inner.py` — AdamW; bf16 autocast (no scaler); grad clip; warmup+cosine; step counter. Pure function over (model, optim, batch) so Step 12 can call it per inner step.
### Acceptance
- [ ] Tests green; `grep -rn "GradScaler" swarm/train/inner.py` → empty.
**Depends on:** Steps 3, 6.

## Step 8 — Frozen eval harness + resume equality
**Goal:** held-out eval loss/bpb at a fixed token cadence, reproducible; resume continues the curve.
**Why now:** no gate means anything without a frozen, reproducible eval (P5).
### Tests first
- [ ] `tests/test_eval_frozen.py` (**P5**) — eval returns the same number twice on a fixed checkpoint; `EVAL_HARNESS_VERSION` constant present and logged.
- [ ] `tests/test_resume.py` (**P5**) — train N, checkpoint, train N more == fresh 2N-step run (same seed) within tight tolerance.
### Implementation
- [ ] `swarm/eval/harness.py` — eval loss + bpb on the val shard; `EVAL_HARNESS_VERSION`; fixed cadence.
### Acceptance
- [ ] Both tests green; eval number logged to `results.tsv`.
**Depends on:** Steps 4, 7.

## Step 9 — Baseline run → stamp baseline + golden path born
**Goal:** train the small model end-to-end, record **the** data-parallel baseline, and create the golden-path spine.
**Why now:** this number anchors every DiLoCo gate; the golden path guards every later step.
### Tests first
- [ ] `tests/test_golden_path.py` (**§A**) — toy CPU run (fixed seed, tiny model, dropout=0, math attn, few steps) reaches a **stamped eval loss within 1e-3**, and a same-seed re-run reproduces it.
### Implementation
- [ ] `scripts/train_baseline.py` — config → data → model → inner loop → eval → checkpoint → `results.tsv`.
- [ ] `tests/fixtures/golden/baseline.json` — stamped golden eval loss.
### Integration check
- [ ] Baseline loss curve descends on the real small model (GPU); golden path green (CPU).
### Acceptance
- [ ] Two same-seed baseline runs produce **equal** eval bpb → that row is the **frozen baseline**; record its `config_hash`.
- [ ] Golden file committed.
**Depends on:** Step 8.

---

# Build-phase 2 — Phase 1 DiLoCo (simulated M)

## Step 10 — Pseudo-gradient
**Goal:** `pseudo_grad(master, local) = master − local`, fp32, unscaled.
**Why now:** highest-uncertainty primitive in the build; isolate and unit-test it before any orchestration.
### Tests first
- [ ] `tests/test_pseudograd.py` (**P2**) — elementwise `master − local` on a toy state-dict; dtype fp32; a sign-flip fixture fails; no autocast in the path.
### Implementation
- [ ] `swarm/train/outer.py::pseudo_grad` — fp32 difference over parameter tensors (skip non-trained buffers; tied weights handled once).
### Acceptance
- [ ] `test_pseudograd` green; `grep -rn "autocast\|GradScaler\|half()\|bfloat16" swarm/train/outer.py` → empty.
**Depends on:** Step 6.

## Step 11 — Nesterov outer optimizer + momentum persistence
**Goal:** outer SGD-Nesterov over master weights, fed the averaged pseudo-grad; momentum persists.
**Why now:** the outer step is the other half of the algorithm; momentum persistence is a named pitfall (P3).
### Tests first
- [ ] `tests/test_outer_step.py` (**P2/P3**) — momentum=0: master moves by `−outer_lr·g`; momentum>0 over two steps: buffer accumulates per Nesterov.
- [ ] `tests/test_outer_resume.py` (**P3**) — buffer non-zero after a block, byte-equal across save/load, resumed run continues the curve.
### Implementation
- [ ] `swarm/train/outer.py::OuterOptimizer` — wraps `torch.optim.SGD(master_params, lr=outer_lr, momentum, nesterov=True)`; pseudo-grad written to `.grad`; buffer saved/loaded via Step 4.
### Acceptance
- [ ] Both tests green; checkpoint includes a non-empty outer-momentum slot after one block.
**Depends on:** Steps 4, 10.

## Step 12 — `train_diloco` driver over `SimComm` → M=1 gate
**Goal:** the backend-agnostic driver runs one full DiLoCo loop with simulated workers; M=1 matches baseline.
**Why now:** M=1 (Lookahead) is the cheapest correctness signal for the whole algorithm.
### Tests first
- [ ] `tests/test_driver_round.py` — one outer round: copies master, runs H inner steps per worker, computes pseudo-grad, `all_reduce_mean`, outer step, `broadcast_master`; master changes, workers realign.
### Implementation
- [ ] `swarm/train/diloco.py::train_diloco(cfg, comm)` — sequential worker loop (P8), reusing `inner.py`; eval cadence; checkpoint.
- [ ] `scripts/train_diloco.py` — entry; `SimComm` backend.
- [ ] `scripts/gate_check.py` — runs a config, compares to the frozen baseline row, appends PASS/FAIL to `results.tsv`.
### Integration check
- [ ] §A golden path still green.
### Acceptance
- [ ] **M=1 gate**: DiLoCo M=1 eval bpb ≤ baseline within tolerance → PASS row (**P2/P7**).
**Depends on:** Steps 5, 9, 11.

## Step 13 — M=2 gate + bounded memory + DiLoCo golden
**Goal:** multi-worker DiLoCo matches DP quality; memory is M-independent; the DiLoCo golden path is stamped.
**Why now:** M=2 is the first real "DiLoCo works" result; the golden path must cover the DiLoCo path too.
### Tests first
- [ ] `tests/test_mem_bounded.py` (**P8**) — peak alloc for M=2 ≈ M=4 (sequential loop; no per-M growth).
- [ ] `tests/test_diloco_golden.py` (**§A ext.**) — toy CPU DiLoCo (M=2, small H, fixed seed) hits a stamped eval loss within 1e-3 and equals a `SimComm` re-run.
### Implementation
- [ ] `configs/diloco-sim.yaml` tuned to a real small-model M=2 run; `tests/fixtures/golden/diloco.json` stamped.
### Acceptance
- [ ] **M=2 gate**: eval bpb within +1% of baseline at equal total tokens → PASS row (**P7**).
- [ ] `test_mem_bounded` + `test_diloco_golden` green; golden committed.
**Depends on:** Step 12.

---

# Build-phase 3 — Phase 2 real collectives

## Step 14 — `GlooComm` + process group + barrier outer step (local)
**Goal:** real `torch.distributed` (Gloo) replaces `SimComm`; driver code unchanged.
**Why now:** proves the distributed plumbing locally before adding the network variable.
### Tests first
- [ ] `tests/test_gloo_allreduce.py` — spawned 2-rank gloo: `all_reduce_mean`/`broadcast_master`/`barrier` behave as the protocol (CPU tensors).
### Implementation
- [ ] `swarm/dist/comm.py::GlooComm` — `init_process_group("gloo")`, rank/world-size from env; sum-then-divide reduce; rank-0 broadcast; barrier.
- [ ] `scripts/launch_local.sh` — `torchrun --nproc_per_node=M scripts/train_diloco.py --backend gloo`.
### Acceptance
- [ ] `test_gloo_allreduce` green; local `torchrun` M=2 run completes outer rounds.
**Depends on:** Steps 5, 12.

## Step 15 — Master-agreement + equivalence gate (local)
**Goal:** master weights stay identical across ranks; the Gloo run reproduces the sim run.
**Why now:** the keystone regression linking Phase 1 ↔ Phase 2.
### Tests first
- [ ] `tests/test_master_agree.py` (**P4**) — 2-rank gloo: `hash(master)` identical across ranks after each outer step.
- [ ] `tests/test_equivalence.py` (**keystone**) — `SimComm` M=2 vs `GlooComm` M=2, same seed+shards → eval trajectories within tolerance (§D4).
### Implementation
- [ ] `swarm/train/diloco.py` — add the per-outer-step master-hash assertion (active in the dist path). No other change.
### Acceptance
- [ ] Both tests green; equivalence tolerance documented next to the test.
**Depends on:** Step 14.

## Step 16 — Bandwidth accounting
**Goal:** measure and assert the communication reduction.
**Why now:** this is the number that justifies DiLoCo's existence (P6).
### Tests first
- [ ] `tests/test_bandwidth.py` (**P6**) — counted bytes per outer step == analytic `param_count · 4 · 2 / H` (tied weights once) within 5%; and `≈ DP_bytes / H`.
### Implementation
- [ ] `swarm/train/diloco.py` — count bytes moved per outer step; log `bytes_per_sync` + `bytes_per_token`.
### Acceptance
- [ ] `test_bandwidth` green; the ratio logged to `results.tsv`.
**Depends on:** Step 14.

## Step 17 — Two-host launch over 5 ms + cross-host equivalence + latency capture
**Goal:** run one worker per remote CPU host across the real 5 ms link; same math, real latency.
**Why now:** the honest distributed test, and the Phase-4 motivating measurement for free.
### Tests first
- [ ] (manual integration; not CI) — documented runbook checklist that must pass.
### Implementation
- [ ] `scripts/launch_two_host.sh` — `MASTER_ADDR`/`MASTER_PORT`/`--node_rank` per host; both hosts use matching seed+config.
- [ ] `docs/two-host-runbook.md` — ports/firewall, seed/config matching, how to read the latency log.
- [ ] `swarm/train/diloco.py` — log per-outer-step `compute_ms` vs `comm_ms`.
### Acceptance
- [ ] Two-host run completes; **master hashes equal across hosts**; eval matches the local run within equivalence tolerance.
- [ ] Per-outer-step compute-vs-comm latency logged (no gate yet — the Phase-4 baseline).
**Depends on:** Steps 15, 16.

## Step 18 — Rank-aware checkpoint/resume across processes
**Goal:** a distributed run checkpoints consistently and resumes across processes/hosts.
**Why now:** long runs and recovery need it; closes the Phase-2 loop.
### Tests first
- [ ] `tests/test_dist_resume.py` — 2-rank gloo: checkpoint at outer step k, restart, continue; eval matches an uninterrupted run within tolerance.
### Implementation
- [ ] `swarm/checkpoint.py` — rank 0 writes master + outer state; all ranks load the same; guard against rank divergence on load.
### Acceptance
- [ ] `test_dist_resume` green; a killed-and-resumed two-host run continues the curve.
**Depends on:** Step 17.

---

# Build-phase 4 — Hardening

## Step 19 — CI guards
**Goal:** mechanize every property as a grep/test so a PR that weakens one fails CI.
**Why now:** the violations are all gone (Steps 3–18), so guards won't red-flag main.
### Implementation (each is a test that greps and fails on a hit)
- [ ] `test_guard_no_scaler_on_pseudograd.py` (**P2**) — `GradScaler`/`autocast` absent from `outer.py`.
- [ ] `test_guard_no_rng_in_shard_assignment.py` (**P1**) — no RNG in the loader's assignment region.
- [ ] `test_guard_eval_frozen.py` (**P5**) — `EVAL_HARNESS_VERSION` unchanged unless the baseline is re-stamped in the same PR.
- [ ] `test_guard_config_logged.py` (**P5**) — every entry script writes config_hash+seed.
- [ ] `test_guard_no_modded_nanogpt.py` — `muon`/`modded` absent from `swarm/model/`.
### Acceptance
- [ ] Synthetic PRs violating each guard go red; reverts go green.
**Depends on:** Step 18.

## Step 20 — Failure injection
**Goal:** prove the named failure modes fail *visibly*, not silently.
### Tests first
- [ ] `test_fail_shard_overlap.py` — an overlapping shard map makes `assert_disjoint` raise.
- [ ] `test_fail_nan_loss.py` — a NaN batch fails fast with a named diagnostic; no checkpoint past the NaN.
- [ ] `test_fail_momentum_reset.py` — a per-block outer re-instantiation is detected (buffer resets → assertion).
- [ ] `test_fail_master_drift.py` — skipping `broadcast_master` fires the hash assertion.
### Acceptance
- [ ] All four green (they assert the failure path).
**Depends on:** Step 19.

## Step 21 — Observability (parallel)
**Goal:** the declared metrics emitted every run and rendered for comparison.
### Tests first
- [ ] `test_metrics_fields.py` — a run emits all declared fields; `bytes_per_token == bytes_per_sync / tokens_per_sync`.
### Implementation
- [ ] Extend `metrics/logger.py` (throughput/bandwidth/latency); `results/README.md` documents columns.
- [ ] `scripts/report.py` — table: baseline vs M=1 vs M=2 vs gloo-two-host (eval bpb, tok/s, bytes/sync, comm-vs-compute).
### Acceptance
- [ ] `python scripts/report.py` prints the table; baseline numbers committed under `results/`.
- [ ] `results.tsv` schema matches what `auto-research` expects (one row/run, config_hash keyed).
**Depends on:** Step 16.

## Step 22 — (optional) extensions
- [ ] **fp16 + GradScaler** — scaler confined to inner; pseudo-grad still fp32/unscaled (mirror `test_pseudograd` under fp16).
- [ ] **150M confirming run** — repeat M=1/M=2 gates at 150M as one confirming row.
- [ ] **Downstream-eval hook** — `eval/downstream.py` interface stub (HellaSwag-style), wired but not built out.
**Depends on:** Steps 13–17. Defer unless requested.

---

## Definition of done (whole plan)

- [ ] **Steps 0–21 acceptance fully checked.** Step 22 done or explicitly deferred.
- [ ] **P1:** `test_sharding` green; shard-RNG guard live.
- [ ] **P2:** `test_pseudograd` green; no-scaler guard live; M=1 gate PASS.
- [ ] **P3:** `test_outer_resume` green; momentum-reset failure test fires.
- [ ] **P4:** `test_master_agree` green local + two-host; master-drift failure test fires.
- [ ] **P5:** two-seed-equal baseline; config-logged + eval-frozen guards live.
- [ ] **P6:** `test_bandwidth` within 5% of `DP/H`.
- [ ] **P7:** M=1 ≤ baseline and M=2 within +1% — PASS rows in `results.tsv`.
- [ ] **P8:** `test_mem_bounded` green.
- [ ] **Equivalence gate:** sim ≈ gloo, local and cross-host.
- [ ] §A golden path green; golden files committed.
- [ ] All CI guards live; observability table renders.

---

## §A — The integration test (golden path)

Lives at `tests/test_golden_path.py` (Step 9) + `tests/test_diloco_golden.py` (Step 13).

```
GIVEN  a fixed toy fixture (tiny model, fixed seed, dropout=0, math attention,
       few-MB tokenized corpus), on CPU/fp32
WHEN   the full pipeline runs (baseline in Step 9; DiLoCo M=2 in Step 13)
THEN   the eval loss matches a stamped golden value within 1e-3
AND    a same-seed re-run reproduces it (determinism)
AND    every capability present at this step logs real values to results.tsv
```

Golden files: `tests/fixtures/golden/`. Updated **only** at the end of Steps 9
and 13 (the steps that intentionally change the trajectory). Mid-step updates are
a red flag. Runs on CPU so it is deterministic; GPU runs use loss-*tolerance*
gates, never byte-equality (§D1).

---

## §B — Iteration loop (when something fails)

```
   Read failing assertion verbatim
        │
        ▼
   Is the test's invariant correct?
   No ◀─┴─▶ Yes
   │        │
   ▼        ▼
   Fix     Fix impl (minimum change OR rewrite the file fresh)
   test    │
   + PR    ▼
   note    Re-run failing test → run §A → green ⇒ done
```

**Stuck >30 min on the same failure:**
1. Stop coding. Expected vs. observed in the PR draft.
2. Print actual values (loss, pseudo-grad norm, master hash); then revert the print.
3. Re-read the step's Acceptance block — solving the right problem?
4. **Consider rewriting the file from scratch** against the test as spec.
5. Escalate in the PR description. Do **not** start the next step.

ML-specific: if a *quality* gate (M=1/M=2) fails but unit tests pass, the bug is
almost always one of four — pseudo-grad sign, shard overlap, momentum reset, or
under-tuned outer LR. Check those before touching the architecture.

---

## §C — Out of scope (do not build)

Stop and re-scope if a step creeps into any of these (Phase 3–7 / future):

- Quantized or streaming/subset pseudo-grad exchange; error feedback.
- Comm/compute overlap, double-buffering, eager updates.
- Async/barrier-removal, straggler/staleness handling, elastic membership, fault tolerance.
- WAN/NAT transport, libp2p/DHT, gossip averaging, DeMo compression.
- Custom CUDA/C++ kernels; the modded-nanoGPT speedrun stack.
- Multi-GPU NCCL, FSDP intra-worker sharding.
- Downstream-benchmark build-out (a hook is the most Step 22 allows).

---

## §D — Design tensions surfaced for review

**D1. Golden-path determinism vs. GPU reality.** bf16 on GPU is not bit-reproducible,
so a byte-equal golden file is impossible there. **Recommendation:** golden path on
**CPU/fp32, tiny config, dropout=0, math attention** (byte-stable, CI-able); GPU
runs gated by loss *tolerance* only.

**D2. Outer optimizer: reuse `torch.optim.SGD` vs. hand-roll.** Reusing SGD
(master weights as its params, pseudo-grad in `.grad`) gets trusted momentum for
free; the novel pseudo-grad/orchestration stays ours. **Recommendation: reuse.**
The *algorithm* is ours; the *SGD arithmetic* is not re-debugged.

**D3. Corpus + headline model size.** **Recommendation:** TinyStories for the
CI/smoke + golden loop; a FineWeb-Edu sample for the headline baseline; 50M for
all gates, 150M only as a Step-22 confirming run. Resolve before Step 2 (corpus)
and Step 6 (size).

**D4. SimComm exact-average vs. Gloo float order.** `SimComm` averages in a fixed
fp32 order; `GlooComm` sums in a backend order, so results differ in the last
bits. The equivalence gate asserts *trajectory within tolerance*, not byte-equality.
**Recommendation:** set the tolerance from observed single-step fp32 reduction-order
noise and document it next to `test_equivalence`.

**D5. nanoGPT vendoring vs. submodule.** We must modify the model (dropout default,
attention flag), so a live submodule fights us. **Recommendation: vendor `model.py`**
with the pinned upstream SHA in its header; clone the repo only transiently for its
data-prep scripts. Re-evaluate only if we need to track upstream model changes.

---

*Plan compiled 2026-06-14. 21 steps (+1 optional) across 5 build-phases. Marked
complete one step at a time, in order, §A green between steps. PRs reference the
property (P<n>) they advance. Execute under the **elves** harness with
`docs/constitution.md` populated from P1–P8 + the M=1/M=2 gates.*
