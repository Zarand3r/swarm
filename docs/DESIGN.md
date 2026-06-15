# DiLoCo From Scratch — System Design (MVP: Foundations + Phase 0–2)

> Companion to [`ROADMAP.md`](../ROADMAP.md). The roadmap defines *what* and *why*
> across 8 phases; this document defines *how* for the MVP the roadmap names —
> **Foundations + Phase 0–2**, the working communication-reduced trainer you can
> measure. Phases 3–7 are designed at architecture altitude only (see
> [Deferred Complexity](#deferred-complexity)). The agent-executable checklist
> lives in [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md).

Target environment: **single GPU** for compute, plus a **two-host CPU testbed —
two separate computers on a network with ~5 ms inter-host latency** — for the
protocol. This shapes one decision throughout: Phase 2 uses a *real*
`torch.distributed` process group on the **Gloo** backend (CPU/TCP collectives),
and it runs in two configurations: (a) multiple processes on the single GPU box
for fast local iteration, and (b) **one worker process per remote CPU host across
the real 5 ms link** for the honest test. Same code, only `MASTER_ADDR`/rank
launch differs. That second config makes the ranks, all-reduce, barrier outer
step, distributed checkpoint, *and the bytes-and-latency-per-sync numbers* real
rather than simulated — and gives an early, real latency signal that motivates
Phase 4 (latency hiding) before we build it. Multi-GPU NCCL and the FSDP×DiLoCo
hybrid remain deferred to when such hardware exists; the abstractions below are
built so that swap is a backend flag, not a rewrite.

---

## Goal

Build a correct, measurable DiLoCo trainer in three provable layers: a clean
single-GPU AdamW inner loop (Phase 0), the two-level DiLoCo optimizer validated
against a data-parallel baseline (Phase 1), and the same algorithm running over
real collectives across processes (Phase 2). Every layer is judged *against a
frozen baseline and eval harness*, never against itself.

## Success Metrics

- **M=1 correctness gate:** DiLoCo with one replica reaches eval loss ≤ the
  Phase-0 AdamW baseline at equal token budget (it is Lookahead; it should match
  or slightly beat). Tolerance: within +0.5% eval bpb, target ≤ baseline.
- **M>1 quality gate:** DiLoCo with M=2 (simulated and, in Phase 2, real)
  reaches eval loss within +1% of the data-parallel baseline at equal *total*
  token budget.
- **Communication reduction (Phase 2):** measured bytes-per-sync is
  `≈ DP_bytes_per_sync / H` (one outer exchange per H inner steps), logged and
  asserted within 5% of the analytic value.
- **Real-network outer step (Phase 2, two-host):** an outer step completes across
  the real 5 ms link, and per-outer-step wall-clock (compute vs. comm/blocking
  time) is logged. This number is the baseline the Phase-4 overlap work must
  later drive toward zero idle — captured now, for free, as the motivating
  measurement rather than a synthetic `netem` one.
- **Equivalence gate (Phase 1 ↔ Phase 2):** real-collectives run reproduces the
  simulated run's eval-loss trajectory within tolerance given identical seed and
  shard assignment.
- **Reproducibility:** a run resumes from checkpoint and continues the same loss
  curve; two runs with the same config+seed produce the same eval numbers.

## Constraints

- Single GPU; model **50–150M params** so an inner block is seconds, not minutes.
- bf16 mixed precision is the default (no gradient scaler needed); fp16+scaler is
  an optional path that must keep the scaler on the *inner* optimizer only.
- Pseudo-gradients are computed and reduced in **fp32**, never scaled.
- The eval harness, eval set, and tokenizer are **frozen** once Phase 0 lands —
  changing any of them invalidates all cross-phase comparisons.
- PyTorch + `torch.distributed`. No third-party DiLoCo/optimizer libraries — the
  two-level optimizer is the thing we are building.

## Non-Goals (this MVP)

Quantization, streaming/subset sync, comm/compute overlap, async/barrier-removal,
fault tolerance, elastic membership, WAN/NAT transport, gossip averaging, custom
CUDA kernels, multi-node, FSDP intra-worker sharding, downstream-benchmark
sweeps. All deferred — see [Deferred Complexity](#deferred-complexity).

## Requirement Audit

| Requirement (from roadmap) | Needed for MVP? | Rationale |
|---|---|---|
| Reference DP baseline + frozen eval | **Yes — first** | Every gate is "vs baseline." Nothing is verifiable without it. |
| Deterministic disjoint sharding | **Yes** | Overlapping shards silently inflate quality and void every comparison. |
| Config + reproducibility + metrics | **Yes** | You will sweep H/M/LRs constantly; un-logged runs are unusable. |
| Single-GPU AdamW inner loop + ckpt | **Yes (P0)** | It *is* DiLoCo's inner optimizer. |
| Two-level optimizer, pseudo-grad, Nesterov outer | **Yes (P1)** | The algorithm itself. |
| M=1 Lookahead sanity check | **Yes (P1)** | Cheapest correctness signal; isolates the outer step from data-parallel effects. |
| Real process group + all-reduce + barrier outer step | **Yes (P2)** | Proves the distributed plumbing; gives the bandwidth measurement. |
| FSDP×DiLoCo hybrid, NCCL | **Deferred** | Needs multi-GPU; reframed as Gloo multi-process here. |
| HellaSwag-style downstream eval | **Deferred (optional hook)** | Eval-loss/bpb is sufficient as the gate; wire the hook, don't build the benchmark. |
| Quant / streaming / overlap / async / WAN | **Deferred** | Phases 3–7. Each earns its place only after the previous gate is green. |

## Existing System Understanding

Greenfield repo. Present: skills/harness wiring (`.claude/`, `CLAUDE.md`,
`docs/constitution.md`), the roadmap. No source, tests, or data pipeline yet.
The build starts from an empty `swarm/` Python package.

## Architecture Decomposition

```
configs/                      YAML configs (baseline, diloco-sim, diloco-dist)
swarm/
  config.py                   one dataclass tree; load/merge/validate/hash
  data/
    prepare.py                corpus -> train.bin / val.bin (uint16, tiktoken gpt2)
    loader.py                 np.memmap loader + deterministic rank-sharded batches
  model/gpt.py                GPT vendored from nanoGPT @ pinned commit (dropout=0 + math-attn for golden path)
  train/
    inner.py                  AdamW inner loop (Phase 0 baseline + DiLoCo inner)
    outer.py                  pseudo-grad + Nesterov-SGD outer optimizer (Phase 1)
    diloco.py                 two-level driver: orchestrates inner blocks + outer step
  dist/comm.py                backend abstraction: simulated | gloo all-reduce (Phase 2)
  eval/harness.py             frozen held-out eval-loss/bpb at fixed token cadence
  metrics/logger.py           append-only results.tsv + per-step JSONL
  checkpoint.py               save/load model+optim+outer+step+RNG (rank-aware)
scripts/
  prepare_data.py
  train_baseline.py           Phase 0 entry
  train_diloco.py             Phase 1 (sim) & Phase 2 (torchrun) entry — same script
tests/                        unit + property + golden-path integration
results/                      results.tsv (append-only), checkpoints/, logs/
```

**Model choice (decided): nanoGPT, vendored.** Use Karpathy's
[nanoGPT](https://github.com/karpathy/nanoGPT) `model.py` (decoder-only GPT, MIT)
as the model. Bring it in by **vendoring the single `model.py` into
`swarm/model/gpt.py`** with an attribution header recording the upstream commit
SHA (we must modify it for the guardrails below, so a pinned vendor beats a live
submodule); the upstream repo is also cloned transiently into a gitignored
`third_party/nanoGPT/` at that same commit only to reuse its dataset-prep
scripts. Why nanoGPT fits: same decoder-only regime as the DiLoCo literature (so
M=1/M=2 numbers are comparable), **no BatchNorm** — only LayerNorm affine params,
which removes the running-statistics headache that plagues averaged training,
clean inspectable state-dict for pseudo-grads, smooth 50M→150M scaling via
`n_layer/n_head/n_embd`, and AdamW+cosine out of the box.

Guardrails on the vendored model (enforced in the plan):
1. **AdamW inner only — do *not* use `modded-nanoGPT`.** Its Muon/speedrun tricks
   change the *inner* optimizer and confound the one variable under study (the
   outer/pseudo-grad step).
2. **`dropout=0` + the math (non-flash) attention path for the deterministic CPU
   golden path.** Dropout RNG and fused/flash kernels are nondeterministic; real
   GPU runs keep dropout and use tolerance gates (see §D1/§D4).
3. **Treat nanoGPT's absolute throughput as non-representative.** At 50–150M an
   inner block is cheap enough that the 5 ms link skews wall-clock vs. real scale.
   Trust it for correctness, quality-vs-baseline, and the bytes-per-sync *ratio* —
   not for headline tokens/sec or MFU.
4. **Weight tying:** nanoGPT ties `wte`↔`lm_head`; count tied params **once** in
   bandwidth accounting (P6) or bytes-per-sync is overstated.

**Tokenizer/data choice (decided):** pre-tokenize a small standard corpus
(recommend OpenWebText sample or FineWeb-Edu sample; TinyStories for the fastest
smoke loop) to a flat `uint16` `.bin` with the **tiktoken gpt2** vocab, à la
nanoGPT (reuse its `data/*/prepare.py` as the starting point). A memory-mapped
flat token array makes deterministic disjoint sharding trivial (index arithmetic
by rank), avoids building a tokenizer, and is fully reproducible. This is the
single most leveraged simplification in the design.

## Core Entities and Interfaces

```python
# config.py — one source of truth; every run logs its resolved config + hash
@dataclass class ModelCfg:  n_layer; n_head; n_embd; block_size; vocab_size; dropout
@dataclass class OptimCfg:  inner_lr; betas; weight_decay; grad_clip; warmup; schedule
@dataclass class DiLoCoCfg: H; M; outer_lr; outer_momentum; nesterov=True
@dataclass class RunCfg:    model; optim; diloco; seed; token_budget; eval_every;
                            data_dir; out_dir; precision="bf16"; backend="sim"

# data/loader.py — determinism is the contract
def get_batch(split, rank, world_size, seed, step) -> (x, y)   # disjoint by rank
def assert_disjoint(world_size, seed) -> None                  # property-test hook

# train/outer.py — the algorithm's heart; sign convention asserted once
def pseudo_grad(master: StateDict, local: StateDict) -> StateDict   # master - local, fp32
class OuterOptimizer:        # Nesterov SGD over pseudo-grads; persistent momentum buffer
    def step(self, avg_pseudo_grad) -> None                    # mutates master in place

# dist/comm.py — the single seam between Phase 1 (sim) and Phase 2 (real)
class Comm(Protocol):
    rank: int; world_size: int
    def all_reduce_mean(self, tensors) -> None                 # in place
    def broadcast_master(self, master) -> None
    def barrier(self) -> None
class SimComm(Comm): ...     # M copies in one process, exact-average loop
class GlooComm(Comm): ...    # torch.distributed, gloo backend, real collectives

# train/diloco.py — backend-agnostic driver
def train_diloco(cfg: RunCfg, comm: Comm) -> None
```

The `Comm` protocol is the load-bearing abstraction: Phase 1 and Phase 2 run the
*same* `train_diloco` driver, swapping only `SimComm`↔`GlooComm`. That is what
makes the Phase-1↔Phase-2 equivalence gate a real regression test rather than a
hope, and what makes the future NCCL/gossip backends a third implementation of
one interface.

## Data Flow / Control Flow

**Phase 0 (baseline):** `get_batch → forward(bf16) → loss → backward → clip →
AdamW.step → log`; every `eval_every` tokens run frozen eval, append to TSV;
checkpoint periodically.

**Phase 1/2 (DiLoCo), one outer round:**
```
for worker i in M (sim: sequential loop; dist: this rank only):
    local_i ← copy(master)
    for h in 1..H:  inner AdamW step on shard(i)         # bf16 compute
    pg_i ← master - local_i                              # fp32
comm.all_reduce_mean(pg)                                 # sim: average; gloo: collective
outer.step(pg_avg)                                       # Nesterov SGD updates master
comm.broadcast_master(master)                            # all workers realign
assert hash(master) equal across workers                 # invariant check (dist)
log bytes_per_sync, tokens/sec, eval at cadence
```

## State Machines / Lifecycles

- **Master weights:** `init → (inner-detached copies) → outer-updated → broadcast
  → next block`. Exactly one authoritative master per outer round; in Phase 2 it
  is kept identical across ranks by reducing/broadcasting, never by per-rank
  drift.
- **Outer momentum buffer:** `init once → persists across every outer step →
  saved in checkpoint → restored on resume`. Never reset between blocks (a named
  roadmap pitfall).
- **Run:** `config-resolve → data-check → train-loop (inner×H ⇄ outer) →
  eval-cadence → checkpoint → resume-safe exit`.

## Architecture Options

| Decision | Options | Choice & why |
|---|---|---|
| Simulated workers (P1) | (a) sequential loop over M copies; (b) M model replicas resident; (c) threads | **(a) sequential** — deterministic, low memory (one extra fp32 master + one local at a time), trivially debuggable. (b) wastes GPU memory; (c) adds nondeterminism for no gain at M small. |
| Phase-2 backend | (a) Gloo multi-process (local + two-host); (b) NCCL multi-process sharing one GPU; (c) skip, stay simulated | **(a) Gloo** — real collectives/ranks/barrier, and it runs *cross-host over the real 5 ms link* (one rank per remote CPU) as well as locally; CPU-side pseudo-grad reduce is fine (fp32, once per H steps). (b) contends one GPU, misleads on perf, and can't span hosts; (c) leaves the distributed path untested. |
| Mixed precision | (a) bf16; (b) fp16+GradScaler | **(a) bf16 default** — no scaler, sidesteps the "don't scale pseudo-grads" pitfall entirely. fp16 kept as an option with the scaler confined to inner. |
| Data pipeline | (a) pre-tokenized memmap `.bin`; (b) streaming HF dataset; (c) on-the-fly tokenize | **(a) memmap** — deterministic index sharding, reproducible, no tokenizer build, fastest iteration. |
| Outer optimizer | (a) hand-written Nesterov SGD over state-dict; (b) reuse `torch.optim.SGD` on master params | **(b) wrap `torch.optim.SGD(nesterov=True)`** with master weights as its params and pseudo-grad written into `.grad` — reuses a trusted, tested momentum implementation; we own only the pseudo-grad and the orchestration. |

## Tradeoff Analysis

The central tension is **fidelity vs. iteration speed**. We resolve it by making
the *math* high-fidelity (real fp32 pseudo-grads, real Nesterov, real collectives
via Gloo) while keeping the *scale* tiny (50–150M, small H during smoke tests,
M=2). This buys correctness signal in seconds. The deferred items (quant,
overlap, async) are exactly the ones whose value only shows up at scale/ WAN, so
deleting them now costs no signal. Reusing `torch.optim.SGD` for the outer step
trades a little "from scratch" purity for not re-debugging momentum — worth it;
the novel part (pseudo-gradient, two-level orchestration) is still ours.

## Risks and Bottlenecks

| Risk (likelihood) | Cheapest experiment | Success criterion | Fallback |
|---|---|---|---|
| **Pseudo-grad sign error** (high) | M=1 run vs baseline | M=1 ≤ baseline loss | Assert `Δ=master−local` once; unit-test direction on a 1-param toy |
| **Shard overlap inflates quality** (high) | `assert_disjoint` property test | no token index in two ranks; union = dataset | Index-arithmetic sharding (no RNG in assignment) |
| **Outer momentum reset between blocks** (med) | resume + continue test | momentum buffer non-zero, persists in ckpt | Buffer owned by the SGD instance, saved/loaded explicitly |
| **Baseline not trustworthy** (med) | lock baseline first; two-seed variance | seed variance < gate tolerance | Don't start P1 until baseline curve is stable |
| **P1≠P2 divergence** (med) | same-seed sim vs gloo run | eval-loss trajectories within tolerance | Rank-keyed deterministic sharding; identical driver code |
| **Single-GPU memory for copies** (low) | M=2, 150M dry run | fits with master(fp32)+local+AdamW state | Keep master on CPU; shrink model |
| **bf16/fp16 scaler pitfall** (low) | default bf16 | no scaler in pseudo-grad path | If fp16 needed, scaler stays inner-only, unscale before pseudo-grad |

**Highest-uncertainty item attacked first:** the pseudo-gradient + outer step
correctness, via the M=1 Lookahead gate on top of a locked baseline. That single
experiment validates the algorithm's core before any distribution exists.

## Invariants

1. Data shards are disjoint across ranks for a given `(seed)`, and their union
   covers the corpus (→ `assert_disjoint` property test).
2. `pseudo_grad = master − local`, computed in fp32, never multiplied by a loss
   scaler (→ assertion + unit test on sign).
3. The outer momentum buffer is created once and persists across every outer
   step and across checkpoint/resume; it is never zeroed mid-run (→ resume test).
4. After every outer step, master weights are bit-identical across all
   workers/ranks (→ hash-equality assertion in the dist path).
5. Every run logs its fully-resolved config + a config hash + seed; the eval
   harness/set/tokenizer are frozen after Phase 0 (→ hash recorded in results.tsv).
6. `bytes_per_sync(DiLoCo) ≈ bytes_per_sync(DP) / H` within 5% (→ bandwidth test).
7. M=1 DiLoCo eval loss ≤ baseline within tolerance; M=2 within +1% (→ gates).
8. Memory is bounded: at most master(fp32) + one local replica + its AdamW state
   resident at once in the simulated path (→ no per-M memory growth).

## Vertical Slice Strategy

The first end-to-end artifact is the **thinnest DiLoCo round that touches the
whole spine**: tiny model (e.g. n_layer=4, n_embd=256), `H=10`, `M=2`, ~a few
outer rounds, on a few MB of tokens, producing an eval-loss number logged to
`results.tsv` next to a baseline run of the *same token budget*. It exercises
config → sharding → inner AdamW → pseudo-grad → Nesterov outer → broadcast →
eval → metrics in one shot. Only once that spine is green do we scale H, M, and
model size and chase the actual quality gates. This slice also *is* the
golden-path integration test.

## Milestone Roadmap

Each milestone ends green (gate passes) before the next begins. The
[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) breaks these into **21 steps
(+1 optional) across 5 build-phases** — one capability per step; the milestone
DoD is the phase's exit gate.

- **M0 — Foundations** (plan Steps 0–5). Logger + config/hash, data prep
  (vendoring nanoGPT + tokenize), deterministic sharded loader, checkpoint format,
  `Comm` protocol + `SimComm`. *DoD:* `test_disjoint_and_complete` passes; config
  round-trips and hashes; `SimComm` exact-averages.
- **M1 — Phase 0 baseline** (Steps 6–9). Vendored nanoGPT model (guardrails),
  AdamW inner loop (bf16), frozen eval harness, baseline run. *DoD:* small model
  trains to a sensible loss curve; resume continues it; baseline eval-bpb stable
  across two seeds (frozen as *the* baseline); golden path born.
- **M2 — Phase 1 DiLoCo, simulated** (Steps 10–13). `pseudo_grad`, Nesterov
  `OuterOptimizer`, `train_diloco` over `SimComm`. *DoD:* M=1 gate green
  (≤ baseline); M=2 gate green (within +1%); momentum persists across resume;
  memory M-independent.
- **M3 — Phase 2 real collectives** (Steps 14–18). `GlooComm` + barrier outer
  step, master-agreement, bandwidth accounting, **two-host run across the real
  5 ms link**, rank-aware resume. *DoD:* equivalence gate green (gloo ≈ sim) local
  *and* cross-host; bandwidth within 5% of `DP/H`; master hashes equal across
  hosts; per-outer-step compute-vs-comm latency logged.
- **M4 — Hardening** (Steps 19–21, +22 optional). CI guards, failure injection,
  observability. *DoD:* every property has a live grep/test guard; named failures
  fail visibly; the comparison table renders.

## Verification Strategy

- **Unit:** pseudo-grad sign on a toy tensor; outer-step direction; checkpoint
  save/load equality (state-dict allclose); config hash stability.
- **Property:** `assert_disjoint` over all rank pairs for several seeds and world
  sizes; shard union == corpus.
- **Golden-path integration:** the vertical slice run reaches a target loss and
  logs to TSV; runs in CI at toy scale (CPU-able) so the spine is guarded.
- **Equivalence regression:** `SimComm` M-run vs `GlooComm` M-run, same seed →
  eval trajectories within tolerance. This is the keystone test linking P1↔P2.
  Run it both single-box and **two-host over the 5 ms link** — same numbers
  cross-host proves the collective/barrier logic is network-correct, not just
  shared-memory-correct.
- **Real-network latency capture (M3):** on the two-host run, log per-outer-step
  compute time vs. comm/blocking time. No gate yet — this is the honest baseline
  Phase 4 must improve, recorded into `results.tsv`.
- **Gate checks (scripted, not CI):** M=1 ≤ baseline; M=2 within +1%; these run
  at real (small-model) scale and append verdicts to `results.tsv`.
- **Benchmark/accounting:** tokens/sec, bytes-per-sync, bytes-per-token logged
  every run; bandwidth invariant asserted in M3.
- **Reproducibility:** two same-seed runs produce equal eval numbers; resume test
  continues the curve.

The results log is **append-only TSV**, one row per run with config hash, seed,
phase, M, H, eval-loss/bpb, tokens/sec, bytes-per-sync — the substrate the
`auto-research` skill will later optimize over when sweeping the DiLoCo knobs.

## Deferred Complexity

Designed-for, not built. The seams that keep these cheap to add later:

- **Phase 3 (bandwidth):** quantize/dequantize wraps `pseudo_grad` before
  `comm.all_reduce_mean`; streaming/subset sync is a partition+schedule over the
  state-dict keys the driver already iterates. Error-feedback is a residual
  buffer alongside the master. *Seam: the pseudo-grad tensor list + `Comm`.*
- **Phase 4 (overlap):** the outer round is already a distinct step; overlap
  means launching `all_reduce` on a side stream/thread and double-buffering
  master. The 5 ms two-host testbed already exposes real blocking latency in M3,
  so Phase 4's payoff is measurable on day one — no `netem` needed to see it.
  *Seam: `Comm` becomes async (return a handle); driver gets a begin/wait split.*
- **Phase 5 (async/FT):** replace barrier with cadence + quorum/timeout; staleness
  down-weighting multiplies pseudo-grads before the outer step; elastic membership
  varies `world_size`. *Seam: `Comm` + outer `step` weighting.*
- **Phase 6 (WAN/decentralized):** a third `Comm` impl over libp2p/coordinator;
  gossip = pairwise average instead of global reduce. *Seam: `Comm`.*
- **Phase 7 (hardening):** goodput metrics extend the logger; netem/chaos extend
  the test harness. *Seam: metrics + tests.*
- **FSDP×DiLoCo / NCCL:** a `Comm` backend swap plus FSDP *inside* a worker;
  unblocked by multi-GPU hardware. *Seam: `Comm` backend flag + intra-worker
  wrapper.*

Every deferred phase resolves to one of two seams — the **`Comm` protocol** or
the **pseudo-grad/outer-step boundary**. That is the design's main bet: keep
those two interfaces clean and the roadmap's later phases are additions, not
rewrites.

## Recommended Next Step

Generate the agent-executable [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)
(checklist-first, vertical-slice steps, binary acceptance gates, property tests
bound to the invariants above), then run M0→M3 under the **elves** harness with
`docs/constitution.md` populated from the invariants/gates here.

## Open Questions / Decision Points

1. **Corpus pick** for the baseline — OpenWebText sample vs FineWeb-Edu sample vs
   TinyStories (smoke). Recommend TinyStories for the CI/smoke loop and a
   FineWeb-Edu sample for the headline baseline. *Pick before M0.*
2. **Model size** for the *headline* gates — 50M (fastest signal) vs 150M
   (closer to interesting). Recommend 50M for all gates in this MVP; 150M only
   as a confirming run. *Pick before M1.*
3. **Token budget** for a gate run — must be large enough that M=1 vs baseline is
   distinguishable above seed variance. Recommend setting it empirically from the
   two-seed baseline variance in M1. *Decide during M1.*
4. **fp16 path** — build it now or defer? Recommend **defer**; bf16 only until a
   reason appears.
