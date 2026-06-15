# Building DiLoCo From Scratch — An Implementation Roadmap

The guiding principle for the whole build: **earn each layer of complexity by proving the previous one is correct.** DiLoCo's only reason to exist is that it matches synchronous data-parallel quality at a fraction of the communication. So the single most important asset you build is a *baseline and an eval harness* — you compare against it at every phase. Get the algorithm correct on one box, then make it distributed, then make it cheap on bandwidth, then hide latency, then make it survive failures, and only then push it onto the open internet.

Build order: **Correctness → Scale → Bandwidth → Latency → Resilience → Decentralization → Hardening.**

---

## Cross-cutting foundations (build these once, use them in every phase)

These are not a phase; they are the substrate everything else sits on.

- **Reference data-parallel baseline.** A vanilla AdamW training run you trust. Every DiLoCo experiment is judged as "loss/accuracy vs. this baseline at the same token budget."
- **Evaluation harness.** Held-out eval loss, a couple of downstream metrics (e.g. HellaSwag-style zero-shot), and a fixed eval cadence. Without this you cannot tell whether a change broke convergence.
- **Deterministic data sharding.** A way to give each worker a *disjoint* slice of data, reproducibly, keyed by worker rank and seed. Overlapping shards silently inflate quality and invalidate comparisons.
- **Reproducibility controls.** Global seed, deterministic dataloader ordering, logged config (model size, H, M, learning rates, quant level).
- **Config system.** All knobs (`H`, `M`, inner LR, outer LR + momentum, quantization level, sync schedule) in one place — you will sweep these constantly.
- **Metrics/logging from day one.** Tokens/sec, step time, and (once distributed) bytes-per-sync and bytes-per-token.

---

## Phase 0 — The Inner Loop (single GPU)

**Goal:** A clean, correct local training loop. This becomes DiLoCo's *inner optimizer*, so get it right in isolation.

**Features / deliverables**
- Model definition (start small — 50M–150M params — so iterations are fast and cheap).
- Tokenizer + streaming dataloader with sharding hooks.
- Standard AdamW training loop with gradient clipping and LR warmup/schedule.
- Mixed-precision training (bf16/fp16) with a gradient scaler.
- Checkpoint save/load (model + optimizer state + step counter + RNG state).
- Loss/throughput logging into the metrics system.

**Exit criterion:** trains a small model to a sensible loss curve; checkpoints resume bit-for-bit (or close); eval harness produces stable numbers.

**Pitfalls:** if you use a gradient scaler, remember it belongs to the *inner* optimizer only — pseudo-gradients (Phase 1) are computed in fp32 and must not be scaled.

---

## Phase 1 — The DiLoCo Algorithm (single machine, simulated workers)

**Goal:** Implement the two-level optimizer and *prove it's mathematically correct* before any networking exists. Simulate M workers in one process (a loop, or M model copies) so bugs are debuggable.

**Features / deliverables**
- **Two-level optimizer abstraction:** an outer optimizer wrapping the inner one.
- **Inner phase:** each simulated worker runs `H` AdamW steps on its own data shard, starting from a shared copy of the global ("master") weights.
- **Pseudo-gradient computation:** after H steps, `pseudo_grad = master_weights − local_weights` (fix the sign convention once and assert it). Compute in fp32.
- **Aggregation:** average pseudo-gradients across the M simulated workers.
- **Outer optimizer:** SGD with **Nesterov momentum** consuming the averaged pseudo-gradient to update master weights; persistent outer momentum buffer.
- **Master-weight management:** broadcast updated master weights back to all workers to start the next block.
- **The M=1 sanity check:** with one replica, DiLoCo should behave like an enhanced Lookahead optimizer and *match or slightly beat* the Phase-0 baseline. This is your correctness gate.

**Exit criterion:** with M simulated workers, DiLoCo reaches eval loss within a small tolerance of (ideally at or below) the data-parallel baseline for the same token budget; M=1 matches baseline.

**Pitfalls:** sign errors in the pseudo-gradient; forgetting to re-sync master weights to workers each block; outer momentum being reset between blocks; data shards accidentally overlapping.

---

## Phase 2 — Real Distributed Communication (multi-process / multi-GPU)

**Goal:** Replace the simulated workers with real processes communicating over a process group, on a single node or a tightly-coupled multi-node cluster first.

**Features / deliverables**
- **Process-group setup** (e.g. `torch.distributed`) with rank/world-size, one process per worker.
- **Collective all-reduce** of pseudo-gradients (NCCL on GPU; Gloo as a CPU/portable fallback).
- **Synchronous, barrier-based outer step:** all workers reach the sync point, all-reduce, all apply the same outer update.
- **Hybrid parallelism:** each *worker* can itself be multiple GPUs. Use FSDP/DDP *within* a worker (intra-worker, high-bandwidth) and DiLoCo *across* workers (inter-worker, low-bandwidth). This is the realistic topology.
- **Distributed checkpointing** that captures master weights + outer optimizer state consistently.
- **Bandwidth accounting:** log bytes exchanged per outer step; confirm it's ~H× less than data-parallel.

**Exit criterion:** multi-process DiLoCo reproduces Phase-1 quality, and you can measure the communication reduction directly.

**Pitfalls:** NCCL requires all peers on the same network and cannot traverse NAT — fine here, becomes the central problem in Phase 6. Mismatched master weights across ranks after a sync (use a single source of truth or all-reduce, not per-rank drift).

---

## Phase 3 — Bandwidth Reduction

**Goal:** Shrink the bytes-per-sync. This is where DiLoCo starts to tolerate genuinely slow links.

**Features / deliverables**
- **Quantized pseudo-gradient exchange:** quantize before the collective (fp16 → int8, then 4-bit), dequantize after. Track quality vs. quant level.
- **Custom quant/dequant + pack/unpack** (CUDA/C++ kernel if the PyTorch version becomes a hot spot — see the earlier note that this is the one place low-level code genuinely helps).
- **Subset / streaming synchronization (Streaming DiLoCo):** partition parameters into groups and synchronize one group per outer event on a rolling schedule, so no single step moves the whole model. Requires a partition scheme and a deterministic schedule.
- **Error feedback (optional but recommended):** accumulate quantization residual and fold it into the next sync so compression doesn't bias convergence.
- **Bandwidth dashboards:** peak vs. average bandwidth, bytes-per-token.

**Exit criterion:** 1–2 orders of magnitude less bandwidth than Phase 2 with no meaningful loss regression; peak bandwidth flattened by streaming.

**Pitfalls:** aggressive quantization without error feedback drifts convergence; subset scheduling that's not deterministic across workers causes them to sync different things.

---

## Phase 4 — Latency Hiding / Overlap

**Goal:** Stop the outer step from blocking compute. On real WAN links, *latency* (not bandwidth) is usually what idles your GPUs.

**Features / deliverables**
- **Comm/compute overlap:** launch the pseudo-gradient sync on a background CUDA stream / dedicated communication thread while the inner loop keeps running.
- **Double-buffered weights:** one buffer trains while the other is being synchronized.
- **Eager updates:** apply an *estimate* of the outer update immediately and reconcile when the real synchronized update lands, so the outer step fully overlaps the inner phase.
- **Pipelining of streamed subsets** (combine with Phase 3): subset N+1 computes while subset N is in flight.

**Exit criterion:** GPU utilization stays high (low idle time) even when inter-worker latency is artificially inflated (test with `tc`/`netem`).

**Pitfalls:** staleness introduced by overlap can hurt convergence if uncontrolled — validate against the baseline after enabling overlap, not just throughput.

---

## Phase 5 — Asynchrony & Fault Tolerance

**Goal:** Survive the real world, where nodes lag, crash, and rejoin. This is the difference between a research demo and a usable system.

**Features / deliverables**
- **Barrier removal / asynchronous outer step:** workers sync on their own cadence instead of a global barrier.
- **Straggler tolerance:** proceed without waiting for the slowest worker; define a timeout/quorum policy.
- **Staleness handling:** reintegrate late pseudo-gradients from slow/returning nodes, optionally down-weighting by staleness so old updates don't destabilize the master.
- **Elastic membership:** nodes can join or leave mid-run; world size is dynamic (an `ElasticDeviceMesh`-style abstraction).
- **Live checkpoint recovery:** a joining/recovering node pulls current master weights and resumes without a full restart.
- **Failure detection:** heartbeats, timeouts, and a policy for declaring a node dead and redistributing its shard.

**Exit criterion:** you can kill, slow, and add workers mid-training and the run continues to converge — measured against the baseline, not just "it didn't crash."

**Pitfalls:** unbounded staleness silently degrades quality; membership changes that don't re-shard data leave coverage gaps; recovery that loads stale master weights.

---

## Phase 6 — Internet-Scale / Decentralized

**Goal:** Train across the open internet — heterogeneous hardware, NAT, untrusted or volunteer peers.

**Features / deliverables**
- **WAN-capable transport:** NCCL won't traverse NAT, so move to a peer-to-peer layer — a distributed hash table over libp2p (Hivemind-style) or a coordinator service — for metadata and synchronization.
- **Peer discovery + NAT traversal.**
- **Gossip / pairwise averaging option (NoLoCo-style):** replace the global all-reduce with averaging against a random peer after each local block, removing the global collective entirely — the biggest latency win at scale.
- **Advanced compression stack:** integrate DeMo-style momentum compression (DCT + top-k + error feedback) as an option for the most bandwidth-starved links.
- **Bandwidth-adaptive scheduling:** sync cadence and subset size that adapt to each peer's measured link quality.
- **Trust & integrity (if open participation):** authentication, and verification that contributed updates aren't malicious/garbage.

**Exit criterion:** a model trains to baseline-comparable quality across geographically separated, NAT'd, heterogeneous nodes at high utilization.

**Pitfalls:** the control plane (discovery, membership, gossip) becomes the hard part, not the math; gossip topologies need enough mixing or replicas drift apart.

---

## Phase 7 — Production Hardening & Observability

**Goal:** Make it operable, debuggable, and trustworthy over long multi-week runs.

**Features / deliverables**
- **Goodput metrics:** effective compute uptime, staleness distribution, per-worker throughput, bytes-per-sync over time.
- **Network emulation in CI:** `tc`/`netem` profiles for latency, jitter, and packet loss so you test the bad-link regime deterministically.
- **Chaos testing:** automated node kills, slowdowns, and rejoins as part of the test suite.
- **Full resumability:** any run resumes from checkpoint across total cluster loss.
- **Hyperparameter sweep tooling** for the DiLoCo-specific knobs (below).
- **Dashboards & alerting** for divergence, stalled workers, and bandwidth anomalies.

**Exit criterion:** a multi-day run survives injected failures and link degradation while staying on its loss trajrparameters to expose and sweep

| Knob | What it controls | Notes |
|---|---|---|
| `H` (inner steps) | Sync frequency | Original DiLoCo ≈ 500; the scaling-laws work used much smaller H. The central bandwidth-vs-quality dial. |
| `M` (replicas) | Number of workers | M=2 was found to beat data-parallel at 4B/10B; M=1 is the Lookahead sanity check. |
| Inner LR | AdamW learning rate | Tuned roughly like a normal training LR. |
| Outer LR + momentum | Nesterov outer optimizer | Distinct from inner LR; under-tuning this is a common cause of "DiLoCo underperforms." |
| Quantization level | Bytes per sync | int8 → 4-bit → 2-bit; pair aggressive levels with error feedback. |
| Subset/stream schedule | Peak bandwidth | Number of parameter groups and rotation order. |
| Staleness bound | Async stability | Max tolerated lag before a worker's update is discarded or down-weighted. |

---

## Suggested tech stack

- **Core:** PyTorch + `torch.distributed`; FSDP2 for intra-worker sharding.
- **Collectives:** NCCL (cluster), Gloo (portable/CPU), custom int8/4-bit kernels for the quantized path.
- **Decentralized transport:** Hivemind (libp2p DHT) or a custom coordinator for the WAN phase.
- **Custom kernels:** CUDA/C++ only for the quantize/pack and transform/sparsify hot spots, exposed to the Python loop — not a full-loop rewrite.
- **Testing:** `tc`/`netem` for network emulation; a chaos harness for membership churn.

---

## Sequencing / dependency summary

```
Foundations (baseline + eval + sharding)  ── required by every phase
        │
Phase 0  Inner loop (1 GPU)
        │
Phase 1  DiLoCo algorithm (simulated M)         ◀── correctness gate (M=1 == baseline)
        │
Phase 2  Real collectives (multi-GPU, FSDP×DiLoCo)
        │
Phase 3  Bandwidth (quant + streaming)  ──┐
        │                                  ├─ Phases 3 & 4 interleave well
Phase 4  Latency hiding (overlap, eager) ─┘
        │
Phase 5  Async + fault tolerance
        │
Phase 6  Internet-scale (WAN transport, gossip, DeMo compression)
        │
Phase 7  Hardening + observability
```

A realistic MVP that demonstrates the core idea is **Foundations + Phase 0–2**: that already gives you a working, communication-reduced trainer you can measure. Everything from Phase 3 on is about making it *cheap, fast, and survivable* enough to matter in practice.
