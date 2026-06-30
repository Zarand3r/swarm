# swarm

<!-- Add project-specific context (what this repo is, build/test commands, conventions) above the routing block as the codebase grows. -->

## Skills — use these automatically

The skills below come from the `eng-skills` plugin (auto-installed via
`.claude/settings.json`). They are invoked namespaced (e.g. `/eng-skills:elves`),
but Claude also auto-invokes them by description. **Before and during any coding
work, load the skill(s) whose trigger matches the task** and follow their
guidance. `karpathy-guidelines` applies to essentially all coding; the others
are routed to by task type. When in doubt, start with
`principal-production-engineer` — the single entry point that routes to the rest.

| Skill | Load it when… |
|---|---|
| **karpathy-guidelines** | Always, for any writing/reviewing/refactoring of code. Avoid overcomplication, make surgical changes, surface assumptions, define verifiable success criteria. |
| **principal-production-engineer** | Implementing, reviewing, refactoring, or hardening production code in any language. Single entry point — enforces simple design, dense data, explicit ownership, visible failure, minimal abstraction, honest verification, pipeline discipline. Routes to the skills below. |
| **strategic-engineering-planner** | *Before* implementation when work is architecturally significant, ambiguous, multi-file, distributed, performance-sensitive, concurrency-heavy, or likely to need multiple passes. Produces a written roadmap first. Skip for trivial fixes and obvious CRUD. |
| **spec-driven-development** | *Before* coding on complex or ambiguous work, to prevent drift. Turns a goal into an executable spec — EARS requirements, binary acceptance criteria, scope/invariants, requirement→test traceability — that code and tests are derived from. Sits between the planner and implementation-plan. |
| **implementation-plan** | *After* the design is locked, *before* code. Turns a design doc into a checklist-first `IMPLEMENTATION_PLAN.md` with vertical-slice steps and binary acceptance gates. |
| **test-driven-verification** | Implementing or hardening any nontrivial change. Derive tests from acceptance criteria first, loop red→green→refactor, capture re-runnable evidence (unit/property tests, Playwright/tmux artifacts), and gate merges on binary criteria. |
| **cpp-systems-internals** | Writing or reviewing C++ where hardware behavior, codegen cost, ownership vocabulary, API style, or kernel paging matters (lambdas, templates, cache lines, vtables, smart pointers/spans/arenas, `mmap`/`madvise`, AoS/SoA). Load only the relevant topic file. |
| **data-oriented-design** | Performance is a first-class requirement — hot paths, real-time/embedded, low-latency, SIMD/vectorization, parsers, allocators, codecs, game/engine, HPC, or any "as fast as possible" task. Loads the model-the-computation doctrine, the optimization order, cache/SoA layout, branchless + SIMD + bit-packing idioms, and a measure-first verification protocol. Routes to `cpp-systems-internals` for C++ mechanism depth. |
| **python-style** | Writing or reviewing Python where style/design matters — flattening logical branches (guard clauses, dispatch/`match` over `if`/`elif`), enums/`StrEnum` over magic strings, fail-fast validation (narrow `except`, no silent fallbacks), no optional imports (`try`/`except ImportError`) or redundancy, choosing abstractions (ABC vs `Protocol`, composition over inheritance). Load only the relevant topic file. |
| **auto-research** | Iteratively optimizing a measurable outcome unattended/overnight — loss, latency (p50/p95/p99), throughput, MFU, memory/binary/model size, compile time. Enforces a fixed eval harness, append-only results log, keep-on-improvement / reset-on-regression. |
| **elves** | Executing a *development plan* unattended/overnight — user says "run overnight," "implement this plan," "keep going without me," "I'll be back in the morning." Breaks the plan into sprint-sized batches, implements with tests + PR-based review, and keeps durable memory (survival guide, learnings, execution log) for compaction recovery. Requires `git` + `gh`. |

**How to apply:** for a non-trivial task, the default flow is
`strategic-engineering-planner` (plan) → `implementation-plan` (checklist) →
`principal-production-engineer` (implement, routing into `cpp-systems-internals`
as needed), with `karpathy-guidelines` governing throughout. For
unattended/overnight runs pick by goal: **`auto-research`** when success is *one
number on a fixed harness* (optimize a metric), **`elves`** when success is *a
development plan with test/PR gates* (build features across batches). Read a
skill's `SKILL.md` before acting on its domain.

## Autonomous harness (elves)

The overnight development harness is the `elves` skill. Before the first run the
repo needs: a pushable git remote + authenticated `gh`, a verification gate
(test/lint/build command that exits 0 on a clean checkout), and — recommended —
the ungameable promises in [`docs/constitution.md`](docs/constitution.md) that
the elves Judge enforces every batch. Setup checklist:
[claude-skills `templates/harness-setup.md`](https://github.com/Zarand3r/claude-skills/blob/main/templates/harness-setup.md).
