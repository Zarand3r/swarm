"""swarm — DiLoCo from scratch.

A distributed low-communication training system built in provable layers:
a correct single-GPU inner loop, the two-level DiLoCo optimizer validated
against a data-parallel baseline, then the same algorithm over real collectives.
See docs/DESIGN.md and docs/IMPLEMENTATION_PLAN.md.
"""

__version__ = "0.1.0"
