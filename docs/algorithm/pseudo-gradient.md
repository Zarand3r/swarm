# The Pseudo-Gradient — `δ = θ − θ_local`

> Algorithm deep-dive for the conceptual heart of DiLoCo. Companion to
> [`../DESIGN.md`](../DESIGN.md) and [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md);
> this is the theory behind **Step 10** (and the invariant **P2**). Math renders
> on GitHub (`$…$`).

The pseudo-gradient is the single object that links DiLoCo's two optimization
levels. A sign or precision error in it is silent and fatal, which is why the
plan gives it its own step, its own unit test, and a constitutional invariant.

---

## 1. The skeleton (notation, fixed once)

DiLoCo is two nested loops. Per **outer step** $t$, with master weights
$\theta^{(t)}$, inner learning rate $\gamma$, and $M$ workers each taking $H$
inner steps:

$$
\begin{aligned}
&\varphi_m^{(0)} = \theta^{(t)} &&\text{every worker starts from the same master}\\
&\varphi_m^{(h+1)} = \varphi_m^{(h)} - \gamma\,A\!\big(g_m^{(h)}\big),\quad g_m^{(h)}=\nabla L_m(\varphi_m^{(h)}) && h = 0\dots H-1\\
&\delta_m^{(t)} = \theta^{(t)} - \varphi_m^{(H)} &&\textbf{← the pseudo-gradient (Step 10)}\\
&\Delta^{(t)} = \tfrac1M\textstyle\sum_m \delta_m^{(t)} &&\text{average across workers (Step 12)}\\
&\theta^{(t+1)} = \mathrm{OuterOpt}\big(\theta^{(t)},\,\Delta^{(t)}\big) &&\text{Nesterov-SGD outer step (Step 11)}
\end{aligned}
$$

$A(\cdot)$ is the inner optimizer's preconditioner ($A=\mathrm{Id}$ for SGD;
AdamW's per-parameter scaling otherwise). The inner loop is ordinary training —
already built in Steps 6–9. **The only object connecting the two levels is
$\delta$.**

The bandwidth consequence is immediate: workers exchange $\delta$ (one
weight-sized tensor) **once per $H$ inner steps**, never per step. That factor of
$H$ is DiLoCo's entire reason to exist.

---

## 2. The object

$$\delta_m = \theta - \varphi_m^{(H)}$$

The **master-minus-local displacement** in weight space: a difference of two
weight *snapshots* — the master at the block's start and the worker at its end —
with one entry per parameter, the same shape as the model. It is **not** a
backprop gradient; it is computed under `no_grad` from detached tensors, *after*
the inner optimizer has already finished its $H$ steps.

---

## 3. Why a weight-difference is a legitimate gradient

**Exact statement (SGD inner).** Telescoping the inner recursion with
$A=\mathrm{Id}$:

$$\varphi_m^{(H)} = \theta^{(t)} - \gamma\sum_{h=0}^{H-1} g_m^{(h)}
\quad\Longrightarrow\quad
\delta_m^{(t)} = \gamma\sum_{h=0}^{H-1} g_m^{(h)}.$$

So $\delta$ is **exactly the $\gamma$-scaled sum of the $H$ minibatch gradients
along the block** — not an analogy, an identity. It is a multi-step, low-variance
gradient estimate that packs $H$ steps of signal into a single tensor. With
AdamW inner it becomes the *preconditioned* sum (the same statement under $A$),
which is why feeding $\delta$ to an outer optimizer that does
$\theta \leftarrow \theta - \eta\,\delta$ is genuine — if multi-step and
preconditioned — gradient descent on the master.

Equivalently: the inner loop approximately solves $\min_\varphi L_m$ inside a
trust region around $\theta$, and $\delta$ points from $\theta$ toward that local
solution — the gradient of the implicit "consensus" objective the workers
collectively descend.

---

## 4. The sign — `master − local`, fixed once (P2)

The outer optimizer *descends*: $\theta_{\text{new}} = \theta - \eta g$, with
$\delta$ playing the role of $g$. We want master to move toward the lower-loss
point $\varphi^{(H)}$, i.e. to take the step $(\varphi^{(H)}-\theta)$. One line
forces the sign:

$$\theta_{\text{new}} = \theta + \eta\,(\varphi^{(H)}-\theta)
= \theta - \eta\underbrace{(\theta-\varphi^{(H)})}_{\delta}.$$

So $\delta = \theta - \varphi^{(H)}$ is exactly what the optimizer's built-in
minus sign consumes. Sanity check: $\eta=1$, no momentum $\Rightarrow
\theta_{\text{new}}=\varphi^{(H)}$ — the outer step *recovers the local point
exactly* (this is Lookahead, §7).

**Counterfactual.** Flip to $\delta=\varphi^{(H)}-\theta$ and the update becomes
$\theta+\eta(\theta-\varphi^{(H)})$: master moves *away* from the loss-reducing
direction — it ascends, every block compounds it, loss explodes, and **no
exception is raised.** This is the most probable bug in the system. Two guards:
Step 10's test asserts the sign and includes a sign-flip fixture that must fail;
the **M=1 gate** (Step 12) is the backstop, since a sign error makes M=1
ruinously worse than the baseline.

---

## 5. Precision: fp32, and never scaled

**fp32 — because $\delta$ is a small difference of nearly-equal numbers.** After
$H$ steps the weights have barely moved, so $\theta \approx \varphi^{(H)}$ and
their difference lives in the low-order bits — the textbook setting for
*catastrophic cancellation*. Concretely: two weights near $0.41$ differing by
$\sim 5\times10^{-4}$ both round to the **same** bf16 value (bf16's spacing there
is $\approx 2\times10^{-3}$), so $\delta$ computed in bf16 is *exactly zero* —
total signal loss. bf16's 8-bit mantissa (~2–3 decimal digits) cannot resolve
it; fp32's 23-bit mantissa (~7 digits) can. In our setup the master parameters
are already fp32 (autocast casts only the *compute*, not the params), so both
snapshots are fp32 and the `.float()` in the code is a guarantee, not a
conversion. Step 10's path contains no `autocast`.

**Unscaled — the gradient-scaler hazard.** An fp16 gradient scaler multiplies the
loss by a large $S$ to avoid gradient underflow, then unwinds it *before* the
optimizer steps — entirely an inner-loop concern. $\delta$ is formed *after* that
step, from already-unwound weights, so a scaler must never touch it; if it did,
the outer step would be $\sim S\,(\approx\!65000)\times$ too large → instant
divergence. The structural defense: **default to bf16, which needs no scaler at
all** (same exponent range as fp32 ⇒ no underflow ⇒ nothing to scale). Choosing
bf16 makes this hazard *impossible to trip*.

---

## 6. What is deliberately *excluded* (the bandwidth win)

$\delta$ is **weights only**:

- **No optimizer state.** Each worker's AdamW moments (`exp_avg`,
  `exp_avg_sq`) stay local and are never sent — that is most of the saving: you
  move parameters every $H$ steps and *never* the $2\times$ optimizer state.
- **Parameters, not buffers.** Step 10 iterates `named_parameters()`, not
  `state_dict()`; non-trainable buffers (in our GPT, only the constant causal
  mask — there is no BatchNorm running state, exactly why nanoGPT fits) must not
  be differenced.
- **Tied weights once.** `wte.weight` ≡ `lm_head.weight` share storage;
  `named_parameters()` dedupes by identity, so the tied tensor yields a single
  $\delta$ — correct for the outer update *and* for bandwidth accounting (P6).
  This is why the model test asserts `data_ptr()` equality.

---

## 7. Lineage — and why it makes the M=1 gate principled

| Method | Inner | Outer | Relation to DiLoCo |
|---|---|---|---|
| Local SGD (Stich '18) | $H$ SGD steps | average ($\eta{=}1$, no momentum) | DiLoCo with a trivial outer optimizer |
| **Lookahead** (Zhang '19) | $k$ fast steps | $\varphi\!\leftarrow\!\varphi+\alpha(\varphi_k-\varphi)=\varphi-\alpha\delta$ | **exactly M=1 DiLoCo**, SGD outer |
| FedOpt (Reddi '20) | local client steps | server opt on averaged model-delta | DiLoCo = AdamW client + **Nesterov** server |
| **DiLoCo** (Douillard '23) | $H$ AdamW | **Nesterov-SGD** on $\Delta$ | matches data-parallel at $H\times$ less comms |

The term "pseudo-gradient" is FedOpt's. DiLoCo's load-bearing finding is that
**outer Nesterov momentum** is what closes the gap to fully-synchronous training
(so Step 11 is not mere averaging, and an under-tuned outer LR/momentum is the #1
cause of "DiLoCo underperforms"). And because M=1 DiLoCo *is* Lookahead — a
known-good optimizer that slightly beats vanilla — the **M=1 gate is a
correctness theorem, not an arbitrary threshold.**

---

## 8. Worked example

One scalar parameter $w$, master $\theta=1.00$, outer SGD ($\eta=0.7$, no
momentum).

- **M = 1.** Inner moves $w:1.00\to0.70$: $\delta=1.00-0.70=+0.30$;
  $\theta'=1.00-0.7(0.30)=0.79$. Master steps *toward* the local point, scaled by
  $\eta$. ($\eta{=}1\Rightarrow0.70$ exactly — Lookahead.)
- **M = 2.** Workers end at $0.70,\,0.50$ ($\delta=0.30,\,0.50$):
  $\Delta=0.40$; $\theta'=1.00-0.7(0.40)=0.72$ — toward the **consensus**; outer
  momentum then accumulates consistent $\Delta$s and accelerates.
- **Sign bug** ($\delta=-0.30$): $\theta'=1.00-0.7(-0.30)=1.21$ — master
  *climbs*, diverges, silently.

---

## 9. The code, the test, the scope

```python
# swarm/train/outer.py  (Step 10)
@torch.no_grad()
def pseudo_grad(master: dict[str, Tensor], local: dict[str, Tensor]) -> dict[str, Tensor]:
    """δ = master − local, fp32 (P2). Param tensors only; tied params appear once."""
    return {k: master[k].detach().float() - local[k].detach().float() for k in master}
```

Pinned by: a sign + dtype assertion (`master − local`, fp32), a sign-flip
fixture that must fail, and a source grep for the absence of `autocast` / scaler.

**Out of scope on purpose:** averaging (`Comm.all_reduce_mean`, Step 12), the
outer step / momentum (Step 11), and orchestration (the `train_diloco` driver,
Step 12). Isolating the difference operator gives a sign or precision bug exactly
one place to live and one test to catch it.

---

## Takeaway

$\delta = \theta - \varphi^{(H)}$ is *the $H$-step descent direction the inner
optimizer found — exactly equal (for SGD) to the $\gamma$-scaled sum of its
gradients — repackaged as a gradient for an outer momentum-SGD.* Its three rules
— sign `master − local`, fp32 against cancellation, never scaled — are the line
between DiLoCo converging and DiLoCo silently diverging.

---

## References

- Douillard et al., *DiLoCo: Distributed Low-Communication Training of Language
  Models* (2023).
- Reddi et al., *Adaptive Federated Optimization* (FedOpt) (2020).
- Zhang et al., *Lookahead Optimizer: k steps forward, 1 step back* (2019).
- Stich, *Local SGD Converges Fast and Communicates Little* (2018).
