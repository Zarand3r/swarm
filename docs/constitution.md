# Constitution

> Copy to `docs/constitution.md` in your repo and replace the examples below
> with your own. The **elves** Judge reads this every batch and verifies the app
> still keeps these promises. Delete this quote block once you've filled it in.

This file lists the **deal-breaker behaviors** — the things that, if broken,
would make you revert the whole PR without reading further. It exists to beat
the *gaming problem*: when the agent writes both the code and the tests, it can
satisfy the letter of a test while missing the point. The constitution gives the
Judge success criteria the agent did **not** author and cannot narrow.

## How to write a good intention

Each entry should be:
- **Specific enough to verify** — "A failed payment never results in a fulfilled
  order." Not "the payment system works correctly."
- **Abstract enough to survive refactoring** — "A user can reset their password
  via email." Not "the `resetPassword` function in `auth.service.ts` calls
  SendGrid."
- **Stated as a behavior, not an implementation** — "No endpoint exposes another
  user's private data." Not "we use row-level security in Postgres."

**Do not** put implementation details, UI layouts, exact test values,
experimental features, or nice-to-haves here. Only deal-breakers.

---

## Flows
User / data / auth / payment flows that must hold end to end. A Mermaid diagram
makes a flow unambiguous in a way prose can't.

- _Example:_ A user can sign up, verify their email, and log in. An unverified
  account can never access a protected route.
- _Example:_ Checkout: a charge is created **before** an order is marked paid;
  a failed charge never marks the order paid.

## Business logic
Pricing, eligibility, approval, notification, and formula rules that must stay
correct.

- _Example:_ Discounts never stack beyond 50% total off list price.
- _Example:_ A refund returns exactly the captured amount, never more.

## Invariants
Things that must always be true regardless of what else changes.

- _Example:_ An unauthenticated request can never reach an admin route.
- _Example:_ A soft-deleted record is never returned by any public API.
- _Example:_ No log line or API response ever contains a raw password or token.
