# Tool policy — what a benchmark agent may call, and what may leak

This is the authoritative statement of the agent's permitted tool surface and
the leakage properties of each tool. It exists because a leak went unnoticed
for a long time (see *Incident 1*), and because three other documents happily
described the leaking tool as part of normal agent operation.

**Scope.** Two different things are specified here:

- what the **server exposes** (this repo), and
- what a **benchmark run may offer to the model** (the agent loop in
`oduerr/autostan-private`).

They are not the same set. The server is a general tool; the benchmark is a
measurement instrument with stricter requirements.

---



## What counts as leakage

A benchmark measures how well an agent can *discover* a model from training
data. Three things would invalidate that measurement if they reached the model:


| Leak class                             | Example                                | Why it invalidates                                                                                     |
| -------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **L1 — held-out labels**               | test-set `y` values in a tool response | the agent could fit them directly                                                                      |
| **L2 — cross-session results**         | another run's NLPD or model notes      | turns discovery into copying                                                                           |
| **L3 — the honest-evaluation channel** | `shadow_nlpd`                          | converts the shadow set into a second feedback set, silently destroying the selection-bias measurement |


L1 is the obvious one and was designed against from the start. **L2 and L3 are
the ones that actually bit us.**

---



## Tool table

What each tool does is documented in [docs/REFERENCE.md](docs/REFERENCE.md).
This table states only the leakage properties and whether a benchmark agent
may be offered the tool — do not restate tool behaviour here.


| Tool                      | Returns                                                                                               | Leak risk                                                           | Offered in benchmark runs?       |
| ------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- | -------------------------------- |
| `get_capabilities`        | tool list, server config                                                                              | none                                                                | ✅ yes                            |
| `list_datasets`           | dataset names and tiers                                                                               | none                                                                | ✅ yes                            |
| `get_data_summary`        | per-column stats of **train** only, plus `dataset.md`                                                 | none — test columns are never read                                  | ✅ yes                            |
| `check_model`             | compile status, error text                                                                            | none                                                                | ✅ yes                            |
| `sample`                  | diagnostics + `run_id`; draws stay on disk                                                            | none                                                                | ✅ yes                            |
| `fit_and_evaluate`        | **NLPD on the held-out set**, diagnostics, `run_id`                                                   | **by design** — this is the feedback channel the benchmark measures | ✅ yes                            |
| `get_upload_instructions` | HTTP upload URL and fields                                                                            | none                                                                | ✅ yes                            |
| `get_run_history`         | **every logged iteration for the dataset, across all sessions and agents** — NLPDs, notes, rationales | **L2**                                                              | ❌ **no — must never be offered** |




## HTTP sidecar endpoints

The sidecar (`--upload-port`) is part of the surface too — bulk data moves
through it in both directions, deliberately outside LLM context.


| Endpoint               | Direction | Carries                              | Leak risk                                                                                                                                                               |
| ---------------------- | --------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `POST /dataset/{name}` | in        | train CSV + dataset.md (uploads)     | none — test data is never accepted; the operator places it manually                                                                                                     |
| `GET /train/{dataset}` | out       | `train.csv` or `dataset.md` **only** | none — train data is the agent's input by definition; the filename is whitelisted, dataset names are traversal-checked, and any path containing `protected/` is refused |


`GET /train` exists so clients can run EDA code **locally** on the raw train
set: the loop (or a coding agent) downloads the CSV to disk, computes
aggregates there, and only the aggregates enter context.  This moves the L1
boundary from "sandbox around agent code" to "the server refuses to serve
`protected/`" — much easier to guarantee and to test.

**Caveat — same-machine runs.** The guarantee above holds only when the
machine executing agent-written EDA code has no filesystem access to
`datasets/*/protected/`.  If the loop runs on the server host, local code can
read the protected files directly off disk, bypassing HTTP entirely.  Either
run code-enabled loops on a different machine, or deny the loop's user read
permission on `protected/` (`chmod 700`, owned by the server user).

**Comparability.** Runs where the agent had raw-train EDA (via this endpoint
or any code-execution tool) measure a different task than summary-stats-only
runs.  The loop must record the capability in its run log (`eda_code_enabled`)
so the two populations are never pooled silently.

### Notes on the two sensitive entries

`fit_and_evaluate` **returns the test NLPD, and that is intended.** The
benchmark's premise is an agent iterating against held-out feedback; removing
it would measure something else. What it must *not* return is `shadow_nlpd`
(L3) — that value is computed server-side and written only to the run log.
There is a regression test asserting no shadow-named key and no shadow value
appears in the tool response.

`get_run_history` **is not safe to expose to a benchmark agent** and cannot be
made safe by filtering, because the histories of different runs on the same
dataset are exactly what must stay separated. It remains available for
interactive/human use, where cross-run context is a feature.

---



## Enforcement — two layers, because one failed

1. **Offer-side.** The loop removes `get_run_history` from the tool list sent
  to the model.
2. **Call-side.** The loop rejects any tool name outside the offered list
  instead of forwarding it to the server.

Layer 2 exists because layer 1 alone was insufficient: the model asked for a
tool it had never been offered, and the loop forwarded it anyway.

---



## Incident 1 — cross-session history leak (2026-07-30)

`get_run_history` was excluded from the *offered* tool list, but the loop
forwarded **any** tool name the model produced. A `deepseek-v4-flash` run —
steered by a stale system-prompt line that told it to "reference previous
iterations (from `get_run_history`)" — called the tool anyway and received the
full cross-session history for the dataset, including another run's best NLPD
and its model notes.

Fixed by the call-side allowlist, plus removing the system-prompt reference.
Affected runs were discarded.

**Two lessons, both generalisable:**

- *Not offering a capability is not the same as denying it.* Agents call
tools they were never shown.
- *The system prompt is part of the tool surface.* It named a tool the loop
had deliberately withheld, which is how the model learned the tool existed.

---



## Incident 2 — documentation drift (found 2026-08-02)

For a considerable time after the fix, three documents still described
`get_run_history` as part of normal agent operation:

- `README.md` (this repo) — listed among tools, no leakage annotation
- `auto-stan-agent/README.md` — *"Iteratively improves the Stan model,
referencing* `get_run_history` *before …"*
- `CLAUDE.md` — listed as an exposed tool

Nothing in the code was wrong; the documentation was. Anyone re-implementing
the loop from the docs would have reintroduced the leak. This file is the
single place where the tool surface is specified; the others now link here
rather than restating it.

---



## Adding a tool — checklist

1. Which leak class can its return value carry (L1/L2/L3, or none)?
2. Does it return anything derived from data the agent should not see —
  including *other runs*?
3. If it must be withheld: is it excluded from the offered list **and** covered
  by the call-side allowlist?
4. Is there a test asserting the sensitive value is absent from the response?
5. Update this table. Do not restate the policy elsewhere; link to it.

---



## Related

- `oduerr/autostan-private` → `paper_v2/findings/shadow_evaluation.md` — why
L3 matters and how the shadow set is constructed
- Issue #3 — server-side sampling timeout (availability, not leakage)

