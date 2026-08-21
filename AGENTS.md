# AGENTS.md — working on this codebase

(Installing the server for a user? That is [AGENTS_SETUP.md](AGENTS_SETUP.md),
not this file.)

- **Never read, copy, or serve anything under `datasets/*/protected/`** —
  held-out test labels. [TOOL_POLICY.md](TOOL_POLICY.md) is the authoritative
  leakage policy; link to it, don't restate it.
- **Adding or exposing a tool?** It needs a TOOL_POLICY row (leak class,
  offered-in-benchmarks?) and a test that the sensitive value stays absent.
- **Test before pushing:**
  `pytest test_helpers.py test_shadow_isolation.py test_server_stdio.py tests/use_cases -k "not runs_shadow"`
  (fast, no CmdStan needed; CI runs it on Python 3.11 and 3.12 — the two
  resolve different arviz majors, both must pass).
- **Features follow use cases**: close a gap in
  [docs/USE_CASES.md](docs/USE_CASES.md) and flip its verdict with a scripted
  test under `tests/use_cases/`; prompts that really ran belong in
  [docs/DEMO_PROMPTS.md](docs/DEMO_PROMPTS.md) with client + model + date.
- Do not paste CSV or draw contents into context — compute on files, read
  back aggregates.
