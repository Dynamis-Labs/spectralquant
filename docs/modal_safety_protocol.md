# SpectralQuant v2 — Modal Safety Protocol

This document is the operational safety contract for SpectralQuant v2 runs on
Modal. It is the companion to `docs/execution_audit_and_modal_runbook.md` and
must be read before any Modal run that loads weights or spends GPU credit.

**Today's anchor date.** 2026-04-29.

The rules below are written conservatively. They cost a few minutes of
preflight on every run; in exchange they prevent leaked secrets, silent
ratelimits, half-written JSONs, and unrecoverable mid-sweep crashes.

---

## 1. Credential handling

### 1.1 What counts as a credential

- Hugging Face access token (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`).
- Modal token id and secret (`MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`).
- W&B API key if used (`WANDB_API_KEY`).
- Any OAuth or personal-access-token equivalent.

### 1.2 Hard rules

1. **Never commit credentials.** No token value may appear in any file under
   git — not in `.env`, not in scripts, not in JSON output, not in commit
   messages, not in PR descriptions. The `.gitignore` includes `.env`,
   `.env.*` (except `.env.example`), `modal_token*`, `hf_token*`, `secrets/`,
   `*.pem`, `*.key`, `.modal/`, and `.huggingface/` so a stray file in the
   working tree is not staged by accident.
2. **Never echo credentials.** Scripts must not `print(os.environ["HF_TOKEN"])`,
   not even partially. The preflight script (`scripts/preflight_modal.py`)
   reports presence/absence only — `[ok] HF_TOKEN is set (length: N)` is
   acceptable; the value is not.
3. **Never log credentials.** When invoking shell commands that take secrets
   as arguments, use environment variables instead of CLI flags so the value
   does not appear in process listings or shell history. Example:
   `huggingface-cli login` reads from stdin or env, not `--token <value>`.
4. **Pass tokens via Modal secrets**, not via image baking. In a Modal app:

   ```python
   import modal
   image = modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04")
   app = modal.App("spectralquant-v2")
   @app.function(
       image=image,
       gpu="H200",
       secrets=[
           modal.Secret.from_name("hf-token"),       # exposes HF_TOKEN
           modal.Secret.from_name("wandb-api-key"),  # if used
       ],
       timeout=60 * 60,
   )
   def run_three_way(...):
       ...
   ```

   The Modal CLI itself authenticates via `~/.modal.toml`, which is gitignored
   and lives outside the repo.
5. **If a secret leaks** (committed by accident, posted in a log, sent in a
   message), treat it as compromised: rotate the token immediately on the
   issuing service, force-push remediation only with the maintainer's
   approval, and add a note to the audit doc.

### 1.3 Local development convention

- Keep an `.env.example` (committed) listing the *names* of variables only:
  `HF_TOKEN=`, `MODAL_TOKEN_ID=`, `MODAL_TOKEN_SECRET=`. Real values live in
  `.env` (gitignored).
- Source `.env` with `set -a; source .env; set +a` or `direnv` for local
  shells. Do not export secrets from the global shell rc — they will leak
  into unrelated processes.

---

## 2. Hugging Face license verification

Mistral-7B-v0.3 is a **gated model**. Without a license-accepting account,
`AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.3")` returns
`403`.

Preflight requirements before a Mistral run:

1. The HF account behind `HF_TOKEN` must have visited
   `https://huggingface.co/mistralai/Mistral-7B-v0.3` and accepted the
   community license while logged in.
2. A dry, weight-free check — `huggingface_hub.HfApi().model_info(model_id,
   token=os.environ["HF_TOKEN"])` — succeeds. If this returns 403, **abort
   the run before model load**; do not pay for a GPU minute on a run that
   will fail at weight download.
3. Record the tokenizer revision and model revision (commit SHA on the HF
   side) into the result JSON's `software` field so the run is reproducible
   if the model is later updated upstream.

For Qwen2.5-7B (public): no license gate, but rate limits apply. Authenticate
anyway to get higher quotas.

---

## 3. GPU choice

- Default: **H200 SXM** — has enough VRAM for Mistral-7B-v0.3 with margin
  and is meaningfully faster than H100 SXM for FP16/bfloat16 matmul. H100
  SXM is acceptable as fallback.
- B200 / GB200: only if explicitly requested. The image in
  `scripts/setup_b200.sh` targets a B200; check that the cutile baseline
  builds against that arch before launching a sweep.
- A100 80GB: acceptable for smoke tests only. Do not run the full sweep on
  A100 — calibration timing will be unrepresentative of the production
  target.
- L40S / A10 / T4: refuse. Insufficient VRAM for 7B FP16 weights + KV cache
  + intermediate activations.

Pick the smallest GPU that fits the run and matches the production target.
Bigger does not mean safer — it means more dollars per crashed run.

---

## 4. Resumable result paths

The Phase 6 sweep is multiple runs. A crash mid-sweep must not lose previous
runs.

1. **Per-run JSON path is deterministic.** The output file name encodes
   `model_short`, `bits`, `seed`, and `n_calib` so that if a run is repeated
   it overwrites itself rather than collides with a different run:

   ```
   results/three_way/{model_short}_b{bits}_calib{n_calib}_seed{seed}.json
   ```

2. **Skip-if-exists by default.** `experiments/run_three_way.py` accepts
   `--skip-if-exists` (default true), `--resume` (alias), and `--force`
   (default false). With skip, a half-completed sweep can be resumed by
   re-running the same shell loop without redoing finished cells; `--force`
   re-runs and overwrites.
3. **Per-run scratch directory.** Intermediate state during a single run
   (compressed key tensors, calibration matrices) lives under
   `local_results/scratch/<run_id>/` (gitignored). Clean up on success;
   leave intact on failure for postmortem.
4. **Dry-run / synthetic-smoke before any Modal call.**
   `experiments/run_three_way.py` exposes three safe modes:

   - `--dry-run` validates argv, prints the run plan and the deterministic
     output path, optionally schema-checks an in-memory placeholder
     payload, and writes nothing.
   - `--synthetic-smoke` runs the three engines (TurboQuant baseline,
     SpectralQuant v1, SpectralQuant v2) end-to-end on small synthetic
     Q/K/V tensors and writes a schema-valid JSON under
     `results/three_way_smoke/` (prefixed `synthetic_smoke__…` so it
     can never be confused with a real Modal artifact).
   - `--inline-corpus-smoke` runs the **full HF model path** (real model
     load, adapter discovery, hooks, calibration, quantization, eval) but
     swaps the WikiText / `datasets.load_dataset` corpus for a tiny
     deterministic in-memory list. It is intended to validate the harness
     plumbing on a small Qwen variant (e.g. `Qwen/Qwen2.5-0.5B`) when the
     dataset download is hanging or unavailable. Outputs are prefixed
     `inline_corpus_smoke__…`, mark `mode: "inline-corpus-smoke"`, set
     `paper_valid: false`, and use `calibration_corpus: "inline_smoke"` so
     they are never mistaken for benchmark evidence. **Full sweeps must
     still use the WikiText / proper eval corpus.**

   `--dry-run` and `--synthetic-smoke` refuse to call HuggingFace, never
   download a model, and never read or echo `HF_TOKEN`.
   `--inline-corpus-smoke` does load the model (the whole point is to
   validate the model-load → hook → quant → eval path) but skips the
   dataset download, so it cannot stand in for benchmark evidence.

   The **full HF model path** (`run_three_way.py` without
   `--dry-run`/`--synthetic-smoke`) lazy-imports `transformers` and
   `datasets`. Locally, with those packages absent, it raises a clear
   `RuntimeError` saying so — never a silent fallback. On Modal, where
   the image installs both, the path:

   1. Loads tokenizer + `AutoModelForCausalLM` in the requested dtype.
      `HF_TOKEN` is consumed by `huggingface_hub` automatically; the
      script never reads its value.
   2. Discovers attention layers via `experiments/model_adapters.py`
      (Mistral / Llama / Qwen2 split-QKV layout supported; packed-QKV
      raises `UnsupportedArchitectureError`).
   3. Runs the eigenspectral calibrator on a `wikitext` slice (or the
      `--dataset-name`/`--dataset-config`/`--dataset-split` you pass).
   4. Optionally saves / reloads the calibration `.pt` artifact under
      `--calibration-dir`, keyed deterministically by
      `{model_short}_calib{N}_tok{T}_seed{S}` so resumed sweeps share
      one calibration per (model, seed, n_calib).
   5. Hooks `q_proj`/`k_proj`/`v_proj` of the sampled layers, captures
      the live Q/K/V tensors during a forward pass on each eval text,
      and computes per-layer attention-cosine vs FP16 reference for
      TurboQuant baseline, SpectralQuant v1, and SpectralQuant v2.
   6. Writes the schema-valid JSON via `atomic_write_json` (validates
      *before* `os.replace`).

5. **Calibration artifact resumability.** When `--calibration-dir` is
   set with `--save-calibration`, the calibrator artifact is written
   to disk before the engines fit so a crash mid-fit does not lose the
   eigendecomposition. Subsequent runs in the same sweep should pass
   `--load-calibration` to skip the re-calibration step.

---

## 5. Atomic JSON writes

Half-written JSON files break the schema validator and corrupt downstream
plot scripts. All result writes must be atomic:

```python
import json, os, tempfile

def atomic_write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic on POSIX
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
```

`os.replace` is atomic on POSIX. Validate the JSON against
`schemas/three_way_result.schema.json` *before* the rename — a file that
cannot validate must never appear at the canonical path.

---

## 6. Frequent artifact sync

Modal volumes are durable but not infinite, and a Modal app's local FS is
ephemeral. Sync artifacts back to durable storage as soon as they exist:

1. After every run completes successfully, copy the JSON and stdout log
   into the Modal volume mount (e.g. `/results`). Do not wait for the whole
   sweep.
2. If a Modal volume is not used, `modal nfs put <volume> <local>` after
   each run, or `git push` from the runner — but only the JSONs, never the
   token files. The audit script (`scripts/audit_results.py`) makes it
   easy to confirm what made it back.
3. Locally, after pulling results: re-run `pytest tests/test_result_schema.py
   -q` to confirm every JSON validates.

---

## 6a. Git provenance without `.git`

The Modal image deliberately excludes `.git` (it is large, often dirty, and
adds nothing to the run). That means a `git rev-parse HEAD` *inside* the
container falls through to the placeholder commit `0000000`, which silently
poisons the `commit` field in result JSONs.

To preserve provenance:

1. The launcher (`scripts/launch_modal_three_way.py`) detects the local
   repo's HEAD via `detect_local_git_commit(REPO_ROOT)` at submit time.
2. It forwards that SHA into the remote subprocess as the env var
   `SPECTRALQUANT_GIT_COMMIT`.
3. `experiments/run_three_way.py::_git_commit` reads the env var (and
   accepts an explicit `--git-commit <SHA>` override) before falling back
   to `git rev-parse`. Only hex SHAs of length ≥ 7 are accepted; bogus
   values are ignored so the placeholder is preferable to a silently
   wrong commit.
4. The launcher's `--dry-run` and per-run output prints the SHA it will
   forward so the operator can confirm provenance before launching.

If `detect_local_git_commit` returns `None` (run from outside a git
checkout) the launcher prints a warning and the result JSON records
`0000000`. **Do not silence this warning by patching the placeholder;
fix the working tree instead.**

---

## 6c. Heartbeat / progress status artifacts

Modal logs are often inaccessible due to control-plane resource limits. To
keep a remote run observable even when streaming logs are gone, the
launcher (`scripts/launch_modal_three_way.py`) and the harness
(`experiments/run_three_way.py`) emit small JSON status files at well-
known paths on the Modal volume.

Layout (anchored to the `/results` Modal volume):

```
/results/
  status/
    <run_id>/
      status.json     # latest snapshot — atomically rewritten on every emit
      events.jsonl    # append-only history of every emitted event
```

The `<run_id>` matches the result-JSON naming so the status directory is
trivially derived from the run config:
`<model_short>_b<bits>_calib<N>_eval<M>_seed<S>` (prefixed
`synthetic_smoke__` for smoke runs). Both `expected_output_path` and
`status_path` are printed by `main_entry` at submission time, e.g.:

```
[launch_modal_three_way] expected_output_path: /results/three_way/Mistral-7B-v0.3_b3_calib32_eval8_seed42.json
[launch_modal_three_way] status_path: /results/status/Mistral-7B-v0.3_b3_calib32_eval8_seed42/status.json
[launch_modal_three_way] poll status: modal volume get spectralquant-v2-results /results/status/Mistral-7B-v0.3_b3_calib32_eval8_seed42/status.json -
```

Two layers of status are emitted into the same `<status_dir>/<run_id>/`
directory:

1. **Modal-runner stages** — written by
   `scripts/launch_modal_three_way.py::run_one` *before*, *around*, and
   *after* the child `run_three_way.py` subprocess. These give the
   operator a heartbeat even when the child never starts (e.g. broken
   image, bad PYTHONPATH, OOM at Popen). Stages emitted in order:

   * `modal_run_one_entered` — first write *after* `run_one` is
     dispatched on the worker, before any project import. Contains the
     run_id, expected_output_path, status_path, cwd, python executable
     + version, and forwarded git commit. **If you see no
     `modal_run_one_entered` artifact, the remote function never ran.**
   * `subprocess_env_configured` — PYTHONPATH and HF cache env vars are
     set; the child has not yet been spawned.
   * `subprocess_starting` — about to call `subprocess.Popen`.
   * `subprocess_started` — Popen succeeded, the child has a PID.
   * `subprocess_progress` — every ~30 s while the child is alive,
     with bounded sanitized stdout/stderr tails.
   * `subprocess_end` — child exited 0.
   * `failure` (terminal) — Popen itself raised, the streaming loop
     raised, or the child exited non-zero.

2. **Benchmark stages** — written by
   `experiments/run_three_way.py` from inside the child subprocess.
   They overwrite `status.json` as the child progresses past each
   milestone: `start`, `import_ok`, `model_load_start`,
   `model_load_end`, `dataset_load_start`, `dataset_load_end`,
   `calibration_start`, `calibration_end`, `eval_start`,
   `eval_progress` (periodic), `eval_end`, terminal `success` or
   `failure`.

If `status.json` is stuck on a Modal-runner stage (`modal_run_one_entered`
or earlier), the child never started — investigate the image and import
chain. If it is stuck on a benchmark stage, the child started but hung at
that step.

Every payload includes:

- `run_id`, `commit`, `model`, `avg_bits`, `n_calib`, `n_eval`,
  `n_layers_sample`, `host`, `pid`, `timestamp` (UTC ISO);
- For non-terminal stages: optional `message`, `details`, and bounded
  sanitized `stdout_tail` / `stderr_tail` (≤4000 chars each);
- For `failure`: `error`, full sanitized `traceback`, and the latest
  stdout/stderr tails so the operator can triage without log access.

**Sanitization is enforced at write time.** `experiments/run_status.py::
sanitize_text` redacts (a) values of any environment variable whose name
matches the secret patterns `TOKEN`, `SECRET`, `API_KEY`, `APIKEY`,
`PASSWORD`, `PASSWD`, `PRIVATE_KEY` and (b) token-like literals
(`hf_...`, `ghp_...`, `github_pat_...`, `AKIA...`, `ak-...`, `as-...`).
Empty/short values (<8 chars) are skipped. Status artifacts therefore
carry no token values even if a noisy traceback printed one.

Polling commands:

```bash
# Read the latest snapshot (single JSON document):
modal volume get spectralquant-v2-results \
  /results/status/<run_id>/status.json -

# Read the full event history:
modal volume get spectralquant-v2-results \
  /results/status/<run_id>/events.jsonl -

# Or list the status tree to see all in-flight runs at once:
modal volume ls spectralquant-v2-results /results/status/
```

Failure rule: **if `subprocess` exits non-zero or raises**, the launcher
writes a `failure` status JSON containing the sanitized stdout/stderr
tails, the exit code, and (for raised exceptions) the traceback. This
artifact is the source of truth for triage when Modal logs are
unavailable.

### Smoke-gate rule

**No full sweep is launched until the HF smoke writes a `success` status
JSON.** The smoke run is the cheapest way to confirm: the image builds,
HF auth works, dataset loads, calibration runs, the result schema is
satisfied, and the volume is writable. If `status.json` does not show
`stage: success` for the smoke, do not pay for a full sweep — fix the
failure mode revealed by the smoke's `failure` artifact first.

#### Inline-corpus smoke (harness validation only)

`--inline-corpus-smoke` is a *harness validation* gate, **not** an
evidence-collection mode. It uses the full HF model-load and hook path
but bypasses `datasets.load_dataset` with a tiny deterministic in-memory
corpus. Use it when:

* the WikiText (or whichever) dataset download is hanging or rate-limited
  and you need to confirm the model-load → adapter-discovery → hooks →
  calibration → eval pipeline is intact;
* you want a fast hermetic gate on a small Qwen variant before paying
  for a full sweep on a 7B-class model.

Inline-corpus runs emit a distinct file (`inline_corpus_smoke__…json`),
set `mode: "inline-corpus-smoke"` and `paper_valid: false` in the result
JSON, and mark both calibration and eval corpora as `inline_smoke`. They
**must not** be cited as benchmark evidence; full sweeps require the
WikiText / proper eval corpus path. The status writer emits
`dataset_inline_start` / `dataset_inline_end` events instead of
`dataset_load_start` / `dataset_load_end` so it is unambiguous from
status alone which corpus path ran.

---

## 6d. Persistent HuggingFace cache on the volume

Retries should never re-download the model. The launcher pins the HF
and `datasets` caches to a persistent subtree of the Modal volume so a
killed-and-resumed run reuses the previous downloads:

```
/results/
  hf_cache/
    home/         # HF_HOME (modules, token cache)
    hub/          # HUGGINGFACE_HUB_CACHE / TRANSFORMERS_CACHE
    datasets/     # HF_DATASETS_CACHE
    xdg/          # XDG_CACHE_HOME
```

Env vars set by `experiments/run_status.py::configure_persistent_hf_cache`:
`HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`,
`HF_DATASETS_CACHE`, `XDG_CACHE_HOME`. The launcher's `run_one` calls
this function before spawning the subprocess.

**Cleanup cautions.**

- Never blanket-delete `/results/hf_cache/` while a run is in flight —
  the snapshot lock files under `hub/` are keyed by partial download
  sentinels and pulling them mid-fetch will brick the active retry.
- `modal volume rm spectralquant-v2-results /results/hf_cache` is the
  big-hammer reset; only run it after `modal app list` confirms there
  are no in-flight functions, and after `scripts/audit_results.py`
  confirms there are no `.tmp_*` leftovers (see §6b).
- The cache shares the volume with result JSONs. Deleting one must not
  touch the other; always specify the full subpath.
- Because the cache lives on the volume, it does **not** consume Modal
  ephemeral disk and will not be wiped between container starts.

---

## 6b. Stale `.tmp_*` files in result directories

`atomic_write_json` creates a `.tmp_<random>` file in the destination
directory and renames it on success. If the process is killed between
`mkstemp` and `os.replace` (Modal timeout, OOM, Ctrl-C, network blip),
the tempfile is left behind. These files are zero-byte or partially
written, are never schema-valid, and are a strong signal that a run
crashed mid-write.

`scripts/audit_results.py` scans the known result subdirectories
(`results/three_way`, `results/calibration`, etc.) for `.tmp_*`
leftovers and reports them in both the human table and the `--json`
payload (`stale_tmp_files`, `summary.stale_tmp_count`).

Defaults:

- `--scan-tmp` is **on by default**.
- `--no-scan-tmp` disables the scan if the operator only wants the
  manifest summary.
- `--delete-stale-tmp` removes the leftover files. **Off by default**:
  inspect first, then delete. The CLI refuses to delete anything whose
  name does not start with `.tmp_` or whose resolved path falls outside
  the repo root.

Workflow when a run crashes:

1. Run `python3 scripts/audit_results.py --json` and inspect the
   `stale_tmp_files` block to see which run(s) failed mid-write.
2. Cross-check against the launcher's `git_commit` and timestamps in
   the operator console to identify the responsible run.
3. Once the crash is triaged (re-run, infrastructure fix, etc.), run
   `python3 scripts/audit_results.py --delete-stale-tmp` to clean up.

---

## 7. Timeout strategy

### 7.0 Detached vs. attached invocation

Long Modal jobs (real-model smoke, full sweep cells) **must** be launched
in detached mode so the GPU run continues even if the local client
disconnects, hits a tool/wall-clock timeout, or the operator's network
flaps. Attached mode (`modal run` without `-d`, or the
`python3 scripts/launch_modal_three_way.py …` wrapper) is fine for
fast feedback during development but is **not** safe for paid GPU work
because killing the local client also kills the remote app.

The launcher exposes a top-level Modal `App` and a `main_entry` local
entrypoint precisely so `modal run -d` can submit the job directly with
no Python wrapper sitting between Modal and the operator's terminal.
Concretely:

```bash
# ---- Detached real-model smoke (Mistral-7B, 3-bit, n_calib=4, n_eval=2)
# Survives a local Ctrl-C / tool timeout / SSH disconnect.
modal run -d scripts/launch_modal_three_way.py::main_entry -- \
  --model mistralai/Mistral-7B-v0.3 \
  --avg-bits 3 \
  --n-calib 4 --n-eval 2 --n-layers-sample 2 \
  --output-dir /results/three_way_smoke \
  --status-dir /results/status_hf_smoke \
  --smoke
```

`--status-dir` is the *parent* directory; per-run heartbeat artifacts land
at `<status-dir>/<run_id>/` (status.json + events.jsonl). Omit the flag to
fall back to the default sibling-of-output-dir layout described below.

Modal prints an app id on submission, e.g. `App spectralquant-v2 has
started running. View at https://modal.com/apps/<workspace>/<app-id>`.
Copy that URL — it is the polling target.

The launcher's `main_entry` calls `run_one.spawn(...)` (not `.remote()`)
so the GPU function keeps running after the local `modal run -d` client
disconnects. The previous failure mode was that `.remote()` issued
inside a detached app is *cancelled* when the local caller drops its
connection, with this Modal warning in stderr:

```
remote() and .map() calls in detached apps may be canceled when the
local caller disconnects. Use .spawn() for detached or background work.
```

`spawn()` returns a `FunctionCall`; the launcher prints its `object_id`
(the poll handle) and the deterministic artifact path so you can pick up
the result later without needing the streaming logs.

Expected stdout from a detached submission (sanitized):

```
[launch_modal_three_way] forwarding local commit <sha12> via SPECTRALQUANT_GIT_COMMIT
[launch_modal_three_way] launching (entrypoint): mistralai/Mistral-7B-v0.3 b=3 seed=42 smoke=True
[launch_modal_three_way] spawned remote call: call_id=fc-XXXXXXXXXXXXXXXX
[launch_modal_three_way] expected_output_path: /results/three_way_smoke/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42.json
[launch_modal_three_way] poll: modal volume ls spectralquant-v2-results /results/three_way_smoke/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42.json
[launch_modal_three_way] result: python3 -c "import modal; print(modal.FunctionCall.from_id('fc-XXXXXXXXXXXXXXXX').get(timeout=0))"
```

Copy both `call_id=...` and `expected_output_path: ...` — they are the
two things you need to retrieve the run.

### 7.0a Polling and recovery

After detaching, monitor the run from the *same* terminal or a fresh one.
You have two independent polling channels:

**(a) Volume-based polling** (fastest, no log access required):

```bash
# Watch for the artifact to appear at the printed expected_output_path.
# Use the *exact* path printed by the launcher above.
modal volume ls spectralquant-v2-results /results/three_way_smoke

# Pull the result JSON locally once it exists:
modal volume get spectralquant-v2-results \
  /results/three_way_smoke/synthetic_smoke__Mistral-7B-v0.3_b3_calib4_eval2_seed42.json \
  ./local_results/
```

**(b) FunctionCall-based polling** (returns the structured run result):

```bash
# Non-blocking peek (timeout=0 raises if not ready, useful for status):
python3 -c "import modal; print(modal.FunctionCall.from_id('<call_id>').get(timeout=0))"

# Block until done (use for an attended check):
python3 -c "import modal; print(modal.FunctionCall.from_id('<call_id>').get())"
```

**(c) Log polling** (slower; helpful for triage):

```bash
modal app logs <app-id>           # tail logs (Ctrl-C does NOT stop the app)
modal app list                    # running apps + states
modal app stop <app-id>           # only if you must force-stop a hung run
```

Recovery rules:

1. The result JSON is written to the Modal volume `spectralquant-v2-results`
   under `--output-dir`. If the local client died but the run finished,
   the artifact is still there — confirm with `modal volume ls
   spectralquant-v2-results /three_way_smoke`. The exact filename is the
   `expected_output_path` printed at submission time.
2. A `.tmp_*` leftover in the result directory (see §6b) means the run
   crashed mid-write; pull the logs, triage, and clean with
   `python3 scripts/audit_results.py --delete-stale-tmp` once triage is
   done.
3. Never restart a crashed run with `--force` until you have confirmed
   the failure mode. Re-running with the default `--skip-if-exists`
   semantics will silently no-op if a stale artifact exists.
4. If `FunctionCall.from_id(<call_id>).get()` raises
   `modal.exception.ExecutionError` it usually means the call was
   cancelled (e.g. the app was stopped) — check `modal app logs` and
   the volume for partial state before re-submitting.

### 7.0b Per-call timeout matrix

Every Modal function call has a hard timeout. Set it conservatively but not
infinitely:

| Scope | Timeout | Notes |
|---|---|---|
| Smoke run (`--smoke`, `n_calib=4`, `n_eval=2`, 2 layers) | 15 min | If exceeded, something is wrong before model load. |
| Single Mistral b=B run | 90 min | Model load + 8 layers + n_eval=8. |
| Full Mistral b∈{2,3,5} sweep in one function | 4 h | Prefer one function per b — easier to retry. |
| Qwen 3-bit run | 90 min | |
| Accounting audit (no model load) | 5 min | Pure CPU. |

Inside the function, set per-step soft timeouts (`signal.alarm` or a thread
watchdog) to fail fast if calibration alone takes more than 10 minutes —
that is a sign the wrong tokenizer or wrong dataset path is in use.

---

## 8. Retry strategy

- **Transient errors** (HF rate limit, network blip, transient CUDA OOM
  from a previous tenant): retry up to 3 times with exponential backoff
  (60s, 180s, 600s). Modal's `retries=` argument can do this for the
  function call as a whole; finer-grained retries belong inside the
  HTTP/HF client.
- **Permanent errors** (license 403, missing dataset, schema validation
  fail): do not retry. Fail loudly, write a `.failure.json` with the error
  message and stack trace, and abort the sweep loop so the operator can
  fix the root cause before paying for further GPU minutes.
- **Partial-progress retries**: when retrying a run, honor `--skip-if-exists`
  so already-done bits don't re-run.

---

## 9. Preflight checks before expensive runs

Run the preflight script locally and on the Modal worker before kicking off
a sweep:

```bash
python3 scripts/preflight_modal.py --strict
```

If the preflight reports `disk_space` low (or you simply expect a large
download), free local cache space with `scripts/safe_local_cleanup.py`
**before** the Modal launch. **Never** use a broad `rm -rf` over `/tmp`,
`~/.cache`, or any directory that contains a working clone — that is exactly
how an active local checkout was lost in a previous incident.

```bash
# 1. Always start with a dry run — nothing is deleted without --yes:
python3 scripts/safe_local_cleanup.py \
    --delete-hf-cache --delete-playwright-cache --delete-temp-clones

# 2. Apply once you've reviewed the plan:
python3 scripts/safe_local_cleanup.py \
    --delete-hf-cache --delete-playwright-cache --delete-temp-clones --yes

# 3. Optional gate: fail the script if the repo's disk has < N GB free:
python3 scripts/safe_local_cleanup.py --min-disk-gb 10
```

The cleanup script:

- defaults to dry-run; nothing is deleted unless `--yes` is also passed;
- refuses to delete the active repo root (resolved via `git rev-parse
  --show-toplevel`), any ancestor of it, or anything inside it;
- excludes the active repo path exactly when scanning for temp clones under
  `/tmp` and `/var/tmp` — by both name comparison and resolved-symlink
  comparison;
- only touches an explicit allow-list of caches (`--delete-hf-cache`,
  `--delete-playwright-cache`, `--delete-temp-clones`,
  `--delete-torch-cache`, `--delete-pip-cache`); each must be selected
  individually;
- never reads or echoes `HF_TOKEN` or any other credential.

Any time the local workspace runs low on disk before a Modal launch, this is
the script to use. Broad `rm -rf` globs are an anti-pattern — see §11.

It must succeed with exit code 0 before the sweep begins. The script
verifies (without printing any token value):

1. Python version is 3.10–3.12.
2. We are inside a git repo, on a known branch, with no uncommitted changes
   (or `--allow-dirty` if intentional).
3. The `HF_TOKEN` environment variable is **present**, length is reported
   but value is not.
4. The Modal CLI is on PATH if Modal is being used (warn if not — the local
   preflight may be running outside the Modal image).
5. Disk space at the result-output directory is sufficient.
6. No tracked file path matches the secret patterns in `.gitignore` (catches
   accidentally committed `.env`, `modal_token`, etc.).
7. The expected result directories exist (or can be created).

The preflight script is **read-only** — it never starts a model run, never
downloads a weight, never spends GPU credit.

---

## 10. After a run

1. Use `scripts/audit_results.py` to list expected vs. present JSONs:

   ```bash
   python3 scripts/audit_results.py --expected docs/expected_artifacts.json
   ```

   The audit prints a "present / missing" table and exits non-zero if any
   expected artifact is missing. It does not interpret the contents — it
   only confirms file presence.

2. Re-run schema validation:

   ```bash
   pytest tests/test_result_schema.py -q
   ```

3. Update `docs/execution_audit_and_modal_runbook.md` §3 with the new
   results, the commit hash that produced them, and any caveats.

---

## 11. Anti-patterns (do not do)

- ❌ `os.environ.setdefault("HF_TOKEN", "hf_xxx...")` in any committed file.
- ❌ `subprocess.run(["modal", "token", "set", "--token-id", token_id, ...])`
  in any committed script — token values would appear in process listings.
- ❌ Catching exceptions broadly (`except Exception: pass`) around a
  weight-download call. A 403 must surface, not be swallowed.
- ❌ Writing partial JSONs to the canonical result path before validation.
- ❌ Skipping the smoke run because "the script worked yesterday." Models
  and datasets get re-pinned upstream; smoke tests catch that for free.
- ❌ Hard-coding seeds, model names, or output paths in committed scripts —
  use argparse so the same script runs every cell of the matrix.
- ❌ Running the full sweep on a laptop. Phase 6 is **Modal-only**; the
  cutile baseline is not buildable in a generic local Python env.
- ❌ `rm -rf /tmp/*`, `rm -rf ~/.cache/*`, or any other broad cleanup glob.
  An incident occurred where this pattern deleted the active local clone
  along with the intended cache directories. Use
  `scripts/safe_local_cleanup.py` instead — it refuses to touch the active
  repo root and requires explicit per-category flags plus `--yes`.

---

## 9. Engine selection for Phase 6

`src/spectralquant/__init__.py` exports two engines and benchmark scripts
must record which one they used in the result JSON's `software.engine`
field (V1-GAP-010 is now closed):

- **`SpectralQuantEngine`** — canonical pure-Python pipeline. Use this
  for every quality / accuracy benchmark (perplexity, attention cosine
  similarity, NIAH, three-way comparisons). It carries `use_water_fill`,
  `wf_min_bits`, `wf_max_bits` via `EngineConfig`, and exposes
  `allocation_metadata()` and `waterfill_allocations()` for the
  per-head bit budget that must be embedded in every result JSON.
- **`KernelSpectralQuantEngine`** — cuTile-accelerated subclass.
  Use **only** for kernel-path latency benchmarks where the CUDA
  `launch_*` methods are exercised; it is the same code path as the v1
  legacy engine. The kernel engine accepts the same water-fill flags
  but currently records the metadata only — per-dim execution on the
  kernel path is a future task. Constructing this class outside Modal
  raises `RuntimeError`; do not silently fall back to the canonical
  engine if the kernel engine fails to import.

Result JSONs for Phase 6 must record `software.engine` as either
`"SpectralQuantEngine"` (canonical) or `"KernelSpectralQuantEngine"`
(kernel) — these are the two valid values. Anything else is a bug.
