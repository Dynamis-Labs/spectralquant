#!/usr/bin/env bash
# Polls the LongBench relaunch on Modal volume, snapshots status & partials
# into the repo, and emits a stdout line on each poll plus a terminal line.
# Designed for use with the harness Monitor tool (each line is an event).
set -uo pipefail

VOL="spectralquant-v2-results"
STATUS_DIR_REMOTE="/status/longbench/longbench__Qwen2.5-7B__b3_seed42_subsetdeterministic_n50_in8192_out128_fp16+spectralquant_v2+turboquant"
SNAP_DIR="results/v3/modal/longbench_relaunch_2026-04-30/snapshots"
CALL_ID="fc-01KQFKCFNP61M33NMBN7CQ4DQB"
SLEEP_S="${WATCH_SLEEP:-1500}"  # 25 min default; pass WATCH_SLEEP=60 for fast tests
DEADLINE_S="${WATCH_DEADLINE:-46800}"  # 13h hard cutoff (12h + 1h buffer)

mkdir -p "$SNAP_DIR"
START=$(date +%s)

while :; do
  NOW_S=$(date +%s)
  ELAPSED=$((NOW_S - START))
  TS=$(date -u +"%Y%m%dT%H%M%SZ")
  TMP="$SNAP_DIR/$TS"
  mkdir -p "$TMP/partial"

  # Pull status.json and partial dir; ignore failures (files may not yet exist)
  modal volume get "$VOL" "$STATUS_DIR_REMOTE/status.json" "$TMP/status.json" >/dev/null 2>&1 || true
  modal volume get "$VOL" "$STATUS_DIR_REMOTE/partial" "$TMP/partial" >/dev/null 2>&1 || true

  # Build a one-line summary
  if [[ -f "$TMP/status.json" ]]; then
    STAGE=$(python3 -c "import json,sys
try:
    b=json.load(open('$TMP/status.json'))
    print(b.get('stage','?'),'|',b.get('message','?'),'|ts=',b.get('timestamp','?'))
except Exception as e:
    print('parse_err',e)" 2>/dev/null)
  else
    STAGE="no-status-yet"
  fi

  # Count method__*.json shards (paper-valid recovery indicator)
  METHOD_RECS=$(ls "$TMP/partial/partial/method__"*.json 2>/dev/null | wc -l)
  PROGRESS_RECS=$(ls "$TMP/partial/partial/"*.json 2>/dev/null | grep -v "/method__" | grep -v "partial_status.json" | wc -l)

  echo "[watch] elapsed=${ELAPSED}s method_recs=${METHOD_RECS} progress_recs=${PROGRESS_RECS} | $STAGE"

  # Check if function call finished (modal CLI: best-effort)
  if python3 -c "import modal,sys
try:
    fc=modal.FunctionCall.from_id('$CALL_ID')
    r=fc.get(timeout=0)
    print('CALL_DONE')
except Exception:
    sys.exit(1)
" 2>/dev/null | grep -q CALL_DONE; then
    echo "[watch] terminal: function call completed"
    break
  fi

  # 3 method records => paper-valid achievable; harness should still be writing canonical JSON
  if [[ "$METHOD_RECS" -ge 3 ]]; then
    echo "[watch] terminal: all 3 method records present (paper-valid recoverable)"
    break
  fi

  if (( ELAPSED > DEADLINE_S )); then
    echo "[watch] terminal: deadline ${DEADLINE_S}s exceeded"
    break
  fi

  sleep "$SLEEP_S"
done
