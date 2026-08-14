#!/usr/bin/env bash
# Shared body of run_tier1.sh and run_tier2.sh.
#
# Not meant to be run directly. Each tier script sets TIER and TIER_BACKENDS,
# then sources this; the two differ only in which models they select and which
# backends have any, so the launching, reporting and argument handling live
# here once rather than in both.
#
# Expects, from the caller:
#   TIER            1 or 2
#   TIER_BACKENDS   backends with models in this tier, as "label:env:script"
#   TIER_SUMMARY    one line describing the tier, for the report header
#
set -uo pipefail

: "${TIER:?_tier.sh must be sourced from a tier script}"
: "${TIER_BACKENDS:?TIER_BACKENDS is unset}"
: "${TIER_SUMMARY:=medium models}"

REPO="$(cd "$HERE/../.." && pwd)"
EPOCHS=1
GPU=0
ONLY=""
REPORT_DIR="${CHEESE3D_REPORT_DIR:-$REPO/reports/medium_models}"
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --epochs)     EPOCHS="$2";     shift 2 ;;
        --gpu)        GPU="$2";        shift 2 ;;
        --only)       ONLY="$2";       shift 2 ;;
        --report-dir) REPORT_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -/{/^set -/!p;}' "$0"; exit 0 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

cd "$REPO"
ROOT="$(dirname "$REPO")/cheese3d_medium_model_tests"
REPORT="$REPORT_DIR/MEDIUM_${TIER}_RESULTS.md"
mkdir -p "$ROOT" "$REPORT_DIR"

# A whole-tier run owns this tier's report; --only refreshes one backend and
# leaves the other sections as they were.
if [ -z "$ONLY" ] || [ ! -f "$REPORT" ]; then
    {
        echo "# Medium models, tier ${TIER}"
        echo
        echo "${TIER_SUMMARY}"
        echo
        echo "${EPOCHS} epoch(s) each: enough to prove a model builds, accepts"
        echo "this project's data and takes optimizer steps."
        echo
        echo "Generated: $(date '+%Y-%m-%d %H:%M')"
        echo
        echo '```'
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
            2>/dev/null || echo "nvidia-smi unavailable"
        echo '```'
    } > "$REPORT"
fi

declare -a FAILED=()
for entry in $TIER_BACKENDS; do
    label="${entry%%:*}"; rest="${entry#*:}"
    env="${rest%%:*}"; script="${rest#*:}"
    [ -n "$ONLY" ] && [ "$ONLY" != "$label" ] && continue
    echo
    echo "=============================================================="
    echo "tier $TIER / $label  (pixi env: $env)"
    echo "=============================================================="
    pixi run -e "$env" python "$HERE/$script" \
        --tier "$TIER" --epochs "$EPOCHS" --gpu "$GPU" --root "$ROOT" \
        --report-dir "$REPORT_DIR" --report-name "MEDIUM_${TIER}_RESULTS.md" \
        "${EXTRA[@]+"${EXTRA[@]}"}" \
        || FAILED+=("$label")
done

echo
echo "=============================================================="
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "tier $TIER: every model ran"
else
    echo "tier $TIER, backends with failures: ${FAILED[*]}"
fi
echo "report: $REPORT"
exit ${#FAILED[@]}
