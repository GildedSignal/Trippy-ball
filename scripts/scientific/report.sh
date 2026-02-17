#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 <suite_name> <status> <log_path> <out_dir> [repro_command]" >&2
  exit 2
fi

SUITE_NAME="$1"
STATUS="$2"
LOG_PATH="$3"
OUT_DIR="$4"
REPRO_COMMAND="${5:-}"

mkdir -p "$OUT_DIR"

TIMESTAMP_UTC="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
REPO="${GITHUB_REPOSITORY:-unknown}"
SHA="${GITHUB_SHA:-unknown}"
RUN_URL=""
if [[ -n "${GITHUB_SERVER_URL:-}" && -n "${GITHUB_REPOSITORY:-}" && -n "${GITHUB_RUN_ID:-}" ]]; then
  RUN_URL="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}"
fi

SUMMARY_PATH="${OUT_DIR}/scientific_report.md"
JSON_PATH="${OUT_DIR}/scientific_report.json"
DETAILS_PATH="${OUT_DIR}/scientific_failures_excerpt.txt"

if [[ -f "$LOG_PATH" ]]; then
  grep -E "mismatch|failed|panic|assert|non-finite|negative|orthogonality|normalization|stationarity" "$LOG_PATH" | tail -n 80 > "$DETAILS_PATH" || true
else
  : > "$DETAILS_PATH"
fi

cat > "$SUMMARY_PATH" <<EOF
# Scientific Validation Report

- Suite: \`${SUITE_NAME}\`
- Status: \`${STATUS}\`
- Timestamp (UTC): \`${TIMESTAMP_UTC}\`
- Repository: \`${REPO}\`
- Commit: \`${SHA}\`
$( [[ -n "$RUN_URL" ]] && echo "- Run URL: ${RUN_URL}" )

## Repro
\`\`\`bash
${REPRO_COMMAND:-"(not provided)"}
\`\`\`

## Suspect Areas
- \`src/math/mod.rs\`
- \`src/math/spherical.rs\`
- \`src/math/radial.rs\`
- \`src/render/gpu_wavefunction.rs\`
- \`src/render_metal/shaders/wavefunction.metal\`
- \`docs/scientific_contract.md\`

## Failure Excerpt
\`\`\`text
$(cat "$DETAILS_PATH")
\`\`\`
EOF

cat > "$JSON_PATH" <<EOF
{
  "suite": "${SUITE_NAME}",
  "status": "${STATUS}",
  "timestamp_utc": "${TIMESTAMP_UTC}",
  "repository": "${REPO}",
  "commit": "${SHA}",
  "run_url": "${RUN_URL}",
  "repro_command": "${REPRO_COMMAND}",
  "suspect_files": [
    "src/math/mod.rs",
    "src/math/spherical.rs",
    "src/math/radial.rs",
    "src/render/gpu_wavefunction.rs",
    "src/render_metal/shaders/wavefunction.metal",
    "docs/scientific_contract.md"
  ]
}
EOF

echo "wrote ${SUMMARY_PATH}"
echo "wrote ${JSON_PATH}"
