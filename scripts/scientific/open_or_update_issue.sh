#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <suite_name> <report_markdown_path>" >&2
  exit 2
fi

SUITE_NAME="$1"
REPORT_PATH="$2"

if [[ -z "${GITHUB_TOKEN:-}" || -z "${GITHUB_REPOSITORY:-}" ]]; then
  echo "GITHUB_TOKEN or GITHUB_REPOSITORY missing; skipping issue update"
  exit 0
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for issue automation; skipping"
  exit 0
fi

API_ROOT="https://api.github.com/repos/${GITHUB_REPOSITORY}"
ISSUES_URL="${API_ROOT}/issues"
TITLE="[Scientific Regression] ${SUITE_NAME}"
LABEL="scientific-regression"
OWNER="${TRIPPY_BALL_SCI_OWNER:-}"
REPORT_BODY="$(cat "${REPORT_PATH}")"

set +e
curl -fsSL \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "${API_ROOT}/labels/${LABEL}" >/dev/null
label_status=$?
set -e
if [[ "${label_status}" -ne 0 ]]; then
  LABEL_PAYLOAD="$(jq -cn --arg name "$LABEL" --arg color "B60205" '{name:$name, color:$color}')"
  curl -fsSL -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -d "$LABEL_PAYLOAD" \
    "${API_ROOT}/labels" >/dev/null || true
fi

EXISTING_JSON="$(curl -fsSL \
  -H "Authorization: Bearer ${GITHUB_TOKEN}" \
  -H "Accept: application/vnd.github+json" \
  "${ISSUES_URL}?state=open&labels=${LABEL}&per_page=100")"

ISSUE_NUMBER="$(echo "$EXISTING_JSON" | jq -r --arg title "$TITLE" '.[] | select(.title == $title) | .number' | head -n1)"

if [[ -z "${ISSUE_NUMBER}" ]]; then
  ASSIGNEES_JSON="[]"
  if [[ -n "${OWNER}" ]]; then
    ASSIGNEES_JSON="$(jq -cn --arg owner "$OWNER" '[ $owner ]')"
  fi

  PAYLOAD="$(jq -cn \
    --arg title "$TITLE" \
    --arg body "$REPORT_BODY" \
    --arg label "$LABEL" \
    --argjson assignees "$ASSIGNEES_JSON" \
    '{title:$title, body:$body, labels:[$label], assignees:$assignees}')"

  curl -fsSL -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -d "$PAYLOAD" \
    "${ISSUES_URL}" >/dev/null

  echo "created issue: ${TITLE}"
else
  COMMENT_PAYLOAD="$(jq -cn --arg body "$REPORT_BODY" '{body:$body}')"
  curl -fsSL -X POST \
    -H "Authorization: Bearer ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    -d "$COMMENT_PAYLOAD" \
    "${ISSUES_URL}/${ISSUE_NUMBER}/comments" >/dev/null

  echo "updated issue #${ISSUE_NUMBER}: ${TITLE}"
fi
