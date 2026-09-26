#!/usr/bin/env bash
set -euo pipefail

BASE_REF="${BASE_REF:-upstream/main}"

git rev-parse --verify "${BASE_REF}" >/dev/null

git diff --exit-code "${BASE_REF}" -- \
  src/backend.py \
  src/frontend.py \
  src/config.yaml \
  demo \
  evaluation

echo "OK: FD-BADCAT core matches ${BASE_REF}"
