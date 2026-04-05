#!/usr/bin/env bash
set -euo pipefail

MODELS=()
for arg in "$@"; do
  case "$arg" in
    --with-awq)
      MODELS+=("codet5" "local-awq" "platform-awq")
      ;;
    --with-full)
      MODELS+=("codet5" "local-full" "platform-full")
      ;;
    *)
      MODELS+=("$arg")
      ;;
  esac
done

if [ "${#MODELS[@]}" -eq 0 ]; then
  MODELS=("codet5" "local-awq")
fi

python scripts/download_models.py --models "${MODELS[@]}"

