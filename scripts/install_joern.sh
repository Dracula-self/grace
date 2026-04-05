#!/usr/bin/env bash
set -euo pipefail

JOERN_VERSION="${JOERN_VERSION:-2.0.42}"
INSTALL_ROOT="${INSTALL_ROOT:-tools/joern}"
LOCAL_ARCHIVE_DIR="${LOCAL_ARCHIVE_DIR:-third_party/joern}"
TARGET_DIR="${INSTALL_ROOT}/${JOERN_VERSION}"
ARCHIVE_NAME="joern-cli-${JOERN_VERSION}.zip"
DOWNLOAD_URL="https://github.com/joernio/joern/releases/download/v${JOERN_VERSION}/${ARCHIVE_NAME}"

mkdir -p "${INSTALL_ROOT}"

if [ -d "${TARGET_DIR}" ]; then
  echo "[joern] already installed at ${TARGET_DIR}"
  exit 0
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [ -f "${LOCAL_ARCHIVE_DIR}/${ARCHIVE_NAME}" ]; then
  ARCHIVE_PATH="${LOCAL_ARCHIVE_DIR}/${ARCHIVE_NAME}"
  echo "[joern] using local archive ${ARCHIVE_PATH}"
else
  ARCHIVE_PATH="${TMP_DIR}/${ARCHIVE_NAME}"
  echo "[joern] downloading ${DOWNLOAD_URL}"
  curl -L "${DOWNLOAD_URL}" -o "${ARCHIVE_PATH}"
fi

unzip -q "${ARCHIVE_PATH}" -d "${TMP_DIR}"
mv "${TMP_DIR}/joern-cli" "${TARGET_DIR}"
echo "[joern] installed to ${TARGET_DIR}"
echo "[joern] export JOERN_HOME=\$PWD/${TARGET_DIR}"

