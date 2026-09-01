#!/bin/sh

# This script rebuilds the vendored adk-web bundle.
# It builds the pinned revision of https://github.com/google/adk-web in a docker
# container and copies the result into cmd/launcher/web/webui/distr/.
#
# The revision is pinned in the Dockerfile (ADK_WEB_REF). Set ADK_WEB_REF in the
# environment to build a different revision for a one-off check; to move the pin
# for everyone, edit the Dockerfile. See README.md in this directory.


# Use directory of the script for references
SCRIPT_DIR="$(dirname "$0")"

OUTPUT_DIR="${SCRIPT_DIR}/../../cmd/launcher/web/webui/distr/"
CONTAINER_BUILD_DIR="adk-web/dist/agent_framework_web/browser"
VERSION_FILE="adk-web-version.json"

# Print the upstream commit recorded in a provenance file, or nothing if the
# file is absent or has no commit in it.
read_upstream_sha() {
    if [ -f "$1" ]; then
        sed -n 's/.*"upstream_sha"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$1"
    fi
}

# Read the current provenance before the output directory is removed, so the
# refresh can report which upstream range it pulls in.
OLD_SHA="$(read_upstream_sha "${OUTPUT_DIR}${VERSION_FILE}")"

# ADK_WEB_REF overrides the pin in the Dockerfile for this build only.
if [ -n "${ADK_WEB_REF}" ]; then
    echo "Building adk-web ref ${ADK_WEB_REF} (overriding the Dockerfile pin)."
    set -- --build-arg "ADK_WEB_REF=${ADK_WEB_REF}"
else
    set --
fi

if ! docker build "$@" -t adk-web-builder:latest "${SCRIPT_DIR}"; then
    echo "Failed to build container. Stopping the update."
    exit 1
fi

CONTAINER_ID=$(docker create adk-web-builder:latest)
if [ $? -ne 0 ]; then
    echo "Failed to create container. Stopping the update."
    exit 1
fi
trap "docker rm -f ${CONTAINER_ID}" EXIT

echo "Cleaning up the output directory."
rm -rf "${OUTPUT_DIR}"
echo "Copying the built files from the container to the output directory."
if ! docker cp "${CONTAINER_ID}":/${CONTAINER_BUILD_DIR}/. "${OUTPUT_DIR}"; then
    echo "Failed to copy the built files, and the output directory is now empty."
    echo "Restore the committed bundle with 'git restore ${OUTPUT_DIR}' before"
    echo "retrying."
    exit 1
fi

NEW_SHA="$(read_upstream_sha "${OUTPUT_DIR}${VERSION_FILE}")"
if [ -z "${NEW_SHA}" ]; then
    echo "Failed to find an upstream commit in ${OUTPUT_DIR}${VERSION_FILE}."
    echo "The container did not record provenance. Check the Dockerfile."
    exit 1
fi

OLD_SHA_TEXT="${OLD_SHA}"
if [ -z "${OLD_SHA_TEXT}" ]; then
    OLD_SHA_TEXT="unknown, the previous bundle recorded no provenance"
fi

echo
echo "Done. The bundle in ${OUTPUT_DIR} now comes from:"
echo "  previous upstream commit: ${OLD_SHA_TEXT}"
echo "  new upstream commit:      ${NEW_SHA}"
if [ -n "${OLD_SHA}" ] && [ "${OLD_SHA}" != "${NEW_SHA}" ]; then
    echo "  upstream changes:         https://github.com/google/adk-web/compare/${OLD_SHA}...${NEW_SHA}"
fi
echo
echo "Put those commits and the compare URL in the refresh commit message, and"
echo "run the UI-to-server route contract test. See scripts/adk-web/README.md."
