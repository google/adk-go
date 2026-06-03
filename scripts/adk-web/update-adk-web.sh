#!/bin/sh

# This scripts pulls the latest version of adk-web.
# It uses the latest version from https://github.com/google/adk-web and builds it in a docker container.


# Use directory of the script for references
SCRIPT_DIR="$(dirname "$0")"

OUTPUT_DIR="${SCRIPT_DIR}/../../cmd/launcher/web/webui/distr/"
CONTAINER_BUILD_DIR="adk-web/dist/agent_framework_web/browser"

if ! docker build -t adk-web-builder:latest "${SCRIPT_DIR}"; then
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
docker cp "${CONTAINER_ID}":/${CONTAINER_BUILD_DIR}/. "${OUTPUT_DIR}"

echo "Done."
