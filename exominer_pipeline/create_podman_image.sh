#!/usr/bin/env bash
#
# build_images.sh - Automate the creation of podman images for ExoMiner Pipeline
# Builds container images for both ARM64 and AMD64 architectures.
#
# Usage:
#   EXPORT_TO_REPO="ghcr.io/nasa" ./build_images.sh [arm64|amd64|all]

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
IMAGE_NAME="${IMAGE_NAME:-exominer}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

# Build metadata
SOFTWARE_VERSION_NUMBER="2.1.0"
GIT_REVISION="$(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

# Target architecture (default: all)
TARGET="${1:-all}"

# Custom environment and requirements file; don't use when building for both architectures unless you are sure the files are compatible with both
CUSTOM_ENV_FILE="${2:-}"
CUSTOM_REQS_FILE="${3:-}"

EXPORT_TO_REPO="${EXPORT_TO_REPO:-}"


build_image() {
    local arch="$1"
    
    # If a custom env file was provided as the 2nd argument, use it. 
    # Otherwise, default to the original naming convention.
    if [[ -n "${CUSTOM_ENV_FILE}" ]]; then
        local env_file="${CUSTOM_ENV_FILE}"
    else
        local env_file="environment_${arch}.yml"
    fi

    # If a custom requirements file was provided as the 3rd argument, use it. 
    # Otherwise, default to the original naming convention.
    if [[ -n "${CUSTOM_REQS_FILE}" ]]; then
        local reqs_file="${CUSTOM_REQS_FILE}"
    else
        local reqs_file="requirements_${arch}.txt"
    fi

    echo "============================================================"
    echo "Building for ${arch} architecture"
    echo "  Image:       ${IMAGE_NAME}:${arch}"
    echo "  Dockerfile:  ${DOCKERFILE}"
    echo "  Env file:    ${env_file}"
    echo "  Req file:    ${reqs_file}"
    echo "  Software version number:   ${SOFTWARE_VERSION_NUMBER}"
    echo "  Revision:    ${GIT_REVISION}"
    echo "  Created:     ${BUILD_DATE}"
    echo "============================================================"

    podman build \
        --arch "${arch}" \
        -f "${DOCKERFILE}" \
        -t "${IMAGE_NAME}:${arch}" \
        --no-cache \
        --label "org.opencontainers.image.version=${SOFTWARE_VERSION_NUMBER}" \
        --label "org.opencontainers.image.revision=${GIT_REVISION}" \
        --label "org.opencontainers.image.created=${BUILD_DATE}" \
        --build-arg "ENV_FILE=${env_file}" \
        --build-arg "REQS_FILE=${reqs_file}" \
        --build-arg "GIT_COMMIT_HASH=${GIT_REVISION}" \
        --build-arg "BUILD_DATE=${BUILD_DATE}" \
        .

    echo "Successfully built ${IMAGE_NAME}:${arch}"
    echo
}

usage() {
    echo "Usage: [EXPORT_TO_REPO=ghcr.io/nasa] $0 [arm64|amd64|all] [optional_custom_env_file.yml] [optional_custom_requirements_file.txt]"
    exit 1
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
case "${TARGET}" in
    arm64)
        build_image "arm64"
        ;;
    amd64)
        build_image "amd64"
        ;;
    all)
        build_image "arm64"
        build_image "amd64"

        echo "============================================================"
        echo "Creating Multi-Arch Manifest: ${IMAGE_NAME}:latest"
        echo "============================================================"
        
        # Remove old manifest if it exists so we can start fresh
        podman manifest rm "localhost/${IMAGE_NAME}:latest" 2>/dev/null || true
        
        # Create the new manifest list
        podman manifest create "localhost/${IMAGE_NAME}:latest"
        
        # Add the newly built architectures to the manifest
        podman manifest add "localhost/${IMAGE_NAME}:latest" "localhost/${IMAGE_NAME}:arm64"
        podman manifest add "localhost/${IMAGE_NAME}:latest" "localhost/${IMAGE_NAME}:amd64"
        
        echo "Manifest localhost/${IMAGE_NAME}:latest created successfully."

        # Only push if EXPORT_TO_REPO is set and not empty
        if [[ -n "${EXPORT_TO_REPO}" ]]; then
            echo "============================================================"
            echo "Pushing manifest to ${EXPORT_TO_REPO}..."
            echo "============================================================"
            podman manifest push "localhost/${IMAGE_NAME}:latest" "docker://${EXPORT_TO_REPO}/${IMAGE_NAME}:latest"
            podman manifest push "localhost/${IMAGE_NAME}:latest" "docker://${EXPORT_TO_REPO}/${IMAGE_NAME}:${SOFTWARE_VERSION_NUMBER}"
            echo "Successfully pushed manifest to ${EXPORT_TO_REPO}!"
        else
            echo "EXPORT_TO_REPO is not set. Skipping the push to a remote registry."
        fi
        ;;
    *)
        usage
        ;;
esac

echo "All requested builds completed successfully."