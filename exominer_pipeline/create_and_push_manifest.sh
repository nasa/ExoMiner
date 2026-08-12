#!/bin/bash

# Creates a manifest with the new images and pushes it to GitHub Container Registry
# it assumes you have already built the images for both architectures and tagged them as localhost/exominer:amd64 and localhost/exominer:arm64

# Accept version_tag as the first argument (optional)
version_tag="$1"

echo "Creating local manifest for 'latest'..."
podman manifest rm ghcr.io/nasa/exominer:latest 2>/dev/null || true
podman manifest create ghcr.io/nasa/exominer:latest

echo "Adding amd64 and arm64 images to manifest..."
podman manifest add ghcr.io/nasa/exominer:latest localhost/exominer:amd64
podman manifest add ghcr.io/nasa/exominer:latest localhost/exominer:arm64

# Always push the latest tag
echo "Pushing manifest to ghcr.io/nasa/exominer:latest..."
podman manifest push --all ghcr.io/nasa/exominer:latest

# If a version tag was provided, push the same manifest with the new tag
if [ -n "$version_tag" ]; then
    echo "Pushing manifest to ghcr.io/nasa/exominer:${version_tag}..."
    # By providing a second argument, we tell podman to push the local 'latest' manifest to a different destination tag
    podman manifest push --all ghcr.io/nasa/exominer:latest docker://ghcr.io/nasa/exominer:${version_tag}
    echo "Successfully pushed both 'latest' and '${version_tag}' tags!"
else
    echo "Successfully pushed 'latest' tag! (No version tag provided)"
fi
