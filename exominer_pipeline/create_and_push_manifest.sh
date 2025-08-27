# Creates a manifest with the new images and pushes it to GitHub Container Registry
# it assumes you have already built the images for both architectures and tagged them as localhost/exominer:amd64 and localhost/exominer:arm64

# Create a manifest list
podman manifest create ghcr.io/nasa/exominer:latest

# Add your architecture-specific images
podman manifest add ghcr.io/nasa/exominer:latest localhost/exominer:amd64
podman manifest add ghcr.io/nasa/exominer:latest localhost/exominer:arm64

# Push the manifest list
podman manifest push --all ghcr.io/nasa/exominer:latest
