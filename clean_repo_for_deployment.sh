#!/bin/bash

set -e  # Exit on error

echo "Switching to master to start clean..."
git checkout master

echo "Creating or resetting 'deploy_exominer' to match local master..."
git branch -f deploy_exominer master
git checkout deploy_exominer

echo "Preparing repository for deployment by cleaning unnecessary files..."

# Add paths to .gitignore
cat <<EOL >> .gitignore
xai/
transit_detection/
archived_experiments/
tess_spoc_ffi/
data_wrangling/
exominer_pipeline/test_pipeline/
job_scripts/
others/3rd_party_licenses.md
others/get_3rd_party_licenses_pkgs.md
*.pbs
__pycache__/
src_preprocessing/diff_img/search_neighbors
models/exominer_new.yaml
models/model_config.yaml
clean_repo_for_deployment.sh
.dockerignore
EOL

find others/envs -type f -name "*.yml" ! -name "*amd64.yml" ! -name "*arm64.yml" | sed 's/^/others\/envs\//' >> .gitignore

# Remove tracked files from index
git rm -r --cached xai/ transit_detection/ archived_experiments/ tess_spoc_ffi/ data_wrangling/ exominer_pipeline/test_pipeline/ job_scripts/
git rm --cached others/3rd_party_licenses.md || true
git rm --cached others/get_3rd_party_licenses_pkgs.md || true
git rm -r --cached src_preprocessing/diff_img/search_neighbors || true
git rm --cached models/exominer_new.yaml models/model_config.yaml || true
find others/envs -type f -name "*.yml" ! -name "*amd64.yml" ! -name "*arm64.yml" -exec git rm --cached {} \; || true
git rm --cached clean_repo_for_deployment.sh .dockerignore || true
find . -type f -name "*.pbs" -exec git rm --cached {} \; || true
find . -type f -path "*/__pycache__/*" -exec git rm --cached {} \; || true

git add .gitignore

echo "Committing cleanup changes..."
git commit -am "Preparing repository for deployment by cleaning unnecessary files."

echo "Pushing cleaned branch to NASA GitHub..."
git push -f nasa_github deploy_exominer:main
