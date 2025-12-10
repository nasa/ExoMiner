#!/bin/bash

WORK_TREE_DIR=/Users/miguelmartinho/Projects/exominer_deployment
BRANCH_DEPLOYMENT=exominer_deployment
REMOTE_REPO=nasa_github
REMOTE_BRANCH=main

set -e  # Exit on error

# Check if a commit message file is provided as the first argument
if [[ -n "$1" && -f "$1" ]]; then
  COMMIT_MSG=$(<"$1")
  echo "Using commit message from file: $1"
else
  COMMIT_MSG="Preparing repository for deployment by cleaning unnecessary files."
  echo "Using default commit message."
fi

ORIGINAL_DIR=$(pwd)

echo "Switching to master to start clean..."
git checkout master

echo "Creating or resetting '$BRANCH_DEPLOYMENT' to match local master..."
git branch -f "$BRANCH_DEPLOYMENT" master

echo "Setting up worktree for '$BRANCH_DEPLOYMENT' in $WORK_TREE_DIR..."
git worktree add "$WORK_TREE_DIR" "$BRANCH_DEPLOYMENT"

echo "Navigating to worktree directory $WORK_TREE_DIR..."
cd "$WORK_TREE_DIR"

# git checkout "$BRANCH_DEPLOYMENT"

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
local_scripts/
others/3rd_party_licenses.md
others/get_3rd_party_licenses_pkgs.md
*.pbs
__pycache__/
models/model_config.yaml
clean_repo_for_deployment.sh
.dockerignore
EOL

find others/envs -type f -name "*.yml" ! -name "*amd64.yml" ! -name "*arm64.yml" | sed 's/^/others\/envs\//' >> .gitignore

# Remove tracked files from index
git rm -r --cached xai/ transit_detection/ archived_experiments/ tess_spoc_ffi/ data_wrangling/ exominer_pipeline/test_pipeline/ job_scripts/
git rm --cached others/3rd_party_licenses.md || true
git rm --cached others/get_3rd_party_licenses_pkgs.md || true
git rm --cached models/model_config.yaml || true
find others/envs -type f -name "*.yml" ! -name "*amd64.yml" ! -name "*arm64.yml" -exec git rm --cached {} \; || true
git rm --cached clean_repo_for_deployment.sh .dockerignore || true
find . -type f -name "*.pbs" -exec git rm --cached {} \; || true
find . -type f -path "*/__pycache__/*" -exec git rm --cached {} \; || true

git add .gitignore

if ! git diff --cached --quiet; then
  echo "Committing cleanup changes with message: '$COMMIT_MSG'"
  git commit -am "$COMMIT_MSG"
else
  echo "No changes to commit."
fi

echo "Pushing cleaned branch to $REMOTE_REPO on branch $REMOTE_BRANCH..."
git push -f "$REMOTE_REPO" "$BRANCH_DEPLOYMENT":"$REMOTE_BRANCH"

echo "Returning to original directory $ORIGINAL_DIR..."
cd "$ORIGINAL_DIR"

echo "Cleaning up worktree..."
git worktree remove "$WORK_TREE_DIR" --force

echo "Deployment branch '$BRANCH_DEPLOYMENT' pushed to $REMOTE_REPO in branch $REMOTE_BRANCH and worktree cleaned up."
