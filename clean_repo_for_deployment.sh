
echo "Check out to track public ExoMiner repository."
git checkout -b deploy_exominer nasa_github/main

echo "Merging latest changes from local master branch."
# git merge master
git reset --hard master

echo "Preparing repository for deployment by cleaning unnecessary files."

# add paths to .gitignore

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

# remove tracked files from index
git rm -r --cached xai/
git rm -r --cached transit_detection/
git rm -r --cached archived_experiments/
git rm -r --cached tess_spoc_ffi/
git rm -r --cached data_wrangling/
git rm -r --cached exominer_pipeline/test_pipeline/
git rm -r --cached job_scripts/
git rm --cached others/3rd_party_licenses.md
git rm --cached others/get_3rd_party_licenses_pkgs.md
git rm -r --cached src_preprocessing/diff_img/search_neighbors
git rm --cached models/exominer_new.yaml
git rm --cached models/model_config.yaml
find others/envs -type f -name "*.yml" ! -name "*amd64.yml" ! -name "*arm64.yml" -exec git rm --cached {} \;
git rm --cached clean_repo_for_deployment.sh
find . -type f -name "*.pbs" -exec git rm --cached {} \;
git rm --cached .dockerignore
git rm --cached .gitignore
# find . -type d -name "__pycache__" -exec git rm -r --cached {} +
find . -type f -path "*/__pycache__/*" -exec git rm --cached {} +

git add .gitignore

# commit changes
echo "Committing changes."
git commit -am "Preparing repository for deployment by cleaning unnecessary files."

echo "Rebasing to remove cleanup commit from history..."
git rebase -i master
# NOTE: In the interactive editor that opens:
# - Change `pick` to `drop` for the cleanup commit
# - Save and close the editor

# push to remote
echo "Pushing changes to NASA GitHub."
git push nasa_github deploy_exominer:main
