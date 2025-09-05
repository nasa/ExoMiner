
echo "Check out to track public ExoMiner repository."
git checkout -b deploy_exominer nasa_github/main

echo "Merging latest changes from local master branch."
git merge master

echo "Preparing repository for deployment by cleaning unnecessary files."

# add paths to .gitignore
echo "xai/" >> .gitignore
echo "transit_detection/" >> .gitignore
echo "archived_experiments/" >> .gitignore
echo "tess_spoc_ffi/" >> .gitignore
echo "data_wrangling/" >> .gitignore
echo "exominer_pipeline/test_pipeline/" >> .gitignore
echo "job_scripts/" >> .gitignore
echo "others/3rd_party_licenses.md" >> .gitignore
echo "others/get_3rd_party_licenses_pkgs.md" >> .gitignore
echo "*.pbs" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "src_preprocessing/diff_img/search_neighbors" >> .gitignore
echo "models/exominer_new.yaml" >> .gitignore
echo "models/model_config.yaml" >> .gitignore
find others/envs -type f -name "*.yml" ! -name "*amd64.yml" ! -name "*arm64.yml" | sed 's/^/others\/envs\//' >> .gitignore
echo "clean_repo_for_deployment.sh" >> .gitignore
echo ".dockerignore" >> .gitignore

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
git commit -am "Preparing repository for deployment by cleaning unnecessary files2."

# push to remote
# echo "Pushing changes to NASA GitHub."
# git push nasa_github main
