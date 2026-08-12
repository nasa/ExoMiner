"""
Publish models to Hugging Face in a single commit.
"""

# Standard library
from pathlib import Path

# 3rd party
from huggingface_hub import HfApi, CommitOperationAdd

api = HfApi()
repo_id = "miguelmartinho/exominer-pipeline-models" # Change to nasa/exominer-models later
version_tag = "v2.1" # Should match your GitHub release version

# List to hold all the files we want to upload in this single commit
operations = []

# 1. Add models.tar (Ensure this path is correct!)
operations.append(
    CommitOperationAdd(
        path_in_repo="models.tar",
        path_or_fileobj="exominer_pipeline_data/models/models.tar" 
    )
)

# 2. Add the Model Card (renaming it to README.md in the repo)
operations.append(
    CommitOperationAdd(
        path_in_repo="README.md",
        path_or_fileobj="docs/models_specs.md"
    )
)

# Helper function to gather all files in a local folder
def add_folder_to_operations(local_folder: str, repo_prefix: str):
    base_path = Path(local_folder)
    # Recursively find all files in the directory
    for filepath in base_path.rglob("*"):
        if filepath.is_file():
            # Get the path relative to the folder to maintain internal structure
            rel_path = filepath.relative_to(base_path)
            # Hugging Face uses forward slashes for paths
            repo_path = f"{repo_prefix}/{rel_path.as_posix()}"
            
            operations.append(
                CommitOperationAdd(
                    path_in_repo=repo_path,
                    path_or_fileobj=str(filepath)
                )
            )

# 3. Add normalization statistics folders
print("Gathering phot_vetting stats...")
add_folder_to_operations(
    local_folder="exominer_pipeline_data/norm_stats/phot_vetting",
    repo_prefix="normalization_stats_phot-vetting"
)

print("Gathering planet_validation stats...")
add_folder_to_operations(
    local_folder="exominer_pipeline_data/norm_stats/planet_validation",
    repo_prefix="normalization_stats_planet-validation"
)

# 4. Push everything in one single commit
print(f"Pushing {len(operations)} files in a single commit...")
api.create_commit(
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Release {version_tag} models and stats",
    operations=operations
)
print("Commit successful!")

# 5. Create a tag for this pipeline version
print(f"Tagging repository with {version_tag}...")
api.create_tag(
    repo_id=repo_id, 
    tag=version_tag, 
    repo_type="model"
)

print(f"Successfully uploaded and tagged {version_tag} in one commit!")