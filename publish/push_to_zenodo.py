"""
Publish ExoMiner models to Zenodo.
Supports creating a new record OR updating an existing one.
"""

import os
import shutil
import requests
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Publish ExoMiner models and stats to Zenodo.")
    
    # Required arguments
    parser.add_argument("-t", "--token", required=True, 
                        help="Zenodo Personal Access Token.")
    parser.add_argument("-v", "--version", required=True, 
                        help="Version tag for this release (e.g., v2.1).")
    
    # Optional environment flag (defaults to sandbox)
    parser.add_argument("-e", "--env", choices=["sandbox", "prod"], default="sandbox", 
                        help="Target environment: 'sandbox' (testing) or 'prod' (production). Default is sandbox.")
    
    # Optional existing record ID (for versioning)
    parser.add_argument("-r", "--record-id", type=int, default=None, 
                        help="Existing Zenodo Record ID (if you are releasing a new version of an existing record).")
    
    args = parser.parse_args()

    # ==========================================
    # CONFIGURATION
    # ==========================================
    ACCESS_TOKEN = args.token
    VERSION_TAG = args.version
    EXISTING_RECORD_ID = args.record_id
    
    if args.env == "prod":
        BASE_URL = "https://zenodo.org/api"
        print("⚠️ WARNING: Running in PRODUCTION mode (zenodo.org) ⚠️")
    else:
        BASE_URL = "https://sandbox.zenodo.org/api"
        print("ℹ️ Running in SANDBOX mode (sandbox.zenodo.org)")

    HEADERS = {"Authorization": f"Bearer {ACCESS_TOKEN}"}

    # Verify that local files exist before starting the API process
    models_tar_path = "exominer_pipeline_data/models/models.tar" # <--- UPDATE THIS PATH
    stats_dir = "exominer_pipeline_data/norm_stats"
    docs_path = "docs/models_specs.md"
    
    if not os.path.exists(stats_dir) or not os.path.exists(docs_path):
        print("❌ Error: Could not find stats directory or docs. Make sure you run this from the repo root.")
        sys.exit(1)

    # ==========================================
    # STEP 1: PREPARE FILES (ZIP STATS FOLDER)
    # ==========================================
    print("Zipping normalization statistics...")
    shutil.make_archive(
        base_name="normalization_stats", 
        format="zip", 
        root_dir=stats_dir
    )

    files_to_upload = {
        "models.tar": models_tar_path,
        "normalization_stats.zip": "normalization_stats.zip",
        "README.md": docs_path
    }

    # ==========================================
    # STEP 2: GET OR CREATE THE DRAFT
    # ==========================================
    if EXISTING_RECORD_ID is None:
        print("Creating a brand new Zenodo record...")
        r = requests.post(f"{BASE_URL}/deposit/depositions", json={}, headers=HEADERS)
        r.raise_for_status()
        deposition = r.json()
    else:
        print(f"Creating a new version draft for record {EXISTING_RECORD_ID}...")
        r = requests.post(f"{BASE_URL}/deposit/depositions/{EXISTING_RECORD_ID}/actions/newversion", headers=HEADERS)
        r.raise_for_status()
        
        new_draft_url = r.json()['links']['latest_draft']
        r = requests.get(new_draft_url, headers=HEADERS)
        deposition = r.json()

    dep_id = deposition['id']
    bucket_url = deposition['links']['bucket']

    print(f"Draft created! Deposition ID: {dep_id}")

    # ==========================================
    # STEP 3: UPLOAD FILES
    # ==========================================
    for target_name, local_path in files_to_upload.items():
        print(f"Uploading {target_name}...")
        with open(local_path, "rb") as fp:
            r = requests.put(
                f"{bucket_url}/{target_name}", 
                data=fp, 
                headers=HEADERS
            )
            r.raise_for_status()

    # ==========================================
    # STEP 4: UPDATE METADATA
    # ==========================================
    print("Updating metadata...")
    with open(docs_path, "r") as f:
        description_text = f.read()

    # ==========================================
    # STEP 4: UPDATE METADATA
    # ==========================================
    print("Updating metadata...")
    with open(docs_path, "r") as f:
        description_text = f.read()

    metadata = {
        "metadata": {
            "title": "ExoMiner Pipeline Models and Normalization Statistics",
            "upload_type": "model", # Keep as 'dataset' since these are model weights/stats
            "description": f"Model weights and statistics for ExoMiner.<br><br><pre>{description_text}</pre>",
            "creators": [
                {"name": "Martinho, Miguel", "affiliation": "NASA"}
            ],
            "version": VERSION_TAG,
            "access_right": "open",
            "license": "Apache-2.0",
            
            # --- NEW METADATA FIELDS ADDED BELOW ---
            
            # 1. Language (Zenodo requires the 3-letter ISO 639-3 code for English)
            "language": "eng",
            
            # 2. GitHub URL (Linked as a related software resource)
            "related_identifiers": [
                {
                    "identifier": "https://github.com/nasa/ExoMiner",
                    "relation": "isSupplementTo", # Means this dataset supplements the GitHub repo
                    "resource_type": "software"
                }
            ],
            
            # 3. Programming Language (Added as a keyword)
            "keywords": [
                "exoplanets", "vetting", "NASA", "TESS", "machine learning", "Python", "photometry", "astronomy",
            ],
            
            # 4. Development Status & additional tech details (Added to 'notes' which appears prominently on the page)
            "notes": (
                "**Development Status:** Active / Production<br>"
                "**Programming Language:** Python 3.11<br>"
                "**Framework:** TensorFlow 2.13 / Keras<br>"
                "**Deployment:** Podman Containers",
                "**Alternative Model Repository**: https://huggingface.co/Miguelmartinho/exominer-pipeline-models"
            )
        }
    }

    r = requests.put(
        f"{BASE_URL}/deposit/depositions/{dep_id}", 
        data=json.dumps(metadata), 
        headers={**HEADERS, "Content-Type": "application/json"}
    )
    r.raise_for_status()

    # ==========================================
    # STEP 5: CLEANUP & NEXT STEPS
    # ==========================================
    os.remove("normalization_stats.zip")

    print("\n" + "="*60)
    print("✅ SUCCESS! Files uploaded and metadata set.")
    print("Review and publish your draft here:")
    print(deposition['links']['html'])
    print("="*60)

if __name__ == "__main__":

    main()