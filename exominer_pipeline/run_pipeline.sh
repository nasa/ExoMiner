### Run ExoMiner Pipeline by running the Python application

while getopts "i:t:r:m:p:j:d:e:s:u:w:h" opt; do
  case $opt in
    i) inputs_dir="$OPTARG" ;;
    t) tics_tbl_fn="$OPTARG" ;;
    r) exominer_pipeline_run_dir="$OPTARG" ;;
    m) data_collection_mode="$OPTARG" ;;
    p) num_processes="$OPTARG" ;;
    j) num_jobs="$OPTARG" ;;
    d) download_spoc_data_products="$OPTARG" ;;
    e) external_data_repository="$OPTARG" ;;
    s) pipeline_python_script="$OPTARG" ;;
    u) stellar_parameters_source="$OPTARG" ;;
    w) ruwe_source="$OPTARG" ;;
    h)
      echo "Usage: $0 -i  -t  -r  -m  -p  -j  -d  -e  -s  -u  -w"
      echo ""
      echo "Arguments:"
      echo "  -i  Directory where input files are stored"
      echo "  -t  Filename of the TICs table"
      echo "  -r  Directory where the pipeline run will be saved"
      echo "  -m  Data collection mode (2min or ffi)"
      echo "  -p  Number of processes"
      echo "  -j  Number of jobs"
      echo "  -d  Whether to download SPOC data products (true/false)"
      echo "  -e  Path to external data repository or 'null'"
      echo "  -s  Path to the Python pipeline script"
      echo "  -u  Source of TICs stellar parameters"
      echo "  -w  Source of TICs Gaia RUWE values"
      echo "  -h  Show this help message and exit"
      exit 0
      ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

mkdir -p "$exominer_pipeline_run_dir"

tics_tbl_fp=$inputs_dir/$tics_tbl_fn

echo "Started ExoMiner Pipeline run $exominer_pipeline_run_dir..."
echo "Running ExoMiner Pipeline with the following parameters:"
echo "Inputs directory: $inputs_dir"
echo "TICs table file: $tics_tbl_fp"
echo "ExoMiner Pipeline run directory: $exominer_pipeline_run_dir"

# conditionally add external_data_repository
if [ "$external_data_repository" != "null" ]; then
  external_data_repository_arg="--external_data_repository=/external_data_repository"
else
  external_data_repository_arg=""
fi

# add mount to external TICs stellar parameters catalog if filepath provided
if [ -f "$stellar_parameters_source" ]; then
    volume_mounts="$volume_mounts -v $stellar_parameters_source:/tics_stellar_parameters.csv:Z"
    stellar_parameters_source_arg=/tics_stellar_parameters.csv
else
    stellar_parameters_source_arg=$stellar_parameters_source
fi

# add mount to external TICs RUWE catalog if filepath provided
if [ -f "$ruwe_source" ]; then
    volume_mounts="$volume_mounts -v $ruwe_source:/tics_ruwe.csv:Z"
    ruwe_source_arg=/tics_ruwe.csv
else
    ruwe_source_arg=$ruwe_source
fi

python "$pipeline_python_script" \
  --tic_ids_fp=/inputs/"$tics_tbl_fn" \
  --output_dir=/outputs \
  --data_collection_mode="$data_collection_mode" \
  --num_processes="$num_processes" \
  --num_jobs="$num_jobs" \
  --download_spoc_data_products="$download_spoc_data_products" \
  --stellar_parameters_source="$stellar_parameters_source_arg" \
  --ruwe_source="$ruwe_source_arg" \
  $external_data_repository_arg \

echo "Finished ExoMiner Pipeline run $exominer_pipeline_run_dir."