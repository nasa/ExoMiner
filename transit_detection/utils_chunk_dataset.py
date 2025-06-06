from pathlib import Path
import glob


def build_chunk_mask(chunks_to_process: list, chunked_dataset_dir: Path):
    """
    Builds mask to exclude chunks that have been successfully processed for an iteration of dataset building
    """
    chunked_dataset_dir = Path(chunked_dataset_dir)
    chunk_mask = [0] * len(chunks_to_process)

    if not any(chunked_dataset_dir.iterdir()):
        # "Directory is empty"
        return chunk_mask

    for chunk_i, chunk in enumerate(chunks_to_process, start=1):
        try:
            shard_pattern = f"raw_shard_{str(chunk_i).zfill(4)}-????.tfrecord"
            chunk_shard_fp = glob.glob(str(chunked_dataset_dir / shard_pattern))

            aux_pattern = f"data_tbl_{str(chunk_i).zfill(4)}-????.csv"
            chunk_aux_tbl_path = glob.glob(str(chunked_dataset_dir / aux_pattern))

            if Path(chunk_shard_fp).exists() and chunk_aux_tbl_path.exists():
                chunk_mask[chunk_i] = 1
        except:
            # chunk not found
            continue

    return chunk_mask
