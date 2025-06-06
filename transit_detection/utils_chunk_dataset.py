from pathlib import Path
import glob


def build_chunk_mask(chunks_to_process: list, chunked_dataset_dir: Path):
    """
    Builds mask to exclude chunks that have been successfully processed for an iteration of dataset building
    """
    print(f"Building chunk mask using {chunked_dataset_dir} with {len(chunks_to_process)} chunks")
    chunked_dataset_dir = Path(chunked_dataset_dir)
    chunk_mask = [0] * len(chunks_to_process)

    if not chunked_dataset_dir.exists():
        print(f"Directory does not exist")
        return chunk_mask

    if not any(chunked_dataset_dir.iterdir()):
        print("Directory is empty")
        return chunk_mask

    for chunk_i, chunk in enumerate(chunks_to_process, start=0):
        try:
            
            shard_pattern = f"raw_shard_{str(chunk_i + 1).zfill(4)}-????.tfrecord"
            chunk_shard_search = glob.glob(str(chunked_dataset_dir / shard_pattern)) # Returns [] if no fps matching pattern found

            aux_pattern = f"data_tbl_{str(chunk_i + 1).zfill(4)}-????.csv"
            chunk_aux_tbl_search = glob.glob(str(chunked_dataset_dir / aux_pattern)) # Returns [] if no fps matching pattern found

            if chunk_shard_search and chunk_aux_tbl_search:
                chunk_mask[chunk_i] = 1

        except:
            # chunk not found
            continue

    return chunk_mask
