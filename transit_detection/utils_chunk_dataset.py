from pathlib import Path

def build_chunk_mask(chunks_to_process, chunked_dataset_dir):
    """
    Builds mask to exclude chunks that have been successfully processed for an iteration of dataset building
    """

    chunk_mask = [0] * len(chunks_to_process)

    if not any(chunked_dataset_dir.iterdir()):
        print('Directory is empty')
        return chunk_mask

    shard_prefix = 'test_shard_0001-'
    aux_tbl_prefix = 'data_tbl_chunk_'
    
    for chunk_i, chunk in enumerate(chunks_to_process):
        try:
            chunk_shard_path = list(chunked_dataset_dir.rglob(f"*{shard_prefix}{str(chunk_i).zfill(4)}"))[0]
            chunk_aux_tbl_path = list(chunked_dataset_dir.rglob(f"*{aux_tbl_prefix}{str(chunk_i).zfill(4)}.csv"))[0]

            if chunk_shard_path.exists() and chunk_aux_tbl_path.exists():
                chunk_mask[chunk_i] = 1
            
        except:
            #chunk not found
            continue

    return chunk_mask