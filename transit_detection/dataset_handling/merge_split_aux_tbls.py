import pandas pd
import numpy as np
import Path



if __name__ == "__main__":

    aux_tbl_dir = "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split/tfrecords/aux_tbls"

    merged_aux_tbl_path = "/nobackupp27/jochoa4/work_dir/data/datasets/TESS_exoplanet_dataset_11-12-2024_split/tfrecords/"

    aux_tbls = []

    for i in range(1, 8611 + 1):
        aux_tbl_path = aux_tbl_dir / f"shards_tbl_7386-{str(i).zfill(4)}.csv"
        aux_tbl = pd.read_csv(aux_tbl)
        aux_tbls.append(aux_tbl)

    merged_df = pd.concat(aux_tbl, ignore_index=True, axis=0)
    merged_df.to_csv(merged_aux_tbl_path / "merged_split_aux_tbl_0001-8611")