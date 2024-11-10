from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

"""
Used to filter tp sh commands corresponding to all sectors in target, sector_run pairs used by build_dataset,
before downloading with bash scripts
"""


def get_tp_curl_commands(sectors, data_dir, sector_unique_targets):
    """
    Get curl commands for list of sectors from tp sh file
    for unique dict of sector: target in the form:
                                                        "sector_1" : [curl1, curl2, ... curl_n]
                                                        "sector_2" : [],
                                                        ...
                                                        "sector_n" : []
    """
    
    file_paths = [Path(f"{data_dir}/tesscurl_sector_{sector}_tp.sh") for sector in sectors] if sectors else []
    commands = {}
    for sector_n, file_path in zip(sectors, file_paths):
        targets = sector_unique_targets[sector_n]
        sector_commands = []
        with open(file_path, 'r') as file:
            for line in file:
                if 'curl' in line:
                    target = line.split('-')[-3]
                    if target in targets:
                        sector_commands.append(line)
        commands.update({f"{sector_n}" : sector_commands})
    return commands

def write_tp_sh_commands(sector_commands, output_dir):
    """
    Write .sh file for sector to output in the form:
                                                    tesscurl_sector_n_tp.sh
    from commands dict sector_commands in the form:
                                                    "sector_1" : [curl1, curl2, ... curl_n]
                                                    "sector_2" : [],
                                                    ...
                                                    "sector_n" : []
    """
    for sector_n, command_arr in sector_commands.items():
        file_path = Path(f"{output_dir}/tesscurl_sector_{sector_n}_tp.sh")
        command_arr.insert(0, "#!/bin/sh\n") #prepend shebang
        with open(file_path, 'w') as tpsh:
            tpsh.writelines(command_arr)

def process_sector_unique_targets(tce_tbl):
    """
    Create dict of unique targets from tce_tbl for sector in the form:
                                                                      sector_n : set(targetX.zfill(16), ... )
    """
    sector_unique_targets = defaultdict(set)

    for (target, sector_run), _ in tce_tbl.groupby(['target_id','sector_run']):
        if '-' in sector_run:
            start_sector, end_sector = [int(sector) for sector in sector_run.split('-')]
            sector_run_arr = np.arange(start_sector, end_sector + 1)
        else:
            sector_run_arr = [int(sector_run)]
        for sector in sector_run_arr: #would be better to check if its in found_sectors in lc but not neccesary
            sector_unique_targets[sector].add(f"{target}".zfill(16))

    return sector_unique_targets

if __name__ == "__main__":
    TP_SH_DATA_DIR = '/Users/jochoa4/Downloads/tess_targetpixel_data'
    TP_SH_DOWNLOAD_DIR = '/Users/jochoa4/Downloads/transit_detection_test/tp_sh_cleaned'

    Path(TP_SH_DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
    
    SECTOR_LOW, SECTOR_HIGH = 1, 67
    sector_range = np.arange(SECTOR_LOW, SECTOR_HIGH + 1).tolist()

    tce_tbl = pd.read_csv('/Users/jochoa4/Projects/exoplanet_transit_classification/ephemeris_tables/preprocessing_tce_tables/tess_2min_tces_dv_s1-s68_all_msectors_11-29-2023_2157_newlabels_nebs_npcs_bds_ebsntps_to_unks.csv')
    tce_tbl = tce_tbl.loc[tce_tbl['label'].isin(['EB','KP','CP','NTP','NEB','NPC'])] #filter for relevant labels

    sector_unique_targets_dict = process_sector_unique_targets(tce_tbl=tce_tbl)

    curl_commands = get_tp_curl_commands(sectors=sector_range, data_dir=TP_SH_DATA_DIR, sector_unique_targets=sector_unique_targets_dict)

    write_tp_sh_commands(sector_commands=curl_commands, output_dir=TP_SH_DOWNLOAD_DIR)
