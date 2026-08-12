# Ephemeris Matching for TESS

## 1. Introduction

The ephemeris matching process consists in using ephemerides (i.e., orbital period, transit epoch and duration) to match 
two transit signals. Based on the orbital period, transit duration, and transit epoch, a train of unitary pulses is 
created within the observation window (more concretely, the start and end timestamps of the sectors for which the target was observed based on the column `sectors_observed` in the primary transit signals table). The cosine similarity is used as matching score, which is thresholded to decide whether two transit signals should be matched.

- Primary table: the transit signals that one wants to get dispositions for. Typically detected from a pipeline like the SPOC pipeline.
- Secondary table: catalog of transit signals (e.g., EXOFOP TOIs).

Requirements:
- Primary table must contain columns:
    - `target_id` (int): TIC ID
    - `sector_run` (str): sector run where the transit signal originated from (e.g., single sector run S1 -> `1`; multisector run S1-S6 -> `1-6`)
    - `sectors_observed` (str): sectors where the target was observed (either binary format like `00011` or `3_4` for target that was observed in sectors 3 and 4)
    - `uid` (str): unique identifier of the signal in the table

- Secondary table must contain columns:
    - `target_id` (int): TIC ID
    - `uid` (str): unique identifier of the signal in the table

## 2. Applications

1. Match TESS TCEs to objects (e.g., TOIs) so one can use the dispositions assigned to these objects to create a catalog 
of labeled TCEs for training and evaluating/benchmarking, for example, transit classification models.
2. Match populations of TCEs for comparison across TCE catalogs.

## 3. Steps

1. **Get start and end timestamps sectors for which the targets were observed**: for each sector any target was observed, get the start and end timestamps.
2. **Run ephemeris matching**: conduct ephemeris matching between two sets of transit signals. Usually, one set are the 
unlabeled TCEs (i.e., with no disposition assigned) and the other set is a catalog of objects with assigned dispositions. This step is performed using [run_ephemeris_matching.py](./run_ephemeris_matching.py) with configuration file [config_ephem_matching.yaml](./config_ephem_matching.yaml).
    - Outputs:
        - One CSV file per target and sector run showing the matching score between all transit signals in the primary table and the ones in the secondary table for the corresponding target and sector run.
3. **Resolve matches**: threshold the matching scores. Furthermore, there can be multiple matches above the matching 
score that need to be resolved for TCEs in the same sector run and for the same target. This can be done using script 
[run_resolve_matchings.py](./run_resolve_matchings.py)
    - Ouputs: CSV with accepted matches for the primary table to the secondary table.
