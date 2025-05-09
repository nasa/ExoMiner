# Ephemeris Matching for TESS

## 1. Introduction

The ephemeris matching process consists in using ephemerides (i.e., orbital period, transit epoch and duration) to match 
two transit signals. Based on the orbital period, transit duration, and transit epoch, a train of unitary pulses is 
created within the observation window. The cosine similarity is used as matching score, which is thresholded to decide 
whether two transit signals should be matched. 

## 2. Applications

1. Match TESS TCEs to objects (e.g., TOIs) so we can use the dispositions assigned to these objects to create a catalog 
of labeled TCEs for training and evaluating/benchmarking, for example, transit classification models.
2. Match populations of TCEs for comparison across TCE catalogs.

## 3. Steps

1. **Get timestamps for observed targets**: for each target (i.e., TIC) with associated TCEs, the start and end 
timestamps are extracted from the light curve files for every sector run the target was observed. This is done using 
the script `get_start_end_timestamps_sector_runs.py` and the wrapper script `run_sector_timestamps_multiproc.py` for 
parallelization through multiprocessing.
2. **Run ephemeris matching**: conduct ephemeris matching between two sets of transit signals. Usually, one set are the 
unlabeled TCEs (i.e., with no disposition assigned) and the other set is a catalog of objects with assigned 
dispositions. This step is performed using `ephemeris_matching.py` with configuration file `config_ephem_matching.yaml` 
and the wrapper script `run_ephemeris_matching_multiproc.py` for parallelization through multiprocessing.
3. **Resolve matches**: threshold the matching scores. Furthermore, there can be multiple matches above the matching 
score that need to be resolved for TCEs in the same sector run and for the same target. This can be done using script 
`resolve_matchings.py` and the wrapper script `run_resolve_matchings_multiproc.py` for parallelization through 
multiprocessing. 
