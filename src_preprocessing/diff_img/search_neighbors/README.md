# TIC Neighbors

## Introduction

The goal of this module is to search for known TIC neighbors in the vicinity of the target stars so we can build a neighbors image that gives information to the model about the location and magnitude of neighboring TICs relative to the target.

## Steps

1. Prepare a table of targets to search for neighbors using script [prepare_targets_table_for_search.py](prepare_targets_table_for_search.py).
2. Search for neighbors: using script [search_target_neighbors.py](search_target_neighbors.py)
3. Map the location of the found neighbors from celestial coordinates (RA, Dec) to the pixel frame using script [map_location_neighbors_to_pixel_frame.py](map_location_neighbors_to_pixel_frame.py).