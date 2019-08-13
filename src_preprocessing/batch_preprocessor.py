"""
Auxiliary script uses multiple preprocessing sets of parameters to create different datasets.
"""

# 3rd party
import os
import subprocess
import itertools


if __name__ == '__main__':
    white_options = [True, False]
    gap_options = [True, False]
    glob_bin_options = [4001, 2001, 1001]
    loc_bin_options = [401, 201, 101]

    combination_list = list(itertools.product(white_options, gap_options, [0, 1, 2]))

    for combination in combination_list:
        whitened = combination[0]
        gapped = combination[1]
        num_bins_glob = glob_bin_options[combination[2]]
        num_bins_loc = loc_bin_options[combination[2]]

        command = 'mpiexec python %s/generate_input_records.py %s %s --num_bins_glob %d --num_bins_loc %d' \
                  % (os.path.dirname(__file__), '--whitened' if whitened else '', '--gapped' if gapped else '',
                     num_bins_glob, num_bins_loc)
        p = subprocess.Popen(command, shell=True)
        _, __ = p.communicate()
        _ = p.wait()
