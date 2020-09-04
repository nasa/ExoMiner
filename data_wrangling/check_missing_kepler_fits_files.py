"""
Script used to create the wget bash scripts to download the missing FITS files for Kepler IDs.
"""

import pandas as pd
import os
import numpy as np
import multiprocessing

#%%

LONG_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131105131"],
    1: ["2009166043257"],
    2: ["2009259160929"],
    3: ["2009350155506"],
    4: ["2010078095331", "2010009091648"],
    5: ["2010174085026"],
    6: ["2010265121752"],
    7: ["2010355172524"],
    8: ["2011073133259"],
    9: ["2011177032512"],
    10: ["2011271113734"],
    11: ["2012004120508"],
    12: ["2012088054726"],
    13: ["2012179063303"],
    14: ["2012277125453"],
    15: ["2013011073258"],
    16: ["2013098041711"],
    17: ["2013131215648"]
}

# Quarter index to filename prefix for short cadence Kepler data.
# Reference: https://archive.stsci.edu/kepler/software/get_kepler.py
SHORT_CADENCE_QUARTER_PREFIXES = {
    0: ["2009131110544"],
    1: ["2009166044711"],
    2: ["2009201121230", "2009231120729", "2009259162342"],
    3: ["2009291181958", "2009322144938", "2009350160919"],
    4: ["2010009094841", "2010019161129", "2010049094358", "2010078100744"],
    5: ["2010111051353", "2010140023957", "2010174090439"],
    6: ["2010203174610", "2010234115140", "2010265121752"],
    7: ["2010296114515", "2010326094124", "2010355172524"],
    8: ["2011024051157", "2011053090032", "2011073133259"],
    9: ["2011116030358", "2011145075126", "2011177032512"],
    10: ["2011208035123", "2011240104155", "2011271113734"],
    11: ["2011303113607", "2011334093404", "2012004120508"],
    12: ["2012032013838", "2012060035710", "2012088054726"],
    13: ["2012121044856", "2012151031540", "2012179063303"],
    14: ["2012211050319", "2012242122129", "2012277125453"],
    15: ["2012310112549", "2012341132017", "2013011073258"],
    16: ["2013017113907", "2013065031647", "2013098041711"],
    17: ["2013121191144", "2013131215648"]
}

#%%  Check missing FITS files sequentially

fitsRootDir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final'
fitsRootDirPfe = '/home6/msaragoc/work_dir/data/Kepler-TESS_exoplanet/FITS_files/Kepler/DR25/dr_25_all_final'

targets = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17_DR25/'
                      'keplerTPS_KSOP2536.csv')['keplerId'].unique()

print('Number of targets: {}'.format(len(targets)))

long_cadence = True

targetsMissingFits = []

quarter_prefixes, cadence_suffix = ((LONG_CADENCE_QUARTER_PREFIXES, "llc")
                                    if long_cadence else
                                    (SHORT_CADENCE_QUARTER_PREFIXES, "slc"))

quarters = sorted(quarter_prefixes.keys())  # Sort quarters chronologically.

injected_group = False

# check the existence of FITS files for each Kepler ID
for target_i, target in enumerate(targets):

    if target_i % 1e4 == 0:
        print('Checking target {} out of {}'.format(target_i, len(targets)))
    # Pad the Kepler id with zeros to length 9.
    targetpadded = "{:09d}".format(int(target))

    targetFilepaths = []
    base_dir = os.path.join(fitsRootDir, targetpadded[0:4], targetpadded)
    for quarter in quarters:
        for quarter_prefix in quarter_prefixes[quarter]:
            if injected_group:
                base_name = "kplr{}-{}_INJECTED-{}_{}.fits".format(
                    targetpadded, quarter_prefix, injected_group, cadence_suffix)
            else:
                base_name = "kplr{}-{}_{}.fits".format(targetpadded, quarter_prefix,
                                                       cadence_suffix)

            targetFilepath = os.path.join(base_dir, base_name)

            # # Not all stars have data for all quarters.
            # if gfile.exists(targetFilepath):
            if os.path.isfile(targetFilepath):
                targetFilepaths.append(targetFilepath)

    if len(targetFilepaths) == 0:  # no FITS files were found for this target
        targetsMissingFits.append(target)

        # write wget command to download the FITS files
        with open(os.path.join(fitsRootDir, 'get_missing_kepler_verified.sh'), 'a') as sh_file:
            getStr = "wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off -R 'index*' -A _llc.fits -P " \
                     "{} http://archive.stsci.edu/pub/kepler/lightcurves/{}/{}/\n".format(os.path.join(fitsRootDir,
                                                                                                       targetpadded[0:4],
                                                                                                       targetpadded),
                                                                                          targetpadded[0:4],
                                                                                          targetpadded)
            sh_file.write(getStr)

        # same but for Pleiades
        with open(os.path.join(fitsRootDir, 'get_missing_kepler_pfe.sh'), 'a') as sh_file:
            getStr = "wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off -R 'index*' -A _llc.fits -P " \
                     "{} http://archive.stsci.edu/pub/kepler/lightcurves/{}/{}/\n".format(os.path.join(fitsRootDirPfe,
                                                                                                       targetpadded[0:4],
                                                                                                       targetpadded),
                                                                                          targetpadded[0:4],
                                                                                          targetpadded)
            sh_file.write(getStr)

# 684 targets missing
print('Number of targets missing: {}'.format(len(targetsMissingFits)))

#%% Split sh script into smaller ones

shFileMain = pd.read_csv(os.path.join(fitsRootDir, 'get_missing_kepler_pfe.sh'), header=None)

numShFiles = 12
shFileSplit = np.array_split(shFileMain, numShFiles)

for df_i in range(len(shFileSplit)):
    shFileSplit[df_i].to_csv(os.path.join(fitsRootDir, 'get_missing_kepler_pfe{}.sh'.format(df_i + 1)), index=False,
                             header=False)

#%% Parallelize checking missing FITS files


def check_targets_fits(targets, fileId, fitsRootDir, cadence_suffix, injected_group):
    """ Check which targets (Kepler IDs) do not have any FITS file available in the `fitsRootDir` directory. That
    directory must follow the structure of the Mikulski Archive for Space Telescopes (MAST).

    :param targets: NumPy array, list of targets.
    :param fileId: int, ID for the file generated for the targets.
    :param fitsRootDir: str, root directory for the FITS files
    :param cadence_suffix: str, either 'llc' or 'slc'
    :param injected_group: bool, True to look at injected Kepler FITS files
    :return:
        targetsMissingFits: list, list of searched targets that no FITS files were found
    """

    # initialize variable to keep targets that no FITS files were found
    targetsMissingFits = []

    print('Checking a total of {} targets in process {}'.format(len(targets), fileId))

    # check the existence of FITS files for each Kepler ID
    for target_i, target in enumerate(targets):

        if target_i % 1e4 == 0:
            print('(Proc {}) Checking target {} out of {}'.format(fileId, target_i, len(targets)))

        # Pad the Kepler id with zeros to length 9.
        targetpadded = "{:09d}".format(int(target))

        targetFilepaths = []
        base_dir = os.path.join(fitsRootDir, targetpadded[0:4], targetpadded)  # folder for the target id
        fitsFound = False
        for quarter in quarters:
            for quarter_prefix in quarter_prefixes[quarter]:
                if injected_group:
                    base_name = "kplr{}-{}_INJECTED-{}_{}.fits".format(
                        targetpadded, quarter_prefix, injected_group, cadence_suffix)
                else:
                    base_name = "kplr{}-{}_{}.fits".format(targetpadded, quarter_prefix,
                                                           cadence_suffix)

                targetFilepath = os.path.join(base_dir, base_name)

                # Not all stars have data for all quarters.
                # if gfile.exists(targetFilepath):
                if os.path.isfile(targetFilepath):
                    targetFilepaths.append(targetFilepath)
                    fitsFound = True

                if fitsFound:
                    break

            if fitsFound:
                break

        if not fitsFound:  # no FITS files were found for this target

            targetsMissingFits.append(target)

            # write wget command to download the FITS files
            with open(os.path.join(fitsRootDir, 'get_missing_kepler_verified{}.sh'.format(fileId)), 'a') as sh_file:
                getStr = "wget -q -nH --cut-dirs=6 -r -l0 -c -N -np -erobots=off -R 'index*' -A _llc.fits -P " \
                         "{} http://archive.stsci.edu/pub/kepler/lightcurves/{}/{}/" \
                         "\n".format(os.path.join(fitsRootDir,
                                                  targetpadded[0:4],
                                                  targetpadded),
                                     targetpadded[0:4],
                                     targetpadded)
                sh_file.write(getStr)

    return targetsMissingFits

# root directory for FITS files
fitsRootDir = '/data5/tess_project/Data/Kepler-Q1-Q17-DR25/dr_25_all_final'

# targets to be verified
targets = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Kepler/TPS_tables/Q1-Q17_DR25/'
                      'keplerTPS_KSOP2536.csv')['keplerId'].unique()

print('Total number of targets to verify: {}'.format(len(targets)))

long_cadence = True
quarter_prefixes, cadence_suffix = ((LONG_CADENCE_QUARTER_PREFIXES, "llc")
                                    if long_cadence else
                                    (SHORT_CADENCE_QUARTER_PREFIXES, "slc"))
quarters = sorted(quarter_prefixes.keys())  # Sort quarters chronologically.
injected_group = False

nProcesses = 8
pool = multiprocessing.Pool(processes=nProcesses)
targetsProcs = np.array_split(targets, nProcesses)
jobs = [(targetsSub, targetsSubi, fitsRootDir, cadence_suffix, injected_group) for targetsSubi, targetsSub in
        enumerate(targetsProcs)]
async_results = [pool.apply_async(check_targets_fits, job) for job in jobs]
pool.close()

targetsMissingFits = []
# Instead of pool.join(), async_result.get() to ensure any exceptions raised by the worker processes are raised here
for async_result in async_results:
    targetsMissingFits.extend(async_result.get())

print('Number of targets missing: {}'.format(len(targetsMissingFits)))
np.save(os.path.join(fitsRootDir, 'kepler_q1-q17dr25_targets_with_missing_fits.npy'),
        np.array(targetsMissingFits, dtype='uint64'))
