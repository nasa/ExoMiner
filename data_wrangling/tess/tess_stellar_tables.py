import pandas as pd
import numpy as np

#%% Compare TIC-8 and CTL catalogs

ticTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/final_target_list_s1-s20-tic8.csv')
ctlTbl = pd.read_csv('/data5/tess_project/Data/Ephemeris_tables/Stellar parameters/TESS/final_target_list_s1-s20-ctl.csv')

stellarColumns = ['[Tmag] [real]', '[Teff] [real]', '[logg] [real]', '[MH] [real]', '[rad] [real]', '[mass] [real]',
                  '[rho] [real]', '[ra] [float]', '[dec] [float]']

# check if there is any target star in CTL with at least one parameter which is not found in TIC-8
for target_i, target in ctlTbl.iterrows():

    targetInTic = ticTbl.loc[ticTbl['[ID] [bigint]'] == target['[ID] [bigint]']]

    if len(targetInTic) == 0:
        raise ValueError('Target in CTL not found in TIC-8.')

    # create NaN boolean arrays for the stellar parameters for the target star in both tables
    nanBoolTic = targetInTic[stellarColumns].isna().values[0]
    nanBoolCtl = target[stellarColumns].isna().values

    idxDiff = np.where(nanBoolCtl != nanBoolTic)

    # both tables share the same NaNs
    if len(idxDiff[0]) == 0:
        continue

    nanInTic = nanBoolTic[idxDiff].sum()
    nanInCtl = len(idxDiff[0]) - nanInTic

    if nanInTic > 0:
        raise ValueError('Stellar parameter missing in TIC-8 but present in CTL.')

    # # it will be 1 for places in which one table is NaN and the other is not
    # xorArr = np.logical_xor(nanBoolCtl, nanBoolTic)
    #
    # if not np.any(xorArr):
    #     continue
