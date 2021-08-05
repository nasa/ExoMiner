"""
Functions used to compute derived planet parameters.
"""

# 3rd party
import matplotlib.pyplot as plt
import numpy as np

EARTH_RADIUS = 6378137  # meters
SUN_RADIUS = 696.30e6  # meters
F_AU = 1.4960e11  # meters
SUN_EFF_TEMP = 5780  # K


def estimate_inclination(b, dor, b_unc=np.nan, dor_unc=np.nan):
    """ Estimate inclination of the planet orbit.

    :param b: impact parameter (deg)
    :param dor: ratio of semi-major axis to stellar radius
    :param b_unc:  impact parameter uncertainty (deg)
    :param dor_unc: ratio of semi-major axis to stellar radius uncertainty
    :return:
        estimated inclination (deg)
        estimated inclination uncertainty (deg)
    """

    incl = 180 / np.pi * np.arccos(b / dor)

    # TODO: TO BE IMPLEMENTED; computation of uncertainty involves non-linear function...
    incl_unc = np.nan

    return incl, incl_unc


def estimate_eff_stellar_flux(st_radius, sma, st_eff_temp, st_logg, st_radius_unc=np.nan,
                              st_eff_temp_unc=np.nan, st_logg_unc=np.nan):
    """ Estimate effective stellar flux (aka insolation flux).

    :param st_radius: stellar radius (Solar Radii)
    :param sma: semi-major axis (AU)
    :param st_eff_temp: stellar effective temperature (K)
    :param st_logg: stellar surface gravity (log10 g); g (cm.s^-2)
    :param st_radius_unc: stellar radius uncertainty (Solar Radii)
    :param sma_unc: semi-major axis uncertainty (AU)
    :param st_eff_temp_unc: stellar effective temperature uncertainty (K)
    :param st_logg: stellar surface gravity uncertainty (log10 g); g (cm.s^-2)
    :return:
        stellar flux
        stellar flux uncertainty
    """

    g = 10 ** st_logg / 100
    g_unc = g * np.log(10) * st_logg_unc

    if st_radius == 0 or sma == 0 or st_eff_temp == 0:
        eff_stellar_flux = 0
    else:
        eff_stellar_flux = (st_radius / sma) ** 2 * (st_eff_temp / SUN_EFF_TEMP) ** 4

    # if np.any(np.isnan([st_radius_unc, sma_unc, st_eff_temp_unc])):
    #     eff_stellar_flux_unc = np.nan
    # else:
    # eff_stellar_flux_unc = np.sqrt((2 * st_radius_unc / st_radius) ** 2 + (2 * sma_unc / sma) ** 2 +
    #                                (4 * st_eff_temp_unc / st_eff_temp) ** 2) * eff_stellar_flux
    if st_radius_unc == -1 or st_eff_temp_unc == -1 or st_logg == 0 or st_logg_unc == -1 or eff_stellar_flux == 0:
        eff_stellar_flux_unc = -1
    else:
        eff_stellar_flux_unc = np.sqrt(((2 / 3) * st_radius_unc / st_radius) ** 2 +
                                       ((2 / 3) * g_unc / g) ** 2 +
                                       (4 * st_eff_temp_unc / st_eff_temp) ** 2) * eff_stellar_flux

    return eff_stellar_flux, eff_stellar_flux_unc


def estimate_plnt_eq_temp(st_eff_temp, st_radius, sma, st_eff_temp_unc=np.nan, st_radius_unc=np.nan,
                          plnt_albedo=0.3, st_logg=np.nan, st_logg_unc=np.nan, sma_unc=np.nan):
    """ Estimate planet equilibrium temperature.

    :param st_eff_temp: stellar effective temperature (K)
    :param st_radius: stellar radius (Solar Radii)
    :param sma: semi-major axis (AU)
    :param st_eff_temp_unc: stellar effective temperature uncertainty (K)
    :param st_radius_unc: stellar radius uncertainty (Solar Radii)
    :param plnt_albedo: planet albedo
    :param st_logg: stellar surface gravity (log10 g)
    :param st_logg_unc: stellar surface gravity uncertainty (log10 g)
    :param sma_unc : semi-major axis uncertainty (AU)
    :return:
        planet equilibrium temperature (K)
        planet equilibrium temperature uncertainty (K)
    """

    g = 10 ** st_logg / 100
    g_unc = g * np.log(10) * st_logg_unc

    if st_eff_temp == 0 or st_radius == 0 or sma == 0:
        plnt_eq_temp = 0
    else:
        plnt_eq_temp = st_eff_temp * (1 - plnt_albedo) ** 0.25 * np.sqrt((st_radius * SUN_RADIUS) / (2 * sma * F_AU))

    # if np.any(np.isnan([st_radius_unc, sma_unc, plnt_eq_temp])):
    #     plnt_eq_temp_unc = np.nan
    # else
    if st_eff_temp_unc == -1 or st_radius_unc == -1 or st_logg == 0 or st_logg_unc == -1 or sma_unc == -1 or \
            plnt_eq_temp == 0:
        plnt_eq_temp_unc = -1
    else:
        plnt_eq_temp_unc = np.sqrt((st_eff_temp_unc / st_eff_temp) ** 2 +
                                   ((st_radius_unc / (6 * st_radius)) ** 2 +
                                    (g_unc / (6 * g)) ** 2)) * plnt_eq_temp

    return plnt_eq_temp, plnt_eq_temp_unc


def estimate_sma(period, st_radius, st_logg, period_unc=np.nan, st_radius_unc=np.nan, st_logg_unc=np.nan):
    """ Estimate semi-major axis.

    :param period: orbital period (day)
    :param st_radius: stellar radius (Solar Radii)
    :param st_logg: stellar surface gravity (log10 g)
    :param period_unc: orbital period uncertainty (day)
    :param st_radius_unc: stellar radius uncertainty (Solar Radii)
    :param st_logg_unc: stellar surface gravity uncertainty (log10 g)
    :return:
        semi-major axis (AU)
        semi-major axis  uncertainty (AU)
    """

    g = 10 ** st_logg / 100
    g_unc = g * np.log(10) * st_logg_unc

    if st_logg_unc == -1 or st_logg == 0 or period == 0 or st_radius == 0:
        sma = 0
    else:
        sma = ((86400 * period * st_radius * SUN_RADIUS * np.sqrt(g)) / (2 * np.pi)) ** (2 / 3) / F_AU

    # if np.any(np.isnan([period, st_radius, st_logg, sma])):
    #     sma_unc = np.nan
    # else:
    if period_unc == -1 or st_radius_unc == -1 or st_logg == 0 or st_logg_unc == -1 or sma == 0:
        sma_unc = -1
    else:
        sma_unc = np.sqrt(((2 / 3) * period_unc / period) ** 2 + ((2 / 3) * st_radius_unc / st_radius) ** 2 +
                          (g_unc / (3 * g)) ** 2) * sma

    return sma, sma_unc


def estimate_plnt_radius(ror, st_radius, ror_unc=np.nan, st_radius_unc=np.nan):
    """ Estimate planet radius.

    :param ror: planet-star radius ratio
    :param st_radius: stellar radius (Solar Radii)
    :param ror_unc: planet-star radius ratio uncertainty
    :param st_radius_unc: stellar radius uncertainty (Solar Radii)
    :return:
        planet radius (Earth Radii)
        planet radius uncertainty (Earth Radii)
    """

    if ror == 0 or st_radius == 0:
        plnt_radius = 0
    else:
        plnt_radius = (SUN_RADIUS / EARTH_RADIUS) * ror * st_radius

    # plnt_radius_new = plnt_radius * st_radius_new / st_radius

    # if np.any(np.isnan([ror_unc, st_radius_unc, plnt_radius])):
    if ror_unc == -1 or st_radius_unc == -1 or plnt_radius == 0:
        plnt_radius_unc = -1
    else:
        plnt_radius_unc = np.sqrt((ror_unc / ror) ** 2 + (st_radius_unc / st_radius) ** 2) * plnt_radius

    return plnt_radius, plnt_radius_unc


def estimate_sec_geo_albedo(sec_tr_depth, plnt_radius, sma, sec_tr_depth_unc=np.nan, plnt_radius_unc=np.nan,
                            sma_unc=np.nan):
    """ Estimate secondary geometric albedo.

    :param sec_tr_depth: fractional secondary transit depth (ppm)
    :param plnt_radius: planet radius, in Earth radii
    :param sma: semi-major axis, in AU
    :param sec_tr_depth_unc: fractional secondary transit depth uncertainty (ppm)
    :param plnt_radius_unc: planet radius uncertainty (ppm)
    :param sma_unc: semi-major axis uncertainty (AU)
    :return:
        secondary geometric albedo
        secondary geometric albedo uncertainty
    """

    if sec_tr_depth <= 0 or sma == 0 or plnt_radius == 0:
        sg_albedo = 0
    else:
        sg_albedo = sec_tr_depth * 1e-6 * (sma * F_AU / (plnt_radius * EARTH_RADIUS)) ** 2

    # if np.any(np.isnan([tr_depth_unc, plnt_radius_unc, sma_unc, sg_albedo])):
    #     sg_albedo_unc = np.nan
    # else:
    if sec_tr_depth_unc == -1 or sma_unc == -1 or plnt_radius_unc == -1 or sg_albedo == 0:
        sg_albedo_unc = -1
    else:
        sg_albedo_unc = sg_albedo * np.sqrt((sec_tr_depth_unc / sec_tr_depth) ** 2 +
                                            (2 * sma_unc / sma) ** 2 +
                                            (2 * plnt_radius_unc / plnt_radius) ** 2)

    return sg_albedo, sg_albedo_unc


def estimate_new_sec_geo_albedo(sg_albedo, plnt_radius, sma, plnt_radius_new, sma_new, sg_albedo_unc=np.nan,
                                plnt_radius_unc=np.nan, sma_unc=np.nan, plnt_radius_unc_new=np.nan, sma_unc_new=np.nan):
    """ Update secondary geometric albedo with new values for the planet radius and semi-major axis.

    :param sg_albedo: old secondary geometric albedo
    :param plnt_radius: planet radius, in Earth radii
    :param sma: semi-major axis, in AU
    :param plnt_radius_new: new planet radius, in Earth radii
    :param sma_new: new semi-major axis, in AU
    :param sg_albedo_unc: uncertainty for the old secondary geometric albedo
    :param plnt_radius_unc: planet radius uncertainty (ppm)
    :param sma_unc: semi-major axis uncertainty (AU)
    :param plnt_radius_unc_new: uncertainty for the new planet radius (Earth radii)
    :param sma_unc_new: uncertainty for the new semi-major axis (AU)
    :return:
        new secondary geometric albedo
        new secondary geometric albedo uncertainty
    """

    # print(sg_albedo, plnt_radius, sma, plnt_radius_new, sma_new, sg_albedo_unc,
    #                             plnt_radius_unc, sma_unc, plnt_radius_unc_new, sma_unc_new)
    # aaaaa

    if sg_albedo == 0 or plnt_radius == 0 or sma == 0 or sma_new == 0 or plnt_radius_new == 0:
        sg_albedo_new = 0
    else:
        sg_albedo_new = sg_albedo * (plnt_radius / sma) ** 2 * (sma_new / plnt_radius_new) ** 2

    # if np.any(np.isnan([tr_depth_unc, plnt_radius_unc, sma_unc, sg_albedo])):
    #     sg_albedo_unc = np.nan
    # else:
    if sg_albedo_unc == -1 or sma_unc == -1 or plnt_radius_unc == -1 or sma_unc_new == -1 or \
            plnt_radius_unc_new == -1 or sg_albedo_new == 0:
        sg_albedo_unc_new = -1
    else:
        sg_albedo_unc_new = sg_albedo_new * np.sqrt((sg_albedo_unc / sg_albedo) ** 2 -
                                                    (2 * sma_unc / sma) ** 2 -
                                                    (2 * plnt_radius_unc / plnt_radius) ** 2 +
                                                    (2 * sma_unc_new / sma_new) ** 2 +
                                                    (2 * plnt_radius_unc_new / plnt_radius_new) ** 2)
        if np.isnan(sg_albedo_unc_new):
            sg_albedo_unc_new = -1

    return sg_albedo_new, sg_albedo_unc_new


def estimate_plnt_eff_temp(st_eff_temp, sec_tr_depth, ror, st_eff_temp_unc=np.nan, sec_tr_depth_unc=np.nan,
                           ror_unc=np.nan):
    """ Estimate planet effective temperature.

    :param st_eff_temp: stellar effective temperature (K)
    :param sec_tr_depth: secondary transit depth (ppm)
    :param ror: planet-stellar radius ratio
    :param st_eff_temp_unc: stellar effective temperature uncertainty (K)
    :param sec_tr_depth_unc: secondary transit depth uncertainty (ppm)
    :param ror_unc: planet-stellar radius ratio uncertainty
    :return:
        planet effective temperature (K)
        planet effective temperature uncertainty (K)
    """

    if st_eff_temp == 0 or sec_tr_depth <= 0 or ror == 0:
        plnt_eff_temp = 0
    else:
        plnt_eff_temp = st_eff_temp * (sec_tr_depth * 1e-6) ** 0.25 * ror ** (-0.5)

    # if np.any(np.isna([tr_depth_unc, st_eff_temp_unc, ror_unc, p_efftemp])):
    #     p_efftemp_unc = np.nan
    # else:
    if st_eff_temp_unc == -1 or sec_tr_depth_unc == -1 or ror_unc == -1 or plnt_eff_temp == 0:
        plnt_eff_temp_unc = -1
    else:
        plnt_eff_temp_unc = plnt_eff_temp * np.sqrt((st_eff_temp_unc / st_eff_temp) ** 2 +
                                                    (sec_tr_depth_unc / (4 * sec_tr_depth) ** 2 +
                                                     (ror_unc / (2 * ror)) ** 2))

    return plnt_eff_temp, plnt_eff_temp_unc


def estimate_new_plnt_eff_temp(plnt_eff_temp, st_eff_temp, st_eff_temp_new, plnt_eff_temp_unc=np.nan,
                               st_eff_temp_unc=np.nan, st_eff_temp_unc_new=np.nan):
    """ Update planet effective temperature with new values for the stellar effective temperature.

    :param plnt_eff_temp: old planet effective temperature (K)
    :param st_eff_temp: old stellar effective temperature (K)
    :param st_eff_temp_new: new stellar effective temperature (K)
    :param plnt_eff_temp_unc: old planet effective temperature uncertainty (K)
    :param st_eff_temp_unc: old stellar effective temperature uncertainty (K)
    :param  st_eff_temp_unc_new: new stellar effective temperature uncertainty (K)
    :return:
        planet effective temperature (K)
        planet effective temperature uncertainty (K)
    """

    if plnt_eff_temp == 0 or st_eff_temp_new == 0 or st_eff_temp == 0:
        plnt_eff_temp_new = 0
    else:
        plnt_eff_temp_new = plnt_eff_temp * st_eff_temp_new / st_eff_temp

    # if np.any(np.isna([tr_depth_unc, st_eff_temp_unc, ror_unc, p_efftemp])):
    #     p_efftemp_unc = np.nan
    # else:
    if plnt_eff_temp_unc == -1 or st_eff_temp_unc == -1 or st_eff_temp_unc_new == -1 or plnt_eff_temp_new == 0:
        plnt_eff_temp_unc_new = -1
    else:
        plnt_eff_temp_unc_new = plnt_eff_temp_new * np.sqrt((plnt_eff_temp_unc / plnt_eff_temp) ** 2 -
                                                            (st_eff_temp_unc / st_eff_temp) ** 2 +
                                                            (st_eff_temp_unc_new / st_eff_temp_new) ** 2
                                                            )
        if np.isnan(plnt_eff_temp_unc_new):
            plnt_eff_temp_unc_new = -1

    return plnt_eff_temp_new, plnt_eff_temp_unc_new


def compute_sec_geo_albedo_stat(sg_albedo, sg_albedo_unc):
    """ Compute secondary geometric albedo comparison statistic.

    :param sg_albedo: secondary geometric albedo
    :param sg_albedo_unc: secondary geometric albedo uncertainty
    :return:
        secondary geometric albedo comparison statistic
    """
    if sg_albedo == 0 or sg_albedo_unc == -1:
        sg_albedo_stat = 0
    else:
        sg_albedo_stat = (sg_albedo - 1) / sg_albedo_unc

    return sg_albedo_stat


def compute_plnt_eff_temp_stat(plnt_eff_temp, plnt_eq_temp, plnt_eff_temp_unc, plnt_eq_temp_unc):
    """ Compute effective planet temperature comparison statistic.

    :param plnt_eff_temp: planet effective temperature (K)
    :param plnt_eq_temp: planet equilibrium temperature (K)
    :param plnt_eff_temp_unc: planet effective temperature uncertainty (K)
    :param plnt_eq_temp_unc: planet equilibrium temperature uncertainty (K)
    :return:
        planet effective temperature comparison statistic
    """
    if plnt_eq_temp_unc == -1 or plnt_eff_temp_unc == -1 or plnt_eff_temp == 0 or plnt_eq_temp == 0:
        plnt_eff_temp_stat = 0
    else:
        plnt_eff_temp_stat = (plnt_eff_temp - plnt_eq_temp) / np.sqrt(plnt_eff_temp_unc ** 2 + plnt_eq_temp_unc ** 2)

    return plnt_eff_temp_stat


def check_valid_sma(sma, sma_unc, st_radius, plnt_radius):
    """ Check if semi-major axis is valid (larger or equal to sum of planet radius and stellar radius).

    :param sma: semi-major axis (AU)
    :param sma_unc: semi-major axis uncertainty (AU)
    :param st_radius: stellar radius (Solar radii)
    :param plnt_radius: stellar radius uncertainty (Earth radii)
    :return:
        semi-major axis (AU)
        semi-major axis uncertainty (AU)
    """

    if sma * F_AU < st_radius * SUN_RADIUS + plnt_radius * EARTH_RADIUS:
        return 0, -1
    else:
        return sma, sma_unc


def check_non_physical_parameter(st_radius, plnt_radius, sma, plnt_eq_temp, eff_stellar_flux, sma_unc=np.nan,
                                 plnt_eq_temp_unc=np.nan, eff_stellar_flux_unc=np.nan):
    """ Set semi-major axis, planet eq. temp. and effective stellar flux to  undefined if semi-major axis is smaller
     than the sum of planet radius and stellar radius.

    :param st_radius: stellar radius (Solar radii)
    :param plnt_radius: planet radius (Earth radii)
    :param sma: semi-major axis (AU)
    :param plnt_eq_temp: planet eq. temp. (K)
    :param eff_stellar_flux: effective stellar flux
    :param sma_unc: semi-major axis uncertainty (AU)
    :param plnt_eq_temp_unc: planet eq. temp. uncertainty (K)
    :param eff_stellar_flux_unc: effective stellar flux uncertainty
    :return:
        semi-major axis (AU)
        semi-major axis uncertainty (AU)
        planet eq. temp. (K)
        planet eq. temp. uncertainty (K)
        effective stellar flux
        effective stellar flux uncertainty
    """

    below_thr = sma * F_AU < st_radius * SUN_RADIUS + plnt_radius * EARTH_RADIUS

    if below_thr:
        return 0, -1, 0, -1, 0, -1
    else:
        return sma, sma_unc, plnt_eq_temp, plnt_eq_temp_unc, eff_stellar_flux, eff_stellar_flux_unc


def check_valid_plnt_eq_temp(plnt_eq_temp, plnt_eq_temp_unc, st_eff_temp):
    """ Check if planet equilibrium temperature is valid, i.e., smaller than the stellar effective temperature.

    :param plnt_eq_temp: planet equilibrium temp. (K)
    :param plnt_eq_temp_unc: planet equilibrium temp. uncertainty (K)
    :param st_eff_temp: stellar effective temperature (K)
    :return:
        planet eq. temp. (K)
        planet eq. temp. uncertainty (K)
    """

    if plnt_eq_temp >= st_eff_temp:

        return 0, -1

    else:
        return plnt_eq_temp, plnt_eq_temp_unc


def check_valid_sec_geo_albedo(sg_albedo, sg_albedo_unc, sma, sma_unc, plnt_radius, st_radius, plnt_radius_unc,
                               sec_tr_depth):
    """ Check if secondary geometric albedo is valid. Semi-major axis must be larger or equal to sum of planet radius
    and stellar radius, semi-major axis and planet radius uncertainties must be defined, and secondary transit depth
    must be positive.

    :param sg_albedo: secondary geometric albedo
    :param sg_albedo_unc: secondary geometric albedo uncertainty
    :param sma: semi-major axis (AU)
    :param sma_unc: semi-major axis uncertainty (AU)
    :param plnt_radius: planet radius (Earth radii)
    :param st_radius: stellar radius (Solar radii)
    :param plnt_radius_unc: planet radius uncertainty (Earth radii)
    :param sec_tr_depth: secondary transit depth (ppm)
    :return:
        secondary geometric albedo
        secondary geometric albedo uncertainty
    """

    if sma * F_AU < plnt_radius * EARTH_RADIUS + st_radius * SUN_RADIUS or sma_unc == -1 or plnt_radius_unc == -1 \
            or sec_tr_depth <= 0:
        return 0, -1
    else:
        return sg_albedo, sg_albedo_unc


if __name__ == '__main__':

    import pandas as pd
    from pathlib import Path

    # import matplotlib.pyplot as plt

    res_dir = Path('/home/msaragoc/Projects/Kepler-TESS_exoplanet/Analysis/q1_q17_dr25_tce_planet_params/')

    tce_tbl_fp = res_dir / 'q1_q17_dr25_tce_2021.05.13_15.04.35.csv'
    tce_tbl = pd.read_csv(tce_tbl_fp, header=67)

    # compute planet radius
    tce_tbl[['tce_prad', 'tce_prad_err']] = \
        tce_tbl.apply(
            lambda x: estimate_plnt_radius(x['tce_ror'], x['tce_sradius'], x['tce_ror_err'], x['tce_sradius_err'],
                                           ), axis=1, result_type='expand')

    # compute semi-major axis
    tce_tbl[['tce_sma', 'tce_sma_err']] = \
        tce_tbl.apply(
            lambda x: estimate_sma(x['tce_period'], x['tce_sradius'], x['tce_slogg'], x['tce_period_err'],
                                   x['tce_sradius_err'], x['tce_slogg_err']), axis=1, result_type='expand')

    # check if semi-major axis is larger than stellar radius plus planet radius
    # tce_tbl[['tce_sma_new_noupdate', 'tce_sma_err_new_noupdate']] = tce_tbl[['tce_sma_new', 'tce_sma_err_new']]
    tce_tbl[['tce_sma', 'tce_sma_err']] = \
        tce_tbl.apply(
            lambda x: check_valid_sma(x['tce_sma'], x['tce_sma_err_new'], x['tce_sradius'], x['tce_prad_new']),
            axis=1, result_type='expand')

    # compute insolation flux
    tce_tbl[['tce_insol_new', 'tce_insol_err_new']] = \
        tce_tbl.apply(
            lambda x: estimate_eff_stellar_flux(x['tce_sradius'], x['tce_sma'], x['tce_steff'], x['tce_slogg'],
                                                x['tce_sradius_err'], x['tce_steff_err'], x['tce_slogg_err'],
                                                ), axis=1, result_type='expand')

    # compute planet equilibrium temperature
    tce_tbl[['tce_eqt_new', 'tce_eqt_err_new']] = \
        tce_tbl.apply(
            lambda x: estimate_plnt_eq_temp(x['tce_steff'], x['tce_sradius'], x['tce_sma'], x['tce_steff_err'],
                                            x['tce_sradius_err'], 0.3, x['tce_slogg'], x['tce_slogg_err'],
                                            x['tce_sma_err']),
            axis=1, result_type='expand')

    # check if planet equilibrium temperature is smaller than stellr effective temperature
    tce_tbl[['tce_eqt_new', 'tce_eqt_err_new']] = \
        tce_tbl.apply(
            lambda x: check_valid_plnt_eq_temp(x['tce_eqt_new'], x['tce_eqt_err_new'], x['tce_steff']),
            axis=1, result_type='expand')

    # compute planet effective temperature
    # tce_tbl[['tce_ptemp_new', 'tce_ptemp_err_new']] = \
    #     tce_tbl[['tce_steff', 'wst_depth', 'tce_ror', 'tce_steff_err', 'tce_ror_err']].apply(
    #         lambda x: estimate_plnt_eff_temp(x['tce_steff'], x['wst_depth'], x['tce_ror'], x['tce_steff_err'],
    #                                np.nan, x['tce_ror_err']), axis=1, result_type='expand')
    for col in ['tce_ptemp_new', 'tce_ptemp_err_new']:
        tce_tbl[col] = np.nan
    for tce_i, tce in tce_tbl.iterrows():
        # try:
        tce_tbl.loc[tce_i, ['tce_ptemp_new', 'tce_ptemp_err_new']] = \
            estimate_new_plnt_eff_temp(
                tce['tce_ptemp'],
                tce['tce_steff'],
                tce['tce_steff'],
                tce['tce_ptemp_err'],
                tce['tce_steff_err'],
                tce['tce_steff_err'],
            )
        # except:
        #     continue

    # compute planet effective temperature comparison stat
    tce_tbl[['tce_ptemp_stat_new']] = \
        tce_tbl.apply(
            lambda x: compute_plnt_eff_temp_stat(x['tce_ptemp'], x['tce_eqt'], x['tce_ptemp_err'], x['tce_eqt_err']),
            axis=1, result_type='expand')

    # compute albedo
    # tce_tbl[['tce_albedo_new', 'tce_albedo_err_new']] = \
    #     tce_tbl[['tce_albedo', 'tce_prad', 'tce_sma', 'tce_prad', 'tce_sma',
    #              'tce_albedo_err', 'tce_prad_err', 'tce_sma_err', 'tce_prad_err', 'tce_sma_err']].apply(
    #         lambda x: estimate_new_sec_geo_albedo(x['tce_albedo'],
    #                                               x['tce_prad'], x['tce_sma'],
    #                                               x['tce_prad'], x['tce_sma'],
    #                                               x['tce_albedo_err'],
    #                                               x['tce_prad_err'], x['tce_sma_err'],
    #                                               x['tce_prad_err'], x['tce_sma_err']),
    #         axis=1, result_type='expand')
    for col in ['tce_albedo_new', 'tce_albedo_err_new']:
        tce_tbl[col] = np.nan
    for tce_i, tce in tce_tbl.iterrows():
        # try:
        tce_tbl.loc[tce_i, ['tce_albedo_new', 'tce_albedo_err_new']] = \
            estimate_new_sec_geo_albedo(
                tce['tce_albedo'],
                tce['tce_prad'], tce['tce_sma'],
                tce['tce_prad'], tce['tce_sma'],
                tce['tce_albedo_err'],
                tce['tce_prad_err'], tce['tce_sma_err'],
                tce['tce_prad_err'], tce['tce_sma_err'])
        # except:
        #     continue

    # check if planet equilibrium temperature is smaller than stellr effective temperature
    tce_tbl[['tce_albedo_new', 'tce_albedo_err_new']] = \
        tce_tbl.apply(
            lambda x: check_valid_sec_geo_albedo(x['tce_albedo_new'], x['tce_albedo_err_new'], x['tce_sma_new'],
                                                 x['tce_sma_err_new'], x['tce_prad_new'], x['tce_sradius'],
                                                 x['tce_prad_err_new'], tce['wst_depth']),
            axis=1, result_type='expand')

    # compute albedo comparison stat
    tce_tbl[['tce_albedo_stat_new']] = \
        tce_tbl.apply(
            lambda x: compute_sec_geo_albedo_stat(x['tce_albedo'], x['tce_albedo_err']), axis=1, result_type='expand')

    final_cols = [
        'kepid',
        'tce_plnt_num',
        'tce_sma',
        'tce_sma_new',
        'tce_sma_err',
        'tce_sma_err_new',
        'tce_prad',
        'tce_prad_new',
        'tce_prad_err',
        'tce_prad_err_new',
        'tce_insol',
        'tce_insol_new',
        'tce_insol_err',
        'tce_insol_err_new',
        'tce_eqt',
        'tce_eqt_new',
        'tce_eqt_err',
        'tce_eqt_err_new',
        'tce_ptemp',
        'tce_ptemp_new',
        'tce_ptemp_err',
        'tce_ptemp_err_new',
        'tce_ptemp_stat',
        'tce_ptemp_stat_new',
        'tce_albedo',
        'tce_albedo_new',
        'tce_albedo_err',
        'tce_albedo_err_new',
        'tce_albedo_stat',
        'tce_albedo_stat_new',
    ]

    tce_tbl[final_cols].to_csv(res_dir / f'{tce_tbl_fp.stem}_new.csv', index=False)

    plot_cols = [
        'tce_sma',
        'tce_sma_err',
        'tce_insol',
        'tce_insol_err',
        'tce_prad',
        'tce_prad_err',
        'tce_eqt',
        'tce_eqt_err',
        'tce_ptemp',
        'tce_ptemp_err',
        'tce_ptemp_stat',
        'tce_albedo',
        'tce_albedo_err',
        'tce_albedo_stat',
    ]

    tce_tbl_aux = tce_tbl.copy(deep=True)
    # cols_to_update_sma = [
    #     'tce_sma_new',
    #     'tce_sma_err_new',
    #     'tce_eqt_new',
    #     'tce_eqt_err_new',
    #     # 'tce_ptemp_new',
    #     # 'tce_ptemp_err_new',
    #     'tce_ptemp_stat_new',
    #     'tce_albedo_new',
    #     'tce_albedo_err_new',
    #     'tce_albedo_stat_new'
    # ]
    # cols_to_update_fit_fail = [
    #     'tce_prad_new',
    #     'tce_prad_err_new',
    #     'tce_sma_new',
    #     'tce_sma_err_new',
    #     'tce_eqt_new',
    #     'tce_eqt_err_new',
    #     'tce_ptemp_new',
    #     'tce_ptemp_err_new',
    #     'tce_ptemp_stat_new',
    #     'tce_albedo_new',
    #     'tce_albedo_err_new',
    #     'tce_albedo_stat_new'
    # ]
    # for col in cols_to_update_sma:
    #     tce_tbl_aux[f'{col}_noupdate'] = tce_tbl_aux[col]
    # for tce_i, tce in tce_tbl_aux.iterrows():
    #     if tce['tce_sma_new'] * F_AU / SUN_RADIUS < tce['tce_sradius'] + tce['tce_prad'] * EARTH_RADIUS / SUN_RADIUS:
    #         tce_tbl_aux.loc[tce_i, cols_to_update_sma] = [0, -1, 0, -1, 0, 0, -1, 0]

    # # set geometric albedo to 0 if sma is smaller than plnt_radius + stellar_radius, and sma and plnt_radius
    # # uncertainties are -1 or if weak secondary transit depth is non-positive
    # for tce_i, tce in tce_tbl_aux.iterrows():
    #     if (tce['tce_sma_new'] * F_AU / SUN_RADIUS < tce['tce_sradius'] + tce['tce_prad'] * EARTH_RADIUS / SUN_RADIUS \
    #             and tce['tce_sma_err_new'] == -1 and tce['tce_prad_err'] == -1) or tce['wst_depth'] <= 0:
    #         tce_tbl_aux.loc[tce_i, ['tce_albedo_new', 'tce_albedo_err_new']] = [0, -1]

    # aaa
    # if tce['tce_full_conv'] == 0:
    #     tce_tbl_aux.loc[tce_i, cols_to_update_fit_fail] = [0, -1, 0, -1, 0, -1, 0, -1, 0, 0, -1, 0]

    for col in plot_cols:
        f, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        ax[0].scatter(tce_tbl_aux[col], tce_tbl_aux[f'{col}_new'], c='b', s=8)
        # ax[0].set_xlabel(col)
        ax[0].set_ylabel(f'{col} new')
        ax[1].scatter(tce_tbl_aux[col], np.abs(tce_tbl_aux[f'{col}'] - tce_tbl_aux[f'{col}_new']) /
                      tce_tbl_aux[f'{col}'], c='b', s=8)
        ax[1].set_xlabel(col)
        ax[1].set_ylabel(f'Relative error {col}')
        if col == 'tce_prad':
            ax[0].set_xlim([0, 10])
            ax[0].set_ylim([0, 10])
            ax[1].set_xlim([0, 10])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_prad_err':
            ax[0].set_xlim([-1, 10])
            ax[0].set_ylim([-1, 10])
            ax[1].set_xlim([-1, 10])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_ptemp':
            ax[0].set_xlim([0, 1e4])
            ax[0].set_ylim([0, 1e4])
            ax[1].set_xlim([0, 1e4])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_ptemp_err':
            ax[0].set_xlim([-1, 1e4])
            ax[0].set_ylim([-1, 1e4])
            ax[1].set_xlim([-1, 1e4])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_eqt':
            ax[0].set_xlim([0, 1e4])
            ax[0].set_ylim([0, 1e4])
            ax[1].set_xlim([0, 1e4])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_eqt_err':
            ax[0].set_xlim([-1, 1e4])
            ax[0].set_ylim([-1, 1e4])
            ax[1].set_xlim([-1, 1e4])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_albedo':
            ax[0].set_xlim([0, 1e4])
            ax[0].set_ylim([0, 1e4])
            ax[1].set_xlim([0, 1e4])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_albedo_err':
            ax[0].set_xlim([-1, 1e4])
            ax[0].set_ylim([-1, 1e4])
            ax[1].set_xlim([-1, 1e4])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_albedo_stat':
            ax[0].set_xlim([-1e2, 1e2])
            ax[0].set_ylim([-1e2, 1e2])
            ax[1].set_xlim([-1e2, 1e2])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_ptemp_stat':
            ax[0].set_xlim([-20, 1e2])
            ax[0].set_ylim([-20, 1e2])
            ax[1].set_xlim([-20, 1e2])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_sma_err':
            ax[0].set_xlim([-1, 1])
            ax[0].set_ylim([-1, 1])
            ax[1].set_xlim([-1, 1])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_insol':
            ax[0].set_xlim([0, 2000])
            ax[0].set_ylim([0, 2000])
            ax[1].set_xlim([0, 2000])
            ax[1].set_ylim([0, 1])
        elif col == 'tce_insol_err':
            ax[0].set_xlim([-1, 200])
            ax[0].set_ylim([-1, 2000])
            ax[1].set_xlim([-1, 2000])
            ax[1].set_ylim([0, 1])
        ax[0].grid(True)
        ax[1].grid(True)
        f.savefig(res_dir / f'scatter_{col}.png')
        plt.close(f)

    # a = tce_tbl_aux.loc[((tce_tbl_aux['tce_sma'] == 0) & tce_tbl_aux['tce_sma_new'] != 0) | (
    #             (tce_tbl_aux['tce_sma'] != 0) & (tce_tbl_aux['tce_sma_new'] == 0))]
    # a['tce_sma_solarradii'] = np.nan
    # a['tce_sma_new_solarradii'] = np.nan
    # a['thr_solarradii'] = np.nan
    # for i, el in a.iterrows():
    #     a.loc[i, 'tce_sma_solarradii'] = el['tce_sma'] * F_AU / SUN_RADIUS
    #     a.loc[i, 'tce_sma_new_solarradii'] = el['tce_sma_new_noupdate'] * F_AU / SUN_RADIUS
    #     a.loc[i, 'thr_solarradii'] = el['tce_sradius'] + el['tce_prad'] * EARTH_RADIUS / SUN_RADIUS
    # a[['kepid', 'tce_plnt_num', 'tce_period', 'tce_period_err', 'tce_ror', 'tce_ror_err', 'tce_depth', 'tce_depth_err', 'tce_sradius', 'tce_sradius_err', 'tce_slogg',
    #    'tce_slogg_err', 'tce_prad', 'tce_prad_err', 'tce_sma', 'tce_sma_err', 'tce_sma_new', 'tce_sma_err_new',
    #    'tce_sma_new_noupdate', 'tce_sma_err_new_noupdate', 'tce_sma_solarradii', 'tce_sma_new_solarradii',
    #    'thr_solarradii']].to_csv('/home/msaragoc/Downloads/kepler_q1q17_dr25_tce_sma_lower_than_thr_5-21-2020.csv',
    #                              index=False)
    #
    # a = tce_tbl_aux.loc[((tce_tbl_aux['tce_eqt'] == 0) & tce_tbl_aux['tce_eqt_new'] != 0) | (
    #             (tce_tbl_aux['tce_eqt'] != 0) & (tce_tbl_aux['tce_eqt_new'] == 0))]
    # a[['kepid', 'tce_plnt_num', 'tce_period', 'tce_period_err', 'tce_sradius', 'tce_sradius_err', 'tce_steff',
    #    'tce_steff', 'tce_slogg', 'tce_slogg_err', 'tce_sma', 'tce_sma_err', 'tce_sma_new', 'tce_sma_err_new',
    #    'tce_eqt', 'tce_eqt_err', 'tce_eqt_new', 'tce_eqt_err_new',
    #    'tce_sma_new_noupdate', 'tce_sma_err_new_noupdate',
    #    ]].to_csv('/home/msaragoc/Downloads/kepler_q1q17_dr25_tce_eqt_zero_5-21-2020.csv', index=False)
    #
    # # a = tce_tbl_aux.loc[((tce_tbl_aux['tce_eqt_err'] == -1) & tce_tbl_aux['tce_eqt_err_new'] != -1) | (
    # #             (tce_tbl_aux['tce_eqt_err'] != -1) & (tce_tbl_aux['tce_eqt_err_new'] == -1))]
    # a = tce_tbl_aux.loc[((tce_tbl_aux['tce_eqt_err'] == -1) & (tce_tbl_aux['tce_eqt_err_new'] != -1)) | (
    #             (tce_tbl_aux['tce_eqt_err'] != -1) & (tce_tbl_aux['tce_eqt_err_new'] == -1))]
    # a['tce_sma_solarradii'] = np.nan
    # a['tce_sma_new_solarradii'] = np.nan
    # a['thr_solarradii'] = np.nan
    # for i, el in a.iterrows():
    #     a.loc[i, 'tce_sma_solarradii'] = el['tce_sma'] * F_AU / SUN_RADIUS
    #     a.loc[i, 'tce_sma_new_solarradii'] = el['tce_sma_new_noupdate'] * F_AU / SUN_RADIUS
    #     a.loc[i, 'thr_solarradii'] = el['tce_sradius'] + el['tce_prad'] * EARTH_RADIUS / SUN_RADIUS
    # a[['kepid', 'tce_plnt_num', 'tce_period', 'tce_period_err', 'tce_sradius', 'tce_sradius_err', 'tce_steff',
    #    'tce_steff', 'tce_slogg', 'tce_slogg_err', 'tce_sma', 'tce_sma_err', 'tce_sma_new', 'tce_sma_err_new',
    #    'tce_eqt', 'tce_eqt_err', 'tce_eqt_new', 'tce_eqt_err_new',
    #    'tce_sma_new_noupdate', 'tce_sma_err_new_noupdate', 'tce_sma_solarradii', 'tce_sma_new_solarradii', 'thr_solarradii'
    #    ]].to_csv('/home/msaragoc/Downloads/kepler_q1q17_dr25_tce_eqt_err_-1_5-21-2020.csv', index=False)
    #
    # a = tce_tbl_aux.loc[np.abs(tce_tbl_aux['tce_eqt_err'] - tce_tbl_aux['tce_eqt_err_new']) > 100]
    # a['tce_sma_solarradii'] = np.nan
    # a['tce_sma_new_solarradii'] = np.nan
    # a['thr_solarradii'] = np.nan
    # for i, el in a.iterrows():
    #     a.loc[i, 'tce_sma_solarradii'] = el['tce_sma'] * F_AU / SUN_RADIUS
    #     a.loc[i, 'tce_sma_new_solarradii'] = el['tce_sma_new_noupdate'] * F_AU / SUN_RADIUS
    #     a.loc[i, 'thr_solarradii'] = el['tce_sradius'] + el['tce_prad'] * EARTH_RADIUS / SUN_RADIUS
    # a[['kepid', 'tce_plnt_num', 'tce_period', 'tce_period_err', 'tce_sradius', 'tce_sradius_err', 'tce_steff',
    #    'tce_steff', 'tce_slogg', 'tce_slogg_err', 'tce_sma', 'tce_sma_err', 'tce_sma_new', 'tce_sma_err_new',
    #    'tce_eqt', 'tce_eqt_err', 'tce_eqt_new', 'tce_eqt_err_new',
    #    'tce_sma_new_noupdate', 'tce_sma_err_new_noupdate', 'tce_sma_solarradii', 'tce_sma_new_solarradii',
    #    'thr_solarradii'
    #    ]].to_csv('/home/msaragoc/Downloads/kepler_q1q17_dr25_tce_eqt_err_diff_5-21-2020.csv', index=False)
    #
    # a = tce_tbl_aux.loc[((tce_tbl_aux['tce_albedo'] == 0) & tce_tbl_aux['tce_albedo_new'] != 0) | (
    #             (tce_tbl_aux['tce_albedo'] != 0) & (tce_tbl_aux['tce_albedo_new'] == 0))]
    # a[['kepid', 'tce_plnt_num', 'tce_period', 'tce_period_err', 'tce_ror', 'tce_ror_err', 'tce_sradius', 'tce_sradius_err',
    #    'tce_depth', 'tce_depth_err', 'wst_depth', 'tce_prad', 'tce_prad_err', 'tce_prad_new',
    #    'tce_prad_err_new',
    #    'tce_sma', 'tce_sma_err', 'tce_sma_new', 'tce_sma_err_new',
    #    'tce_albedo', 'tce_albedo_err', 'tce_albedo_new', 'tce_albedo_err_new',
    #    'tce_sma_new_noupdate', 'tce_sma_err_new_noupdate',
    #
    #    ]].to_csv('/home/msaragoc/Downloads/kepler_q1q17_dr25_tce_albedo_zero_5-21-2020.csv', index=False)
    #
    # a = tce_tbl_aux.loc[((tce_tbl_aux['tce_albedo_stat'] == 0) & tce_tbl_aux['tce_albedo_stat_new'] != 0) | (
    #             (tce_tbl_aux['tce_albedo_stat'] != 0) & (tce_tbl_aux['tce_albedo_stat_new'] == 0))]
    # a[['kepid', 'tce_plnt_num', 'tce_period', 'tce_period_err',  'tce_ror', 'tce_ror_err', 'tce_sradius', 'tce_sradius_err',
    #    'tce_depth', 'tce_depth_err', 'wst_depth', 'tce_prad', 'tce_prad_err',
    #    'tce_sma', 'tce_sma_err', 'tce_sma_new', 'tce_sma_err_new',
    #    'tce_albedo', 'tce_albedo_err', 'tce_albedo_new', 'tce_albedo_err_new', 'tce_albedo_stat', 'tce_albedo_stat_new',
    #    'tce_sma_new_noupdate', 'tce_sma_err_new_noupdate'
    #    ]].to_csv('/home/msaragoc/Downloads/kepler_q1q17_dr25_tce_albedo_stat_zero_5-21-2020.csv', index=False)
