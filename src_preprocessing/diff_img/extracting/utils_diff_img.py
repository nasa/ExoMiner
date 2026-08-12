""" Utility functions for extracting difference imaging. """

# 3rd party
from __future__ import annotations
import xml.etree.cElementTree as et
import os
import matplotlib.pyplot as plt
import numpy as np
import logging
import re
from matplotlib.colors import LogNorm, Normalize
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import xml.etree.ElementTree as ET
from mpl_toolkits.axes_grid1 import make_axes_locatable
from functools import lru_cache
import re
from tqdm import tqdm
import gzip

# local
from src_preprocessing.diff_img.extracting.aperture_wcs_utils import sky_to_aperture

plt.switch_backend('agg')

N_QUARTERS_KEPLER = 17
N_IMGS_IN_DIFF = 4  # diff, oot, it, snr
MAX_MAG = 25
MIN_MAG = 1
MAG_RANGE = MIN_MAG - MAX_MAG
TESS_MAG_SAT = 7
KEPLER_MAG_SAT = 12
MIN_IMG_VALUE = 1e-12

COMMENT_RE = re.compile(r'^\s*#')        # treat lines starting with optional whitespace + '#'
BLANK_RE   = re.compile(r'^\s*$')        # optionally skip purely blank lines at the top


def get_radec_from_tess_dv_xml(dv_xml_fp: Path | str) -> Tuple[Optional[float], Optional[float]]:
    """
    Parse a TESS SPOC DV XML file and return (RA_deg, Dec_deg).
    - RA is returned in degrees (converted from hours if necessary).
    - Dec is returned in degrees.
    - If a value cannot be found or parsed, returns (None, None) or (val, None).

    Robust to DV parameter name variants:
      RA:  'raDegrees', 'raHours', 'rightAscension', 'ra'
      Dec: 'decDegrees', 'declination', 'dec'

    Parameters
    ----------
    dv_xml_fp : Path | str
        File path to the DV XML (e.g., '..._dvr.xml').

    Returns
    -------
    (ra_deg, dec_deg) : (Optional[float], Optional[float])
    """
    
    dv_xml_fp = Path(dv_xml_fp)
    if dv_xml_fp.suffix == '.gz':
        with gzip.open(dv_xml_fp, 'rt', encoding='utf-8') as f:
            tree = ET.parse(f)
    else:
        tree = ET.parse(dv_xml_fp)

    root = tree.getroot()

    # SPOC DV XML namespace (Kepler heritage used by TESS DV)
    NS = {'dv': 'http://www.nasa.gov/2018/TESS/DV'}

    def _get_value(el):
        """Return the element's 'value' attribute if present, else stripped text; None if missing."""
        if el is None:
            return None
        return el.attrib.get('value') or (el.text.strip() if el.text else None)

    def _to_float(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    def _hours_to_deg(h):
        f = _to_float(h)
        return None if f is None else 15.0 * f

    ra_deg = None
    dec_deg = None
    
    # get elements with target coordinates
    ra_deg_el   = root.find('dv:raDegrees', NS)
    ra_hours_el = root.find('dv:raHours',   NS)
    dec_deg_el  = root.find('dv:decDegrees', NS)

    # get ra
    ra_deg_raw   = _get_value(ra_deg_el)
    ra_hours_raw = _get_value(ra_hours_el)

    ra_deg = _to_float(ra_deg_raw) 
    # convert from hours to degrees if necessary 
    if ra_deg is None and ra_hours_raw is not None:
        ra_deg = _hours_to_deg(ra_hours_raw)

    # get dec
    dec_deg_raw = _get_value(dec_deg_el)
    dec_deg     = _to_float(dec_deg_raw)

    return ra_deg, dec_deg


def plot_diff_img_data(
    diff_imgs,
    plot_fp,
    target_coords=None,
    neighbors_coords=None,
    target_mag=None,
    neighbors_mag=None,
    mag_sat=1,
    min_mag=None,
    logscale=True,
    # --- colorbar geometry (outside axes) ---
    flux_cbar_size="5%", flux_cbar_pad=0.08,   # right vertical flux colorbar
    mag_cbar_size="8%",  mag_cbar_pad=0.12     # top horizontal magnitude colorbar
):
    """
    Plot difference image data for a TCE:
      - Vertical flux colorbar on the RIGHT (outside) with full panel title.
      - Horizontal magnitude colorbar on the TOP (outside) with ticks/labels on top only.

    """

    # Magnitude normalization shared across all panels
    vmin = mag_sat
    vmax = MAX_MAG if min_mag is None else float(min_mag)
    mag_norm = Normalize(vmin=vmin, vmax=vmax)
    mag_cmap = 'plasma_r'

    def _create_subplot(ax_img, img, panel_title,
                        target_coords=None, neighbors_coords=None,
                        target_mag=None, neighbors_mag=None,
                        logscale=True, mask_invalid_pixels=False):
        """Render one image panel and place colorbars outside the axes."""

        # Handle invalid pixels
        if mask_invalid_pixels:
            img = np.ma.masked_less(img, 0)
            cmap_img = plt.cm.viridis
            cmap_img.set_bad(color='gray')
        else:
            img = img.copy()
            cmap_img = plt.cm.viridis

        # Log scale handling—avoid zeros
        if logscale:
            img[img == 0] = MIN_IMG_VALUE

        # Plot the image
        im = ax_img.imshow(
            img,
            cmap=cmap_img,
            norm=LogNorm() if logscale else None,
            origin='lower'
        )

        # Scatter target & neighbors using magnitude coloring (if provided)
        if (target_coords is not None) and (target_mag is not None):
            ax_img.scatter(
                target_coords[0], target_coords[1],
                marker='x', c=[target_mag], zorder=2,
                cmap=mag_cmap, norm=mag_norm
            )
        elif (target_coords is not None) and (target_mag is None):
            ax_img.scatter(target_coords[0], target_coords[1],
                           marker='x', zorder=2, color='white')

        if neighbors_coords:
            for i, (cx, cy) in enumerate(neighbors_coords):
                mag_i = None if neighbors_mag is None else neighbors_mag[i]
                if mag_i is not None:
                    ax_img.scatter(
                        cx, cy,
                        marker='*', c=[mag_i], zorder=1,
                        cmap=mag_cmap, norm=mag_norm
                    )
                else:
                    ax_img.scatter(cx, cy, marker='*', zorder=1, color='orange')

        # Axis labels (keep modest padding; title will live on the flux cbar)
        ax_img.set_ylabel('Row')
        ax_img.set_xlabel('Col', labelpad=8)

        # ---- Place colorbars outside using axes_grid1 divider ----
        divider = make_axes_locatable(ax_img)

        # 1) Vertical flux colorbar on the RIGHT (outside) with full panel title
        cax_flux = divider.append_axes("right", size=flux_cbar_size, pad=flux_cbar_pad)
        cbar_im = plt.colorbar(im, cax=cax_flux, orientation='vertical')
        # Make the flux cbar carry the "title + units"
        if panel_title == 'SNR Flux':
            cbar_im.set_label(rf'{panel_title}', labelpad=6)
        else:
            cbar_im.set_label(rf'{panel_title} [$e^-/cadence$]', labelpad=6)
        # Place ticks/labels on the right side, away from the image
        cbar_im.ax.yaxis.set_tick_params(labelright=True, labelleft=False, pad=3)
        cbar_im.ax.yaxis.set_label_position('right')

        # 2) Horizontal magnitude colorbar on the TOP (outside), ticks/labels on top only
        sm = plt.cm.ScalarMappable(cmap=mag_cmap, norm=mag_norm)
        sm.set_array([])
        cax_mag = divider.append_axes("top", size=mag_cbar_size, pad=mag_cbar_pad)
        cbar_sc = plt.colorbar(sm, cax=cax_mag, orientation='horizontal')
        cbar_sc.set_label('Magnitude', labelpad=4)
        # Show ticks/labels on top; hide bottom so nothing intrudes into the subplot
        cbar_sc.ax.xaxis.set_label_position('top')
        cbar_sc.ax.tick_params(labeltop=True, labelbottom=False, pad=3)

        # Optional: thin spine around the magnitude cbar to visually separate from axes
        for spine in cbar_sc.ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    # ---- Figure & subplots (NO constrained_layout to avoid conflicts)
    f, ax = plt.subplots(2, 2, figsize=(14, 14))

    # Panels:
    _create_subplot(
        ax[0, 0], diff_imgs[:, :, 2, 0], 'Difference Flux',
        target_coords=target_coords,
        neighbors_coords=neighbors_coords,
        target_mag=target_mag, neighbors_mag=neighbors_mag,
        logscale=False, mask_invalid_pixels=False
    )

    _create_subplot(
        ax[0, 1], diff_imgs[:, :, 1, 0], 'Out-of-transit Flux',
        target_coords=target_coords,
        neighbors_coords=neighbors_coords,
        target_mag=target_mag, neighbors_mag=neighbors_mag,
        logscale=logscale, mask_invalid_pixels=True
    )

    _create_subplot(
        ax[1, 0], diff_imgs[:, :, 0, 0], 'In-transit Flux',
        target_coords=target_coords,
        neighbors_coords=neighbors_coords,
        target_mag=target_mag, neighbors_mag=neighbors_mag,
        logscale=logscale, mask_invalid_pixels=True
    )

    _create_subplot(
        ax[1, 1], diff_imgs[:, :, 3, 0], 'SNR Flux',
        target_coords=target_coords,
        neighbors_coords=neighbors_coords,
        target_mag=target_mag, neighbors_mag=neighbors_mag,
        logscale=logscale, mask_invalid_pixels=True
    )

    # Give the figure breathing room for appended top/right colorbars
    # Tune these if any labels look clipped:
    f.subplots_adjust(left=0.06, right=0.96, bottom=0.07, top=0.93, wspace=0.35, hspace=0.40)

    f.savefig(plot_fp, dpi=150)
    plt.close(f)


def get_data_from_kepler_dv_xml(dv_xml_fp, tces, plot_dir, plot_prob, logger):
    """ Extract difference image data from the DV XML file for a set of Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}'
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param logger: logger
    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{kic_id}-{tce_plnt_num}'. The value is a dictionary that contains the following items: 'target_ref_centroid'
    is a dictionary that contains the value and uncertainty for the reference coordinates of the target star in the
    pixel domain; 'image_data' is a NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
    out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed by the
    last dimension of the array, in this order); 'image_number' is a list that contains the integer quarter numbers of
    the corresponding sequence of difference image data extracted for the TCE.
    """

    proc_id = os.getpid()

    # get an iterable
    context = et.iterparse(dv_xml_fp, events=("start", "end"))

    # get the root element
    event, root = next(context)

    n_tces = len(tces)
    tce_i = 0  # counter for TCEs in the DV XML
    data = {}
    for event, elem in context:

        n_tces_added = len(data)

        if event == "end" and elem.tag == "planetResults":  # iterate through each planet results container

            tce_i += 1

            uid = f'{elem.attrib["keplerId"]}-{elem.attrib["planetNumber"]}'

            if tce_i % 500 == 0:
                print(f'[{proc_id}] Iterating over TCE {tce_i} in {dv_xml_fp.name}', flush=True)

            if n_tces_added == n_tces:  # stop reading XML file once all TCEs were iterated through
                break

            if uid not in tces.index:
                continue

            logger.info(f'[{proc_id}] Getting difference image data for TCE KIC {uid}... ({n_tces_added}/{n_tces} '
                        f'TCEs)')

            # TODO: test this
            kmag = float([el for el in root if 'keplerMag' in el.tag][0].attrib['value'])

            data[uid] = {
                'target_ref_centroid': [],
                'image_data': [],
                'image_number': [],
                'mag': kmag,
                'neighbor_data': None,
                'quality_metric': [],
            }

            # get difference image results
            diff_img_res = [el for el in elem if el.tag == 'differenceImageResults']

            n_quarters = len(diff_img_res)

            if n_quarters < N_QUARTERS_KEPLER:
                logger.info(f'[{proc_id}] TCE KIC {uid} has less than {N_QUARTERS_KEPLER} quarters ({n_quarters})')

            # iterate over quarters
            for quarter_i in range(n_quarters):

                img_res_q = diff_img_res[quarter_i]

                # get quality metric data
                q_metric_q = [el.attrib for el in img_res_q if 'qualityMetric' in el.tag][0]
                data[uid]['quality_metric'].append(q_metric_q)

                # get quarter information
                data[uid]['image_number'].append(int(img_res_q.attrib['quarter']))

                img_px_data = img_res_q.findall('differenceImagePixelData')

                # n_pxs = len(img_px_data)

                px_dict = {(int(el.attrib['ccdRow']), int(el.attrib['ccdColumn'])): list(el) for el in img_px_data}

                # get max and min row and col
                px_row_lst, px_col_lst = zip(*px_dict.keys())
                # px_row_lst, px_col_lst = [], []
                # for px_row, px_col in px_dict.keys():
                #     px_row_lst.append(px_row)
                #     px_col_lst.append(px_col)

                min_row, max_row = min(px_row_lst), max(px_row_lst)
                min_col, max_col = min(px_col_lst), max(px_col_lst)

                # determine size of images
                row_size = max_row - min_row + 1
                col_size = max_col - min_col + 1

                # populate array with pixel values
                diff_imgs = np.nan * np.ones((row_size, col_size, N_IMGS_IN_DIFF, 2), dtype='float')

                for px_coord, diff_imgs_q in px_dict.items():
                    diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 0] = [float(el.attrib['value'])
                                                                                     for el in diff_imgs_q]
                    diff_imgs[px_coord[0] - min_row, px_coord[1] - min_col, :, 1] = [float(el.attrib['uncertainty'])
                                                                                     for el in diff_imgs_q]

                # get target position in pixel frame
                kic_centroid_ref = img_res_q.findall('kicReferenceCentroid')[0]
                # check for missing value
                if float(kic_centroid_ref.find('column').attrib['uncertainty']) == -1 or \
                        float(kic_centroid_ref.find('row').attrib['uncertainty']) == -1:
                    kic_centroid_ref_dict = {
                        'col': {k: float(v) for k, v in kic_centroid_ref.find('column').attrib.items()},
                        'row': {k: float(v) for k, v in kic_centroid_ref.find('row').attrib.items()}
                    }
                    logger.info(f'[{proc_id}] TCE KIC {uid} has missing reference centroid for target in quarter '
                                f'{img_res_q.attrib["quarter"]}.')
                    # continue
                else:
                    kic_centroid_ref_dict = {
                        'col': {k: float(v) - min_col if k == 'value' else float(v)
                                for k, v in kic_centroid_ref.find('column').attrib.items()},
                        'row': {k: float(v) - min_row if k == 'value' else float(v)
                                for k, v in kic_centroid_ref.find('row').attrib.items()}
                    }

                # plot difference image
                if np.random.uniform() <= plot_prob:
                    plot_diff_img_data(diff_imgs,
                                       target_coords=(kic_centroid_ref_dict['col']['value'],
                                                      kic_centroid_ref_dict['row']['value']),
                                       plot_fp=plot_dir / f'kic_{uid}.png',
                                       neighbors_coords=None,
                                       logscale=True,
                                       target_mag=data[uid]['mag'],
                                       neighbors_mag=None,
                                       mag_sat=KEPLER_MAG_SAT,
                                       )

                data[uid]['target_ref_centroid'].append(kic_centroid_ref_dict)
                data[uid]['image_data'].append(diff_imgs)

        root.clear()

    # np.save(save_dir / f'keplerq1q17_dr25_diffimg_pid{proc_id}.npy', data)

    return data


def get_data_from_kepler_dv_xml_main(dv_xml_fp, tces, save_dir, plot_dir, plot_prob, log_dir, job_i):
    """ Wrapper for `get_data_from_kepler_dv_xml()`. Extract difference image data from the DV XML file for a set of
    Kepler Q1-Q17 DR25 TCEs.

    :param dv_xml_fp: Path, file path to DV XML file
    :param tces: pandas DataFrame, TCEs for which to extract difference image data. Must contain two columns: 'uid' and
    'label'. 'uid' must be of the pattern '{tic_id}-{tce_plnt_num}'
    :param save_dir: Path, save directory
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param log_dir: Path, log directory
    :param job_i: int, job id
    :return:
    """

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_kepler_dv_xml-{job_i}')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_kepler_dv_xml-{job_i}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run {dv_xml_fp.name} ({len(tces)} TCEs)...')

    data = get_data_from_kepler_dv_xml(dv_xml_fp, tces, plot_dir, plot_prob, logger)

    np.save(save_dir / f'keplerq1q17_dr25_diffimg_{job_i}.npy', data)


def _freeze_dtypes(d: dict) -> tuple:
    """Make dtypes hashable for lru_cache keys.

    :param dict d: maps keys to data types
    :return tuple: tuple with keys and their data types
    """
    
    return tuple(sorted(d.items()))


def count_leading_comments(csv_path, skip_blank_top=True):
    """
    Count contiguous comment (and optionally blank) lines at the very top of a file.
    This is O(header_lines) and stops at the first non-comment/non-blank line.
    """
    n = 0
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        for line in f:
            if COMMENT_RE.match(line) or (skip_blank_top and BLANK_RE.match(line)):
                n += 1
            else:
                break
    return n

    
@lru_cache(maxsize=4)
def _load_sector_df(csv_path_str: str, usecols_tuple: tuple, dtypes_items: tuple,
                    engine: str = 'pyarrow', memory_map: bool = True) -> pd.DataFrame:
    """Load neighbors sector table and cache it.

    :param str csv_path_str: neighbors table filepath
    :param tuple usecols_tuple: required columns in neighbors table
    :param tuple dtypes_items: dtypes items pairs (column name, column data type)
    :param str engine: read file engine, defaults to 'pyarrow'
    :param bool memory_map: memory map, defaults to True
    :return pd.DataFrame: returns neighbors table indexed by target ID
    """
    
    usecols = list(usecols_tuple)
    dtypes  = dict(dtypes_items)
    
    if memory_map and engine == 'pyarrow':  # memory_map disabled when engine is pyarrow
        engine = 'c'
        df = pd.read_csv(csv_path_str, usecols=usecols, dtype=dtypes, engine=engine, memory_map=memory_map, comment='#')
    else:
        if engine == 'pyarrow':
            n_skip = count_leading_comments(csv_path_str, skip_blank_top=True)
            df = pd.read_csv(csv_path_str, sep=',', names=usecols, dtype=dtypes, engine=engine, skiprows=n_skip + 1, header=None, encoding='utf-8-sig')
        else:
            df = pd.read_csv(csv_path_str, usecols=usecols, dtype=dtypes, engine=engine, comment='#')
    
    return df.set_index('target_id', drop=False).copy(deep=False)


def get_neighbors_for_target_in_sector(sectors_obs, neighbors_dir, tic_id, sector_run_id, proc_id, logger, skip_no_neighbors=False, cache=False, data_collection_mode='2min'):
    """ Get data on neighbors for target `tic_id` in sectors in `sectors_obs`.
    
    Required columns in neighbors table:
        - ID: neighbor TIC ID (int)
        - Tmag: neighbor TESS Mag (float)
        - dstArcSec: distance from target in arcsec (float)
        - ra: right ascension for neighbor in deg (float)
        - dec: decliation for neighbor in deg (float)
        - target_id: target TIC ID (int)
        - col_px: column pixel coordinate for neighbor in the target's aperture frame (float)
        - row_px: row pixel coordinate for neighbor in the target's aperture frame (float)

    :param sectors_obs: list,
    :param neighbors_dir: Path, path to directory containing target neighbors data
    :param tic_id: int, target ID
    :param sector_run_id: str, sector run ID
    :param proc_id: int, process ID
    :param logger: logger
    :param skip_no_neighbors: bool; if False, it will raise an error when no neighbors table is found for the observed sector or 
        no neigbors were found for the target. If True, it wil return the  neighbors dictionary with empty pandas DataFrames
    :param cache: if True, it will read the neighbors table once (per-process) and cache it (up to maximum of `maxsize` tables - see function `_load_sector_df` decorator)
    :param data_collection_mode: str, either '2min' or 'ffi'. Required when extracting neighbors data
    
    :raise FileNotFoundError: no neighbors table was found for the sector if skip_no_neighbors is False
    :raise KeyError: no neighbors found for target TIC tic_id in the sector table if skip_no_neighbors is False

    :return: neighbors_dict, dict of pandas Dataframes with neighbors of target `tic_id` for each sector in `sectors_obs`.
    """

    usecols = ['ID', 'Tmag', 'ra', 'dec', 'target_id', 'col_px', 'row_px']
    dtypes  = {'ID': 'int64', 'target_id': 'int64', 'Tmag': 'float32', 'ra': 'float64', 'dec': 'float64', 'col_px': 'float32', 'row_px': 'float32'}

    neighbors_dict = {}  # dict of neighbors table for target across observed sectors
    for sector_obs in sectors_obs:
        
        if data_collection_mode == '2min':
            tbl_fp = neighbors_dir / f'S{sector_obs}' / 'mapping_results' / f'neighbors_pxcoords_S{sector_obs}.csv'
        else:
            tbl_fp = neighbors_dir / f'S{sector_obs}' / 'mapping_results' / 'ffi' / f'neighbors_pxcoords_S{sector_obs}.csv'
        
        # table not found
        if not tbl_fp.exists():
            err_str = f'[{proc_id}] [Sector run {sector_run_id}] Neighbors table for sector {sector_obs} not found.'
            logger.warning(err_str)
            if skip_no_neighbors:
                neighbors_target = pd.DataFrame({c: pd.Series([], dtype=dtypes[c]) for c in usecols})
            else:  # create empty table
                raise FileNotFoundError(err_str)
        
        # table found; read it
        else: 
            if cache:
                # read table once and cache it
                neighbors_tbl = _load_sector_df(
                    str(tbl_fp),
                    usecols_tuple=tuple(usecols),
                    dtypes_items=_freeze_dtypes(dtypes),
                    engine='pyarrow',
                    memory_map=True,
                    )
            else:
                # load neighbors table for this sector
                n_skip = count_leading_comments(tbl_fp, skip_blank_top=True)
                neighbors_tbl = pd.read_csv(
                                    tbl_fp,
                                    engine='pyarrow',
                                    sep=',',
                                    header=None,           
                                    names=usecols,           
                                    skiprows=n_skip + 1,   
                                    encoding='utf-8-sig',  
                                )

            neighbors_tbl = neighbors_tbl.set_index('target_id', drop=False)

            if tic_id in neighbors_tbl.index:
                # filter neighbors for this target 
                neighbors_target = neighbors_tbl.loc[[tic_id]]
            else:
                err_str = f'[{proc_id}] [Sector run {sector_run_id}] Target {tic_id} not found in the neighbors table for sector {sector_obs}'
                logger.warning(err_str)
                
                if skip_no_neighbors:
                    neighbors_target = pd.DataFrame({c: pd.Series([], dtype=dtypes[c]) for c in usecols})
                else:
                    raise KeyError(err_str)
                    
            # index target neighbors table on neighbor TIC ID
            neighbors_target = neighbors_target.set_index('ID')
            
            # add target's table to observed sector
            neighbors_dict[sector_obs] = neighbors_target

        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Found {len(neighbors_target)} neighbors for '
                    f'target {tic_id} in sector {sector_obs}.')

    return neighbors_dict


def map_neighbor_loc_aperture_to_local_dv_frame(neighbor_ap, target_ap, target_dv_local):
    """ Convert neighbor pixel coordinates from APERTURE frame to DV local frame.
    
    neighbor_ap: dictionary with keys ['col_px','row_px'] in APERTURE frame
    target_ap: dictionary with keys ('col', 'row') target in APERTURE frame
    target_dv_local: dictionary with keys ('col', 'row') target in DV local frame
    
    Returns a copy with neihgbor pixel coordinates in DV local frame (columns 'col_px','row_px').
    """
    
    col_t_ap, row_t_ap = target_ap['col'], target_ap['row']
    col_t_dv, row_t_dv = target_dv_local['col'], target_dv_local['row']

    neighbor_dv_local = neighbor_ap.copy()
    
    neighbor_dv_local['col_px'] = (neighbor_ap['col_px'] - col_t_ap) + col_t_dv
    neighbor_dv_local['row_px'] = (neighbor_ap['row_px'] - row_t_ap) + row_t_dv
    
    return neighbor_dv_local


def extract_data_sector(
    img_res_s,
    uid: str,
    sector_run_id: str,
    logger: logging.Logger,
    plot_dir: Path,
    plot_prob: float,
    *,
    tce_neighbors_dict: Optional[dict] = None,
    ticid_loc_ap_frame: Optional[tuple] = None,
    delta_mag: Optional[float] = None,
    mag_sat: float = TESS_MAG_SAT,
    n_imgs_in_diff: int = N_IMGS_IN_DIFF,
) -> Optional[Dict[str, Any]]:
    """
    Pure helper: extract and assemble DV difference-image data for a single sector 'img_res_s' of TCE 'uid'.
    Returns a dict with all fields the caller needs to update 'data[uid]' (no mutation here).
    If required tags/attributes are missing, returns None and the caller should skip this sector.
    
    Parameters
    ----------
    img_res_s: xml.etree.Element, differenceImageResults element for this sector
    uid: str, TCE unique ID (ticID-tcePlntNum-sectorRunID)
    sector_run_id: str, sector run ID
    logger: logging.Logger
    plot_dir: Path, directory to save plots
    plot_prob: float, probability to plot difference image for this sector
    tce_neighbors_dict: Optional[dict], dictionary target neighbors observed for this TCE in this sector, 
        keyed by neighbor TIC ID -> row 'row_px' and col 'col_px' aperture frame pixel coordinates 
        (or None if no neighbors data available)
    ticid_loc_ap_frame: Optional[tuple] (row, col), that contains location of target in the aperture frame
    delta_mag: Optional[float], minimum magnitude at which the colormap is clipped in plotting
    mag_sat: float, magnitude saturation threshold for plotting
    n_imgs_in_diff: int, number of images in difference image data (default 4: it, oot, diff, snr)

    Returns dict with keys:
        - 'quality_metric': dict            # parsed qualityMetric (typed)
        - 'sector': int                     # sector number
        - 'image_data': np.ndarray          # DV stamp (row, col, [it,oot,diff,snr], [value,uncert])
        - 'target_ref_centroid': dict       # centroid in DV local stamp coords (or unshifted if uncertainty == -1)
        - 'neighbor_data': dict or None
            # neighbors for this sector, with col_px/row_px shifted by -min_col/-min_row (origin alignment)
            # dict keyed by neighbor TIC ID -> row from neighbors table with updated 'col_px'/'row_px'
        - 'min_row': int
        - 'min_col': int
    """
    
    pid = os.getpid()

    # get quality metric
    try:
        q_metric_s = [el.attrib for el in img_res_s if 'qualityMetric' in el.tag][0]
        q_metric_s['value'] = float(q_metric_s['value'])
        q_metric_s['attempted'] = (q_metric_s['attempted'] == 'true')
        q_metric_s['valid'] = (q_metric_s['valid'] == 'true')
    except IndexError:
        logger.error(f"[{pid}] [Sector run {sector_run_id}] Missing qualityMetric for TCE TIC {uid}")
        return None

    # get sector number
    try:
        sector = int(img_res_s.attrib['sector'])
    except KeyError:
        logger.error(f"[{pid}] [Sector run {sector_run_id}] Missing sector attribute for TCE TIC {uid}")
        return None

    # get difference image pixel data
    img_px_data = [el for el in img_res_s if 'differenceImagePixelData' in el.tag]
    px_dict = {(int(el.attrib['ccdRow']), int(el.attrib['ccdColumn'])): list(el) for el in img_px_data}

    # gather bounds
    px_row_lst, px_col_lst = [], []
    for px_row, px_col in px_dict.keys():
        px_row_lst.append(px_row)
        px_col_lst.append(px_col)

    min_row, max_row = min(px_row_lst), max(px_row_lst)
    min_col, max_col = min(px_col_lst), max(px_col_lst)

    # array shape and allocation (row, col, [it, oot, diff, snr], [value, uncertainty])
    row_size = max_row - min_row + 1
    col_size = max_col - min_col + 1
    diff_imgs = np.nan * np.ones((row_size, col_size, n_imgs_in_diff, 2), dtype='float')

    # populate: shift absolute CCD coords into local DV stamp coords by subtracting min_row/min_col
    for (ccd_row, ccd_col), diff_imgs_q in px_dict.items():
        rr = ccd_row - min_row
        cc = ccd_col - min_col
        diff_imgs[rr, cc, :, 0] = [float(el.attrib['value']) for el in diff_imgs_q]
        diff_imgs[rr, cc, :, 1] = [float(el.attrib['uncertainty']) for el in diff_imgs_q]

    # get target reference centroid and shift to local stamp when valid
    tic_centroid_ref = [el for el in img_res_s if 'ticReferenceCentroid' in el.tag][0]
    tic_centroid_ref_col = [el for el in tic_centroid_ref if 'col' in el.tag][0].attrib
    tic_centroid_ref_row = [el for el in tic_centroid_ref if 'row' in el.tag][0].attrib

    if float(tic_centroid_ref_col['uncertainty']) == -1 or float(tic_centroid_ref_row['uncertainty']) == -1:
        tic_centroid_ref_dict = {
            'col': {k: float(v) for k, v in tic_centroid_ref_col.items()},
            'row': {k: float(v) for k, v in tic_centroid_ref_row.items()},
        }
        logger.info(
            f'[{pid}] [Sector run {sector_run_id}] '
            f'TCE TIC {uid} has missing reference centroid for target in sector {sector}.'
        )
        target_coords = None
    else:
        tic_centroid_ref_dict = {
            'col': {k: (float(v) - min_col) if k == 'value' else float(v) for k, v in tic_centroid_ref_col.items()},
            'row': {k: (float(v) - min_row) if k == 'value' else float(v) for k, v in tic_centroid_ref_row.items()},
        }
        target_coords = (tic_centroid_ref_dict['col']['value'], tic_centroid_ref_dict['row']['value'])

    # shift neighbors from aperture frame to local frame
    tce_neighbors_dict_local_dv_frame = {}
    if tce_neighbors_dict is not None and ticid_loc_ap_frame is not None and target_coords is not None: 
        # skip neighbors if target location in aperture frame could not be mapped from celestial coordinates 
        # to aperture frame using WCS in target light curve
        if np.isnan(ticid_loc_ap_frame['row']):  
            logger.warning(f'Could not map target of TCE {uid} celestial coordinates to the aperture frame.' 
                           ' Skipping all {len(tce_neighbors_dict)} neighbors found for this target.')
        else:
            for neighbor_id, neighbor_data in tce_neighbors_dict.items():
                tce_neighbors_dict_local_dv_frame[neighbor_id] = map_neighbor_loc_aperture_to_local_dv_frame(
                    neighbor_ap=neighbor_data,              
                    target_ap=ticid_loc_ap_frame,
                    target_dv_local={'col': target_coords[0], 'row': target_coords[1]},
                )

    # plotting
    if np.random.uniform() <= plot_prob:
        neighbors_coords = (
            [(v['col_px'], v['row_px']) for v in tce_neighbors_dict_local_dv_frame.values()]
            if len(tce_neighbors_dict_local_dv_frame) > 0 else None
        )
        neighbors_mags = (
            [v['Tmag'] for v in tce_neighbors_dict_local_dv_frame.values()]
            if len(tce_neighbors_dict_local_dv_frame) > 0 else None
        )

        plot_diff_img_data(
            diff_imgs,
            plot_dir / f'tic_{uid}_sector_{sector}.png',
            target_coords=target_coords,
            neighbors_coords=neighbors_coords,
            logscale=True,
            target_mag=None,           
            neighbors_mag=neighbors_mags,
            mag_sat=mag_sat,
            min_mag=delta_mag,
        )

    return {
        'quality_metric': q_metric_s,          
        'sector': sector,                      
        'image_data': diff_imgs,               
        'target_ref_centroid': tic_centroid_ref_dict,  
        'neighbor_data': tce_neighbors_dict_local_dv_frame, 
        'min_row': min_row,
        'min_col': min_col,
    }

def get_neighbors_target_explain_tce(transit_depth: float, target_tmag: float, neighbors_dict: dict, beta_thr: float = 0.3) -> dict:
    """ Filter neighbors based on transit depth - could these objects cause the observed transit depth of the TCE?
    
        flux_ratio = 10^[-0.4 (TMag_neighbor - TMag_target)
        
        accept neighbors whose flux_ratio >= beta_thr * tce_depth_frac
    
    Parameters
    ----------
    transit_depth: float, transit depth in ppm
    target_tmag: float, target TESS magnitude
    neighbors_dict: dict, each item is a DataFrame of neighbors for the target in a given sector
    beta_thr: float, transit depth threshold used when testing whether a full (beta=1) or partial (beta<1) eclipse of the neighbor 
        could explain the observed transit depth.
    
    Returns
    -------
    tce_neighbors_dict: dict, each item is a dict of neighbors for the target in a given sector that could explain
        the observed transit depth of the TCE; added fields 'flux_ratio' and 'explanation_ratio' with the explanation 
        ratio flux_ratio / tce_depth_frac
    """
    
    # filter neighbors based on transit depth - could these objects cause the observed transit depth?
    eps = 1e-12
    tce_depth_frac = max(transit_depth, 0.0) / 1e6
    tce_depth_frac = max(tce_depth_frac, eps)
    
    tce_neighbors_dict = {}
    for sector_id, neighbors_sector in neighbors_dict.items():
        tce_neighbors_sector = {}
        for neighbor_id, neighbor_data in neighbors_sector.iterrows():
            if np.isnan(neighbor_data['row_px']) or np.isnan(neighbor_data['col_px']):
                continue
                
            flux_ratio = 10 ** (-0.4 * (neighbor_data['Tmag'] - target_tmag))
            if flux_ratio >= beta_thr * tce_depth_frac: 
                tce_neighbors_sector.update({neighbor_id: dict(neighbor_data)})
                # add flux / transit depth ratio
                tce_neighbors_sector[neighbor_id]['flux_ratio'] = flux_ratio
                tce_neighbors_sector[neighbor_id]['explanation_ratio'] = flux_ratio / tce_depth_frac

        tce_neighbors_dict[sector_id] = tce_neighbors_sector
                
    return tce_neighbors_dict
    
    
def get_data_from_tess_dv_xml(dv_xml_fp, neighbors_dir, lc_dir, sector_run_id, plot_dir, plot_prob, logger, proc_id=-1, 
                              cache_neighbors_data=False, data_collection_mode='2min'):
    """ Extract difference image data from the TESS target DV XML file for the set of TCEs detected in that star for
    that TESS SPOC sector run.

    :param dv_xml_fp: Path, filepath to DV XML file.
    :param neighbors_dir: Path, path to directory containing target neighbors data
    :param lc_dir: Path, path to directory containing target light curve data
    :param sector_run_id: str, sector run ID
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param logger: logger
    :param proc_id: int, process ID
    :param cache_neighbors_data: if True, it will read the neighbors table once (per-process) and cache it 
        (up to maximum of `maxsize` tables - see function `_load_sector_df` decorator). Defaults to False.
    :param data_collection_mode: str, either '2min' or 'ffi'. Required when extracting neighbors data

    :return: dict, each item is the difference image data for a given TCE. The TCE is identified by the string key
    '{tic_id}-{tce_plnt_num}-S{sector_run}'. The value is a dictionary that contains six items:
        - 'target_ref_centroid' is a list of dictionaries that contain the value and uncertainty for the reference
        coordinates of the target star in the pixel domain in each observed sector;
        - 'image_data' is a list of NumPy array (n_rows, n_cols, n_imgs, 2) that contains the in-transit,
        out-of-transit, difference, and SNR flux images in this order (pixel values and uncertainties are addressed by
        the last dimension of the array, in this order) for each observe sector;
        - 'image_number' is a list that contains the integer sector number of the corresponding sequence of difference
        image data extracted for the TCE.
        - 'mag' is the target's magnitude.
        - 'neighbor_data' is a list that, for each sector, contains a dictionary where each key is the TIC ID of
        neighboring objects that maps to a dictionary with the column 'col_px' and row 'row_px' coordinates of these
        objects in the CCD pixel frame of the target star along with the corresponding magnitude 'TMag' and distance to
        the target in arcseconds 'dst_arcsec'.
    """

    data = {}
    
    # get RA and Dec of target from DV XML
    target_ra_xml, target_dec_xml = get_radec_from_tess_dv_xml(dv_xml_fp)

    try:
        if dv_xml_fp.suffix == '.gz':
            with gzip.open(dv_xml_fp, 'rt', encoding='utf-8') as f:
                tree = et.parse(f)
        else:
            tree = et.parse(dv_xml_fp)
    except Exception as e:
        raise Exception(f'{proc_id}] [Sector run {sector_run_id}] Exception found when reading {dv_xml_fp}: {e}.')

    root = tree.getroot()

    tic_id = int(root.attrib['ticId'])

    tmag = float([el for el in root if 'tessMag' in el.tag][0].attrib['value'])

    planet_res_lst = [el for el in root if 'planetResults' in el.tag]

    n_sectors_expected = root.attrib['sectorsObserved'].count('1')
    sectors_obs = [i for i, char in enumerate(root.attrib['sectorsObserved']) if char == '1']
    
    # get target aperture WCS from light curve file and map target RA and Dec to aperture pixel frame for each observed sector
    if neighbors_dir:
        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Getting aperture WCS centroid for target {tic_id} and' 
                    f' mapping target {tic_id} RA and Dec to aperture frame in sectors {sectors_obs}.')
        ticid_loc_ap_frame = {sector_obs: sky_to_aperture(lc_dir, tic_id, sector_obs, target_ra_xml, target_dec_xml) 
                            for sector_obs in sectors_obs}

    # get neighboring stars
    if neighbors_dir:
        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Getting neighbors information for target {tic_id} in '
                    f'sectors {sectors_obs}...')

        neighbors_dict = get_neighbors_for_target_in_sector(
            sectors_obs, 
            neighbors_dir, 
            tic_id, 
            sector_run_id, 
            proc_id,
            logger,
            skip_no_neighbors=False,
            cache=cache_neighbors_data,
            data_collection_mode=data_collection_mode
            )

    n_tces = len(planet_res_lst)
    tce_i = 0
    logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Found {n_tces} TCEs for target {tic_id}. Iterating through the TCEs...')
    # iterate through each planet (i.e., TCE) results container
    for planet_res in planet_res_lst:

        tce_i += 1

        uid = f'{root.attrib["ticId"]}-{planet_res.attrib["planetNumber"]}-S{sector_run_id}'

        if neighbors_dir:
            # filter neighbors based on transit depth - could these objects cause the observed transit depth?
            tce_depth =float(planet_res.find(
                        './/dv:modelParameter[@name="transitDepthPpm"]',
                        {'dv': 'http://www.nasa.gov/2018/TESS/DV'}).attrib['value'])
            tce_neighbors_dict = get_neighbors_target_explain_tce(tce_depth, tmag, neighbors_dict)

        data[uid] = {
            'target_ref_centroid': [],
            'image_data': [],
            'mag': tmag,
            'image_number': [],
            'quality_metric': [],
        }
        # if neighbors_dir:
        #     data[uid]['neighbor_data'] = tce_neighbors_dict
        if neighbors_dir:
            data[uid]['neighbor_data'] = {}

        logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Getting difference image data for TCE TIC '
                    f'{uid} ({tce_i}/{n_tces} TCEs)...')

        # get difference image results
        diff_img_res = [el for el in planet_res if 'differenceImageResults' in el.tag]

        n_sectors = len(diff_img_res)

        if n_sectors < n_sectors_expected:
            logger.info(f'[{proc_id}] [Sector run {sector_run_id}] TCE TIC {uid} has less than '
                        f'{n_sectors_expected} '
                        f'sectors ({n_sectors})')

        # iterate over sectors
        logger.info(f"[{proc_id}] [Sector run {sector_run_id}] TCE TIC {uid} has {n_sectors} sector(s) "
                    f"(expected: {n_sectors_expected})")
        for sector_i in range(n_sectors):

            img_res_s = diff_img_res[sector_i]

            sector_data = extract_data_sector(
                img_res_s=img_res_s,
                uid=uid,
                sector_run_id=sector_run_id,
                logger=logger,
                plot_dir=plot_dir,
                plot_prob=plot_prob,
                tce_neighbors_dict=tce_neighbors_dict[sectors_obs[sector_i]] if neighbors_dir else None,  
                ticid_loc_ap_frame=ticid_loc_ap_frame[sectors_obs[sector_i]] if neighbors_dir else None,
                delta_mag=None,                   
                mag_sat=TESS_MAG_SAT,
                n_imgs_in_diff=N_IMGS_IN_DIFF,
            )

            if sector_data is None:
                continue

            data[uid]['quality_metric'].append(sector_data['quality_metric'])
            data[uid]['image_number'].append(sector_data['sector'])
            data[uid]['target_ref_centroid'].append(sector_data['target_ref_centroid'])  # needed during preprocessing 
            data[uid]['image_data'].append(sector_data['image_data'])
            # if sector_data['neighbor_data_shifted'] is not None:
            #     if 'neighbor_data' not in data[uid]:
            #         data[uid]['neighbor_data'] = [None] * n_sectors
            if len(sector_data['neighbor_data']) > 0:
                data[uid]['neighbor_data'][sector_data['sector']] = sector_data['neighbor_data']

    return data


def check_data_releases_dv_xmls(tic_id, dv_xml_fp, dv_xml_run_fps):
    """
    Check if the DV XML is associated with the latest SPOC run in the data directory. 
    It handles standard SPOC and HLSP formats, .xml and .xml.gz).
    
    Args:
       tic_id (str): The TIC ID padded with zero to length of 16
       dv_xml_fp (Path): The path to the DV XML file
       dv_xml_run_fps (list of Path): List of paths to the DV XML run files
    Returns:
       bool: True if the data is the latest release, False otherwise
    """
    latest = True
    
    def get_file_info(filepath):
        name = filepath.name
        
        # 1. Extract sector run (Looks specifically for 's' followed by 4 digits, twice)
        # Matches: "s0001-s0092" or "s0084-s0084"
        sector_match = re.search(r'(s\d{4}-s\d{4})', name)
        if not sector_match:
            return None, None
        sector_run = sector_match.group(1)
        
        # 2. Extract Data Release or Version number
        # Matches either "-01022_dvr" OR "_v1_dvr" and extracts the numbers (1022 or 1)
        dr_match = re.search(r'(?:-|_v)(\d+)_dvr', name)
        if not dr_match:
            return None, None
        dr_num = int(dr_match.group(1))
        
        return sector_run, dr_num

    # Get info for the current file
    curr_sector_run, curr_dr = get_file_info(dv_xml_fp)
    if curr_sector_run is None or curr_dr is None:
        return latest  # Fallback if filename is completely malformed
    
    matching_dr_numbers = []
    
    for fp in dv_xml_run_fps:
        # 1. Must end in _dvr.xml OR _dvr.xml.gz
        if not (fp.name.endswith('_dvr.xml') or fp.name.endswith('_dvr.xml.gz')):
            continue
            
        # 2. Must match the TIC ID
        if tic_id not in fp.name:
            continue
            
        # 3. Extract info and check sector run
        fp_sector_run, fp_dr = get_file_info(fp)
        
        if fp_sector_run == curr_sector_run and fp_dr is not None:
            matching_dr_numbers.append(fp_dr)
                
    # If we found multiple data releases for this specific TIC + Sector Run
    if len(matching_dr_numbers) > 1:
        latest_dr = max(matching_dr_numbers)
        
        if curr_dr != latest_dr:
            latest = False
            
    return latest


def get_data_from_tess_dv_xml_main(dv_xml_run, save_dir, neighbors_dir, lc_dir, plot_dir, plot_prob, log_dir, job_i, 
                                   check_existence_multiple_versions=False, targets_sectors_tbl=None, append_data=False, 
                                   cache_neighbors_data=False, data_collection_mode='2min'):
    """ Wrapper for `get_data_from_tess_dv_xml()`. Extract difference image data from the DV XML files for a TESS sector
    run.

    :param dv_xml_run: Path, path to sector run with DV XML files.
    :param save_dir: Path, save directory
    :param neighbors_dir: Path, path to directory containing target neighbors data
    :param lc_dir: Path, path to directory containing target light curve data
    :param plot_dir: Path, plot directory
    :param plot_prob: float, probability to plot difference image for a given example ([0, 1])
    :param log_dir: Path, log directory
    :param job_i: int, job id
    :param check_existence_multiple_versions: bool whether to check existence of multiple versions (different runs) of DV
    :param targets_sectors_tbl: pd.DataFrame or None, if provided, only extracts the DV XML files for the corresponding targets
        in the requested sector runs. The DataFrame must contain two columns: 'tic_id' and 'sector_run', where 'sector_run' follows 
        format <start_sector>-<end_sector> (e.g., '1-92' or '5-5' for single sector runs), and 'tic_id' is the TIC ID as an integer.
    :param append_data: bool, appends data to existing data dictionary in `save_dir` with filename tess_diffimg_<dv_xml_run.name>.npy. 
        If results already exist for target, then extraction from DV XML file is skipped. Defaults to False.
    :param cache_neighbors_data: if True, it will read the neighbors table once (per-process) and cache it 
        (up to maximum of `maxsize` tables - see function `_load_sector_df` decorator)
    :param data_collection_mode: str, either '2min' or 'ffi'. Required when extracting neighbors data

    :return:
    """

    # set up logger
    logger = logging.getLogger(name=f'extract_img_data_tess_dv_xml_{dv_xml_run.name}')
    logger_handler = logging.FileHandler(filename=log_dir / f'extract_img_data_from_tess_dv_xml-{dv_xml_run.name}.log',
                                         mode='w')
    logger_formatter = logging.Formatter('%(asctime)s - %(message)s')
    logger.setLevel(logging.INFO)
    logger_handler.setFormatter(logger_formatter)
    logger.addHandler(logger_handler)
    logger.info(f'[{job_i}] Starting run {dv_xml_run.name}...')

    proc_id = os.getpid()
    
    data_fp = save_dir / f'tess_diffimg_{dv_xml_run.name}.npy'

    new_data_flag = True
    if append_data:
        logger.info(f'Looking for data dictionary in {str(data_fp)}...')
        if data_fp.exists():
            logger.info(f'Found data dictionary in {str(data_fp)}. Loading data...')
            data = np.load(data_fp, allow_pickle=True).item()
            logger.info(f'Loaded data.')
            new_data_flag = False
        else:
            logger.info(f'Data dictionary was not found in {str(data_fp)}.')
            data = {}
    else:
        data = {}

    # get filepaths to xml files
    dv_xml_run_fps = list(dv_xml_run.rglob("*.xml")) + list(dv_xml_run.rglob("*.xml.gz"))
    n_targets = len(dv_xml_run_fps)
    logger.info(f'[{proc_id}] Found {n_targets} targets DV xml files in {n_targets}.')
    
    if targets_sectors_tbl is not None:
        targets_sectors_tbl['tic_id_str'] = targets_sectors_tbl['tic_id'].astype(str).str.zfill(16)

        # normalize sector_run to match the format in filenames (e.g., '1-92' → 's0001-s0092')
        def format_sector_run(sector_run):
            start, end = sector_run.split('-')
            return f"s{int(start):04d}-s{int(end):04d}"

        targets_sectors_tbl['sector_run_str'] = targets_sectors_tbl['sector_run'].apply(format_sector_run)
           
        filtered_dv_xml_run_fps = []
        for _, row in targets_sectors_tbl.iterrows():
            for dv_xml_run_fp in dv_xml_run_fps:
                if row['tic_id_str'] in dv_xml_run_fp.name and row['sector_run_str'] in dv_xml_run_fp.name:
                    filtered_dv_xml_run_fps.append(dv_xml_run_fp)
        
        dv_xml_run_fps = filtered_dv_xml_run_fps
        n_targets = len(dv_xml_run_fps)

        logger.info(f'[{proc_id}] Found {n_targets} targets DV xml files in {len(dv_xml_run_fps)} after excluding files using targets-sector runs table.')


    pbar = tqdm(total=len(dv_xml_run_fps), unit='target', desc=f'Sector run {dv_xml_run.name}')
    remaining = len(dv_xml_run_fps) % 100
    for target_i, dv_xml_fp in enumerate(dv_xml_run_fps):
        
        if target_i % 100 == 0:
                pbar.update(100)

        # get sector run ID from filename
        s_sector, e_sector = re.findall('-s[0-9]+', dv_xml_fp.stem)
        s_sector, e_sector = int(s_sector[2:]), int(e_sector[2:])
        if s_sector != e_sector:  # multisector run
            sector_run_id = f'{s_sector}-{e_sector}'
        else:
            sector_run_id = f'{s_sector}'

        if target_i % 1000 == 0:
            
            if target_i > 0:  # saving partial results
                if new_data_flag:
                    logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Saving partial results into disk in {str(data_fp)} for {target_i} targets ({len(data)} TCEs)...')
                    np.save(data_fp, data)
                    logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Saved.')
                    
            logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Iterating over TIC {target_i}/{n_targets} in '
                        f'{dv_xml_fp.name}.')
            
        tic_id_strpad16 = re.findall(r'\d{16}', dv_xml_fp.name)[0]  # get tic id from filename
        tic_id_int = str(int(tic_id_strpad16))
        
        # check if data from target was already added
        if append_data:
            target_tce_matches = [tce_uid for tce_uid in data if tic_id_int in tce_uid]
            if len(target_tce_matches) > 0:
                logger.info(f'[{proc_id}] [Sector run {sector_run_id}] Found data already extracted for target {tic_id_int} ({len(target_tce_matches)} TCEs). '
                            f'Skipping extraction for {dv_xml_fp.name}...')
                continue
  
        try:
            # check if there are results for more than one processing run for this TIC and sector run
            if check_existence_multiple_versions:
                latest_dr = check_data_releases_dv_xmls(tic_id_strpad16, dv_xml_fp, dv_xml_run_fps)
                if not latest_dr:
                    logger.info(f'[{proc_id}] [Sector run {sector_run_id}] '
                                f'Skipping {dv_xml_fp.name} for TIC {int(tic_id_int)} since there is '
                                f'more recent processed results: latest release {latest_dr})'
                                f'... ({target_i}/{n_targets} targets)')
                    continue

            data_dv_xml = get_data_from_tess_dv_xml(dv_xml_fp, neighbors_dir, lc_dir, sector_run_id, plot_dir, plot_prob,
                                                    logger, proc_id, cache_neighbors_data, data_collection_mode)
            data.update(data_dv_xml)
            if append_data and not new_data_flag:
                new_data_flag = True
            
        except Exception:
            logger.exception(f'[{job_i}] Exception occurred when getting data from {dv_xml_fp.name}')
            continue
    
    if remaining:
        pbar.update(remaining)
    pbar.close()

    np.save(data_fp, data)

    logger.info(f'[{job_i}] Finished run {dv_xml_run.name}.')
