""" 
Copy light curve FITS files and DV XML to a specified directory with a specific hierarchy. 

The hierarchy is as follows:

/
├── sector_0001/
│   ├── tess2018206045859-s0001-0000000142052876-0120-s_lc.fits
│   ├── tess2018206045859-s0001-0000000142052876-0120-s_lc.fits
│   └── ...
├── sector_0002/
│   ├── tess2018234235059-s0002-0000000142082942-0121-s_lc.fits
│   └── ...
├── sector_0003/
│   └── ...
└── ...

Similar structure is used for DV XML files:

/
├── sector_run_0001-0001/
│   └── ...
├── sector_run_0001-0003/
│   ├── tess2024170090940-s0001-s0003-0000000020352516-00899_dvr.xml
│   └── ...
└── ...
"""

# 3rd party
from pathlib import Path
import shutil

#%% setup

lc_dest_root_dir = Path('/data3/exoplnt_dl/lc_fits/2-min')
dv_xml_dest_root_dir = Path('/data3/exoplnt_dl/dv_xml/2-min')
src_root_dir = Path('/data3/exoplnt_dl/experiments/exominer_pipeline/runs')

#%% get lc FITS filepaths

lc_fits_fps = list(src_root_dir.rglob('tess*lc.fits'))
print(f'Found {len(lc_fits_fps)} TESS light curve FITS files under {src_root_dir}.')

# exclude files with duplicate names
unique_files = {}
for fp in lc_fits_fps:
    filename = fp.name
    if filename not in unique_files:
        unique_files[filename] = fp

# keep only unique filepaths
lc_fits_fps = list(unique_files.values())
print(f'After excluding duplicates, {len(lc_fits_fps)} unique TESS light curve FITS files remain.')

#%% get DV XML filepaths

dv_xml_fps = list(src_root_dir.rglob('*dvr.xml'))
print(f'Found {len(dv_xml_fps)} TESS SPOC DV XML files under {src_root_dir}.')

# exclude files with duplicate names
unique_files = {}
for fp in dv_xml_fps:
    filename = fp.name
    if filename not in unique_files:
        unique_files[filename] = fp

# keep only unique file paths
dv_xml_fps = list(unique_files.values())
print(f'After excluding duplicates, {len(dv_xml_fps)} unique TESS SPOC DV XML files remain.')

#%% copy lc FITS files

for lc_fp_i, lc_fp in enumerate(lc_fits_fps):
    
    if lc_fp_i % 50 == 0:
        print(f'Copying {lc_fp_i + 1} out of {len(lc_fits_fps)} TESS light curve FITS files...')
    
    sector = int(lc_fp.name.split('-')[1][1:])
    dest_dir = lc_dest_root_dir / f'sector_{sector}'
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(lc_fp, dest_dir / lc_fp.name)
    
#%% copy DV XML files

for dv_xml_i, dv_xml_fp in enumerate(dv_xml_fps):
    
    if dv_xml_i % 50 == 0:
        print(f'Copying {dv_xml_i + 1} out of {len(dv_xml_fps)} TESS SPOC DV XML...')
    
    sector = '-'.join([sector_id[1:] for sector_id in dv_xml_fp.name.split('-')[1:3]])
    dest_dir = dv_xml_dest_root_dir / f'sector_run_{sector}'
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    shutil.copy2(dv_xml_fp, dest_dir / dv_xml_fp.name)