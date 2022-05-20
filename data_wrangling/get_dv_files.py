"""
Download DV summaries based on URLs.
"""

# 3rd party
from urllib.request import urlretrieve
import os


def download_file(downloadUrl, filePath):
    urlretrieve(downloadUrl, filePath)


def get_dv_files(downloadUrls, downloadDir):

    os.makedirs(downloadDir, exist_ok=True)

    for urlFile in downloadUrls:
        download_file(urlFile, os.path.join(downloadDir, urlFile.split('/')[-1]))


if __name__ == "__main__":

    rootUrl = 'https://exoplanetarchive.ipac.caltech.edu/data/KeplerData/'
    downloadUrls = ['010/010152/010152836/dv/kplr010152836-002-20160209194854_dvs.pdf']
    downloadUrls = [rootUrl + url for url in downloadUrls]
    downloadDir = '/home/msaragoc/Projects/Kepler-TESS_exoplanet/Kepler_planet_finder/results_ensemble/' \
                  'dr25tcert_spline_gapped_glflux-lcentr-loe-6stellar_glfluxconfig/NTPsabovethr_reports'

    get_dv_files(downloadUrls, downloadDir)
