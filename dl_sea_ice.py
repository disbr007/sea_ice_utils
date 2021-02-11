# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 14:00:21 2019

@author: disbr007

Downloads sea-ice concentration rasters from NSDIC FTP.

Resamples NSDIC Sea-Ice Rasters to change non-sea-ice class values
(land, coast, missing, pole hole, ocean) to be NoData value for
subsequent analysis. This is no a legacy preprocessing step, as
determine_sea_ice.py does this on-the-fly.
This was so that rasters can be searched for any valid value.
"""

import argparse
from datetime import datetime
from ftplib import FTP
import logging
import numpy as np
import os
from pathlib import Path
import sys

from osgeo import gdal, osr
from tqdm import tqdm


logger = logging.getLogger(__name__)
logger.setLevel('INFO')
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

gdal.UseExceptions()

FTP_URL = r'sidads.colorado.edu'
FTP_BASE_DIR = r'/pub/DATASETS/NOAA/G02135/'
FTP_POLES = {'north': FTP_BASE_DIR + r'/north/daily/geotiff',
             'south': FTP_BASE_DIR + r'/south/daily/geotiff'}

EXTENT_NAME = 'extent'
CONCENTRATION_NAME = 'concentration'


def dl_sea_ice(last_update: str, dest_dir: str,
                   north: bool = True, south: bool = True,
                   dl_extent: bool = True, dl_concentration: bool = True,
                   overwrite: bool = False):
    """
    Downloads any new rasters since the last update date.
    last_update: str
        Date string like '2019-07-31'
    north: bool
        Update Arctic rasters
    south:
        Update Antarctic rasters
    dl_extent: bool
        Download extent rasters
    dl_concentration: bool
        Download concentration rasters
    overwrite: bool
        Overwrite files if they exist.
    """
    logger.info('Downloading rasters since: {}'.format(last_update))
    # Type conversions
    dest_dir = Path(dest_dir)

    # Get current year, month, day
    now = datetime.now()

    # Convert last update to datetime
    last_update_dt = datetime.strptime(last_update, r'%Y-%m-%d')

    ftp = FTP(FTP_URL)
    ftp.login()
    ftp.cwd(FTP_BASE_DIR)

    poles = [p[0] for p in [('north', north), ('south', south)] if p[1]]

    # Download sea-ice rasters, organized by year and month
    files_dl = 0
    for p in poles:
        ftp.cwd(FTP_POLES[p])
        data_dir = ftp.pwd()
        logger.info('Downloading sea-ice rasters ({})...'.format(p))
        years = [y for y in range(last_update_dt.year, now.year, 1)]
        pbar_years = tqdm(years)
        for year_dir in pbar_years:
            pbar_years.set_description('{}'.format(year_dir))
            month_dirs = ftp.nlst(r'{}/{}'.format(data_dir, year_dir))[2:]
            pbar_months = tqdm(month_dirs, leave=False)
            for month_dir in month_dirs:
                pbar_months.set_description('{}'.format(month_dir))
                month = int(month_dir[:2])
                file_ps = ftp.nlst(r'{}/{}/{}'.format(data_dir, year_dir, month_dir))[2:]

                ## Date is part of file name - day is element 8 and 9
                days_files = []
                if dl_extent:
                    days_files.extend([(x[8:10], x) for x in file_ps if EXTENT_NAME in x])
                if dl_concentration:
                    days_files.extend([(x[8:10], x) for x in file_ps if CONCENTRATION_NAME in x])

                pbar_days = tqdm(days_files, leave=False)
                for day, file_n in days_files:
                    pbar_days.set_description(day)
                    file_date = datetime.strptime('{}-{}-{}'.format(year_dir, str(month).zfill(2), day), '%Y-%m-%d')
                    if file_date > last_update_dt:
                        file_p = r'{}/{}/{}/{}'.format(data_dir, year_dir, month_dir, file_n)
                        out_p = dest_dir / Path(ftp.pwd()).relative_to(FTP_BASE_DIR) / str(year_dir) / month_dir / file_n
                        if not out_p.exists() or overwrite:
                            if not out_p.parent.exists():
                                os.makedirs(out_p.parent)
                            with open(out_p, 'wb') as fhandle:
                                ftp.retrbinary("RETR {}".format(file_p), fhandle.write)
                            files_dl +=1
                    pbar_days.update(1)
                pbar_days.close()
                pbar_months.update(1)

    logger.info('Total files downloaded: {:,}'.format(files_dl))


def resample_nodata(f_p, out_path, convert_vals, out_nodata):
    """
    Takes the NSDIC Sea-ice .tifs and resamples the four 
    classes to be no-data values.
    f_p: file path to .tif
    convert_vals: list
        no data values
    out_path: path to write resampled .tif to
    """
    # Read source and metadata
    ds = gdal.Open(f_p)
    gt = ds.GetGeoTransform()
    
    prj = osr.SpatialReference()
    prj.ImportFromWkt(ds.GetProjectionRef())
    
    x_sz = ds.RasterXSize
    y_sz = ds.RasterYSize
    if out_nodata is None:
        out_nodata = ds.GetRasterBand(1).GetNoDataValue()

    dtype = ds.GetRasterBand(1).DataType
#    dtype = gdal.GetDataTypeName(dtype)

    ## Read as array and convert values to out_nodata
    ar = ds.ReadAsArray()
    ar = np.where(np.in1d(ar, convert_vals).reshape(ar.shape), out_nodata, ar)

    # Write
    # Look up table for GDAL data types - dst is the signed version of src if applicable
    signed_dtype_lut = {
            0: {'src': 'Unknown', 'dst': 0},
            1: {'src': 'Byte', 'dst': 1},
            2: {'src': 'UInt16', 'dst': 3},
            3: {'src': 'Int16', 'dst': 3},
            4: {'src': 'UInt32', 'dst': 5},
            5: {'src': 'Int32', 'dst': 5},
            6: {'src': 'Float32', 'dst': 6},
            7: {'src': 'Float64', 'dst': 7},
            8: {'src': 'CInt16', 'dst': 8},
            9: {'src': 'CInt32', 'dst': 9},
            10:{'src': 'CFloat32', 'dst': 10},
            11:{'src': 'CFloat64', 'dst': 11},
            }
    
    # Create intermediate directories
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Create new raster
    fmt = 'GTiff'
    driver = gdal.GetDriverByName(fmt)
    dst_dtype = signed_dtype_lut[dtype]['dst']
    dst_ds = driver.Create(out_path, x_sz, y_sz, 1, dst_dtype)
    dst_ds.GetRasterBand(1).WriteArray(ar)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(prj.ExportToWkt())
    dst_ds.GetRasterBand(1).SetNoDataValue(out_nodata)
    
    dst_ds = None


def resample_loop(sea_ice_directory, out_dir, out_nodata, overwrite=False):
    """
    Calls resample_nodata in a loop for every *_concetration.tif and 
    *_extent.tif in the given directory, resampling class values to
    no data.
    sea_ice_path: path to directory holding rasters. sub-directories are OK.
    out_dir: path to write resampled rasters to.
    """

    ## Concentration raster no data values
    con_miss = 2550
    con_land = 2540
    con_coast = 2530
    con_pol = 2510
    con_convert_vals = [con_miss, con_land, con_coast, con_pol]
    
    ## Extent raster no data values
    ext_miss = 255
    ext_land = 254
    ext_coast = 253
    ext_pol = 210
    ext_convert_vals = [ext_miss, ext_land, ext_coast, ext_pol]

    concentration_sfx = '_concentration_v3.0.tif'
    extent_sfx = '_extent_v3.0.tif'

    # Loop through rasters
    # TODO: add north/south constraints here
    all_tifs = []
    for root, dirs, files in os.walk(sea_ice_directory):
        for file in files:
            if file.endswith((concentration_sfx, extent_sfx)):
                all_tifs.append(Path(root) / file)

    pbar = tqdm(all_tifs, desc='Resampling: ')
    for tif in pbar:
        spaces = 0
        if len(tif.name) < 33:
            spaces = 7
        pbar.set_description('Resampling: {}{}'.format(tif.name, ' '*spaces))
        out_path = Path(out_dir) / tif.relative_to(sea_ice_directory).parent / '{}_nd{}'.format(tif.stem, tif.suffix)
        if overwrite or not Path(out_path).exists():
            # Resample concentration rasters
            if tif.name.endswith(concentration_sfx):
                resample_nodata(str(tif), str(out_path), con_convert_vals, out_nodata=out_nodata)
            # Resample extent rasters
            if tif.name.endswith(extent_sfx):
                resample_nodata(str(tif), str(out_path), ext_convert_vals, out_nodata=out_nodata)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    resample_args = parser.add_argument_group('Resample Non Ice Arguments')

    parser.add_argument('sea_ice_directory', type=str,
                        help='Path to parent directory stroing sea-ice rasters. New rasters will be '
                             'downloaded here.')
    parser.add_argument('--last_update_date', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                        help="""The date through which rasters have been resampled. I.E. set to 1990-01-01
                            to download and resample everything from 1990-01-01 to present.""")
    parser.add_argument('--north', action='store_true',
                        help='Perform actions (update and/or resample on North Pole rasters.')
    parser.add_argument('--south', action='store_true',
                        help='Perform actions (update and/or resample on South Pole rasters.')
    parser.add_argument('--dl_extent', action='store_true',
                        help='Download extent rasters.')
    parser.add_argument('--dl_concentration', action='store_true',
                        help='Download concentration rasters.')

    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite rasters if they exist rather than skipping. This applies '
                             'to both downloading new rasters and resampling NoData '
                             '(if --nonice2nodata is used).')

    resample_args.add_argument('--nonice2nodata', action='store_true',
                                help='Resample non-ice (land, coast, missing, pole hole) values to be '
                                     'NoData. This was a legacy preprocessing step and is no done on the '
                                     'fly in determine_sea_ice.')
    resample_args.add_argument('--out_directory', type=str,
                               help='Use with --nonice2nodata, the directory to write resampled rasters to.')
    resample_args.add_argument('--out_nodata', type=str,
                               help='Use with --nonice2nodata - The no data value to use for resampled '
                                    'rasters. Default is to use the rasters no data value.')

    args = parser.parse_args()
    
    sea_ice_dir = args.sea_ice_directory
    last_update = args.last_update_date
    out_dir = args.out_directory
    nonice2nodata = args.nonice2nodata
    out_nodata = args.out_nodata
    north = args.north
    south = args.south
    dl_extent = args.dl_extent
    dl_concentration = args.dl_concentration
    overwrite = args.overwrite

    if not any([north, south]):
        logger.error('Must choose at least one of --north and/or --south')
        sys.exit()

    dl_sea_ice(last_update, dest_dir=sea_ice_dir, north=north, south=south,
               dl_extent=dl_extent, dl_concentration=dl_concentration,
               overwrite=overwrite)
    if nonice2nodata:
        resample_loop(sea_ice_dir, out_dir=out_dir, out_nodata=out_nodata, overwrite=overwrite)

