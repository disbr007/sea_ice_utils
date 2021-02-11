import argparse
import dateutil.parser
import logging
import numpy as np
import os
from pathlib import Path

import geopandas as gpd
from shapely.geometry import Polygon
from tqdm import tqdm

from RasterWrapper import Raster

tqdm.pandas()

logger = logging.getLogger(__name__)
logger.setLevel('INFO')
sh = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
sh.setFormatter(formatter)
logger.addHandler(sh)

# CONSTANTS
# Default dir for sea ice if CLI arg not provided
SEA_ICE_DIR = r'E:\disbr007\projects\coastline\noaa_sea_ice'
# Part of raster file name
SEA_ICE_VERSION = '3.0'
SEA_ICE_FMT = 'tif'
ICE_PATH_FLD = 'sea_ice_path'
SEA_ICE_CON = 'SEAICECONC'
DEF_DATE_FIELD = 'acq_time'
NORTH = 'north'
SOUTH = 'south'

# Concentration raster values that are not sea_ice concentration
CON_MISS = 2550  # missing
CON_LAND = 2540  # land
CON_COAST = 2530  # coast
CON_POL = 2510  # pole hole
NON_CON_VALS = [CON_MISS, CON_LAND, CON_COAST, CON_POL]


def which_pole(geometry):
    if geometry.centroid.y > 0:
        pole = NORTH
    else:
        pole = SOUTH

    return pole


def build_sea_ice_path(date : str,
                       pole : str,
                       sea_ice_dir: str = SEA_ICE_DIR) -> str:
    parsed_date = dateutil.parser.parse(date)
    y, m, d = parsed_date.year, parsed_date.month, parsed_date.day
    sea_ice_path = Path(sea_ice_dir) / pole / 'daily' / 'geotiff' / str(y) / \
                   '{}_{}'.format(str(m).zfill(2), parsed_date.strftime('%b')) / \
                   '{}_{}{}{}_concentration_v{}.{}'.format(pole[0].upper(),
                                                           y,
                                                           str(m).zfill(2),
                                                           str(d).zfill(2),
                                                           SEA_ICE_VERSION,
                                                           SEA_ICE_FMT)
    return str(sea_ice_path)


def sample_sea_ice(geometry: Polygon,
                   sea_ice_path: str,
                   expand_search_area: bool = True,
                   null_values: list = NON_CON_VALS) -> float:
    if not os.path.exists(sea_ice_path):
        logger.warning('sea-ice path not found for: {}'.format(Path(sea_ice_path).stem))
        return np.nan
    r = Raster(sea_ice_path)
    concentration = r.SampleWindow(center_point=(geometry.centroid.y, geometry.centroid.x),
                                   window_size=(3, 3),
                                   grow_window=expand_search_area,
                                   null_values=null_values,)
    if concentration == r.nodata_val:
        concentration = np.nan
        logger.warning('NoData returned from sampling: '
                       '{}'.format(Path(sea_ice_path).stem))
    else:
        concentration = concentration / 10

    return round(concentration, 2)


def determine_sea_ice(footprint_path: str,
                      output_path: str,
                      date_field: str = DEF_DATE_FIELD,
                      sea_ice_dir: str = SEA_ICE_DIR,
                      expand_search_area: bool = True,
                      sea_ice_field: str = SEA_ICE_CON
                      ):
    # Load footprint
    logger.info('Loading footprint...')
    fp = gpd.read_file(footprint_path)

    # Get path to associated sea-ice raster
    logger.info('Locating paths to sea ice concentration '
                'rasters for each footprint...')
    fp[ICE_PATH_FLD] = fp.apply(
        lambda x: build_sea_ice_path(date=x[date_field],
                                     pole=which_pole(x.geometry),
                                     sea_ice_dir=sea_ice_dir), axis=1)

    # Sample each raster and return value
    logger.info('Determining sea ice concentrations...')
    fp[sea_ice_field] = fp.progress_apply(
        lambda x: sample_sea_ice(geometry=x.geometry,
                                 sea_ice_path=x[ICE_PATH_FLD],
                                 null_values=NON_CON_VALS,
                                 expand_search_area=expand_search_area), axis=1)

    # Write results
    logger.info('Writing new footprint to: {}'.format(output_path))
    fp.drop(columns=ICE_PATH_FLD, inplace=True)
    fp.to_file(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_footprint', type=os.path.abspath, required=False,
                        help='Path to footprint to calculate sea ice concentrations '
                             'on.')
    parser.add_argument('-o', '--output_footprint', type=os.path.abspath, required=False,
                        help='Path to write copy of footprint with added sea ice '
                             'concentration field.')
    parser.add_argument('--date_field', default=DEF_DATE_FIELD,
                        help='Field in footprint storing date.')
    parser.add_argument('--sea_ice_dir', default=SEA_ICE_DIR,
                        help='Directory holding sea ice concentration rasters. '
                             'Use one level above the pole. For example, the expected '
                             'paths are like:\n '
                             r'../noaa_sea_ice/north/daily/geotiff/2017/12_Dec/'
                             r'N_20171231_concentration_v3.0.tif \n'
                             r'and "../noaa_sea_ice" should be provided.')
    parser.add_argument('--limit_search_area', action='store_true',
                        help='Use to only sample at the center point of each footprint. '
                             'By default, the window will grow until it encounters a value '
                             'that is a valid concentration. This facilitates determining '
                             'concentrations for images all or mostly over land.')
    parser.add_argument('--sea_ice_field', default=SEA_ICE_CON,
                        help='The name of the field to create with sea ice concentrations, '
                             'defaults to: {}'.format(SEA_ICE_CON))

    args = parser.parse_args()

    footprint_path = args.input_footprint
    output_path = args.output_footprint
    date_field = args.date_field
    sea_ice_dir = args.sea_ice_dir
    expand_search_area = not args.limit_search_area
    sea_ice_field = args.sea_ice_field

    determine_sea_ice(footprint_path=footprint_path,
                      output_path=output_path,
                      date_field=date_field,
                      sea_ice_dir=sea_ice_dir,
                      expand_search_area=expand_search_area,
                      sea_ice_field=sea_ice_field
                      )

