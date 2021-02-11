# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 10:20:36 2019

@author: disbr007

"""
import copy
import numpy as np
import numpy.ma as ma
from typing import Union
import pathlib

from osgeo import gdal, osr  # ogr
# from shapely.geometry import Polygon
from shapely.geometry import box
import geopandas as gpd

from misc_utils.logging_utils import create_logger
from misc_utils.gdal_tools import clip_minbb, gdal_polygonize

logger = create_logger(__name__, 'sh', 'DEBUG')

gdal.UseExceptions()


class Raster:
    """
    A class wrapper using GDAL to simplify working with rasters.
    Basic functionality:
        -read array from raster
        -read stacked array
        -write array out with same metadata
        -sample raster at point in geocoordinates
        -sample raster with window around point
    """

    def __init__(self, raster_path):
        self.src_path = raster_path
        self.data_src = gdal.Open(raster_path)
        self.geotransform = self.data_src.GetGeoTransform()

        self.prj = osr.SpatialReference()
        self.prj.ImportFromWkt(self.data_src.GetProjectionRef())
        # try:
        #     self.epsg = self.prj.GetAttrValue("PROJCS|GEOGCS|AUTHORITY", 1)
        # except KeyError as e:
        #     logger.error(""""Trying to get EPSG of unprojected Raster,
        #                      not currently supported.""")
        #     raise e
        self.prj.wkt = self.prj.ExportToWkt()

        self.x_sz = self.data_src.RasterXSize
        self.y_sz = self.data_src.RasterYSize
        self.depth = self.data_src.RasterCount

        self.x_origin = self.geotransform[0]
        self.y_origin = self.geotransform[3]

        self.pixel_width = self.geotransform[1]
        self.pixel_height = self.geotransform[5]

        # TODO: make this class @property, when called, if None, try to get from first band,
        #  otherwise default. This will allow setting nodata_val explicity.
        self.nodata_val = self.data_src.GetRasterBand(1).GetNoDataValue()
        self.dtype = self.data_src.GetRasterBand(1).DataType

        # Get the raster as an array
        # Defaults to band 1 -- use ReadArray() to return stack
        # of multiple bands
        # TODO: Init these here, but then call as method to avoid loading all on Raster() call
        # @property for lazy evaluation?
        self.Array = self.data_src.ReadAsArray()
        self.Mask = self.Array == self.nodata_val
        self.MaskedArray = ma.masked_array(self.Array, mask=self.Mask)
        np.ma.set_fill_value(self.MaskedArray, self.nodata_val)

    # def Masked_Array(self):
    #     masked_array = ma.masked_array(self.Array, mask=self.Mask)
    #     masked_array = np.ma.set_fill_value(self.nodata_val, masked_array)


    def get_projwin(self):
        """Get projwin ordered."""
        gt = self.geotransform

        ulx = gt[0]
        uly = gt[3]
        lrx = ulx + (gt[1] * self.x_sz)
        lry = uly + (gt[5] * self.y_sz)

        return ulx, uly, lrx, lry

    def raster_bounds(self):
        """
        GDAL only version of getting bounds for a single raster.
        """
        gt = self.geotransform

        ulx = gt[0]
        uly = gt[3]
        lrx = ulx + (gt[1] * self.x_sz)
        lry = uly + (gt[5] * self.y_sz)

        return ulx, lry, lrx, uly

    def raster_bbox(self):
        """
        Reorder projwin to conform to shapely.geometry.Polygon ordering and creates
        the shapely Polygon.

        Returns
        -------
        shapely.geometry.Polygon

        """
        ulx, uly, lrx, lry = self.get_projwin()
        # bbox = Polygon([lrx, lry, ulx, uly])
        bbox = box(lrx, lry, ulx, uly)

        return bbox

    def bbox2gdf(self):
        gdf = gpd.GeoDataFrame(geometry=[self.raster_bbox()],
                               crs=self.prj.wkt)

        return gdf

    def GetBandAsArray(self, band_num, mask=True):
        """
        Parameters
        ----------
        band_num : INT
            The band number to return.
        mask : BOOLEAN
            Whether to mask to array that is returned

        Returns
        -------
        np.ndarray

        """
        band = self.data_src.GetRasterBand(band_num)
        band_arr = band.ReadAsArray()
        if mask:
            if self.nodata_val is None:
                self.nodata_val = band.GetNoDataValue()
            mask = band_arr == self.nodata_val
            band_arr = ma.masked_array(band_arr, mask=mask)

        return band_arr

    def ndvi_array(self, red_num, nir_num):
        """Calculate NDVI from multispectral bands"""
        red = self.GetBandAsArray(red_num)
        nir = self.GetBandAsArray(nir_num)
        ndvi = (nir - red) / (nir + red)

        return ndvi

    def mndwi_array(self, green_num, swir_num):
        green = self.GetBandAsArray(green_num)
        swir = self.GetBandAsArray(swir_num)
        mndwi = (green - swir) / (green + swir)

        return mndwi

    def ArrayWindow(self, projWin):
        """
        Takes a projWin in geocoordinates, converts
        it to pixel coordinates and returns the
        array referenced
        """
        xmin, ymin, xmax, ymax = self.projWin2pixelWin(projWin)
        self.arr_window = self.Array[ymin:ymax, xmin:xmax]

        return self.arr_window

    def geo2pixel(self, geocoord):
        """
        Convert geographic coordinates to pixel coordinates
        """

        py = int(np.around((geocoord[0] - self.geotransform[3]) / self.geotransform[5]))
        px = int(np.around((geocoord[1] - self.geotransform[0]) / self.geotransform[1]))

        return (py, px)

    def projWin2pixelWin(self, projWin):
        """
        Convert projWin in geocoordinates to pixel coordinates
        """
        ul = (projWin[1], projWin[0])
        lr = (projWin[3], projWin[2])

        puly, pulx = self.geo2pixel(ul)
        plry, plrx = self.geo2pixel(lr)

        return [pulx, puly, plrx, plry]

    def ReadStackedArray(self, stacked=True):
        '''
        Read raster as array, stacking multiple bands as either stacked array or multiple arrays
        stacked: boolean - specify False to return a separate array for each band
        '''
        # Get number of bands in raster
        num_bands = self.data_src.RasterCount
        # For each band read as array and add to list
        band_arrays = []
        for band in range(num_bands):
            band_arr = self.data_src.GetRasterBand(band).ReadAsArray()
            band_arrays.append(band_arr)

        # If stacked is True, stack bands and return
        if stacked:
            # Control for 1 band rasters as stacked=True is the default
            if num_bands > 1:
                stacked_array = np.dstack(band_arrays)
            else:
                stacked_array = band_arrays[0]

            return stacked_array

        # Return list of band arrays
        else:
            return band_arrays

    def stack_arrays(self, arrays):
        """
        Stack a list of arrays into a np.dstack array, changing fill values to match the
        source.

        Parameters
        ----------
        arrays: list
            List of arrays to be stacked, not including source array

        Returns
        -------
        np.array : Depth = len(arrays)
        """
        logger.debug('Stacking arrays...')
        src_arr = self.MaskedArray
        stacked = np.dstack([src_arr])

        for i, arr in enumerate(arrays):
            if np.ma.isMaskedArray(arr):
                arr_mask = arr.mask
                arr.set_fill_value(self.nodata_val)
                arr = arr.filled(arr.fill_value)
                np.ma.masked_where(arr_mask is True, arr)
            stacked = np.dstack([stacked, arr])
        # The process of stacking is change the fill value - change back to nodata_val
        stacked.set_fill_value(self.nodata_val)

        return stacked

    def WriteArray(self, array, out_path, stacked=False, fmt='GTiff',
                   dtype=None, nodata_val=None):
        """
        Writes the passed array with the metadata of the current raster object
        as new raster.
        """
        # Get dimensions of input array
        dims = len(array.shape)

        # try:
        if dims == 3:
            rows, cols, depth = array.shape
            stacked = True
        elif dims == 2:
        # except ValueError:
            rows, cols = array.shape
            depth = 1

        # Handle dtype
        if not dtype:
            # Use original dtype
            dtype = self.dtype
        # Handle NoData value
        if nodata_val is None:
            if self.nodata_val is not None:
                nodata_val = self.nodata_val
            else:
                logger.warning('Unable to determine NoData value of {}, '
                               'using -9999'.format(self.src_path))
                nodata_val = -9999

        # Create output file
        driver = gdal.GetDriverByName(fmt)
        try:
            dst_ds = driver.Create(out_path, self.x_sz, self.y_sz, bands=depth,
                                   eType=dtype)
        except:
            logger.error('Error creating: {}'.format(out_path))
        dst_ds.SetGeoTransform(self.geotransform)
        dst_ds.SetProjection(self.prj.ExportToWkt())

        # Loop through each layer of array and write as band
        for i in range(depth):
            if stacked:
                lyr = array[:, :, i].filled()
                band = i + 1
                dst_ds.GetRasterBand(band).WriteArray(lyr)
                dst_ds.GetRasterBand(band).SetNoDataValue(nodata_val)
            else:
                # logger.info(array.dtype)
                band = i + 1
                if isinstance(array, np.ma.MaskedArray):
                    dst_ds.GetRasterBand(band).WriteArray(array.filled(self.nodata_val))
                else:
                    dst_ds.GetRasterBand(band).WriteArray(array)
                dst_ds.GetRasterBand(band).SetNoDataValue(nodata_val)

        dst_ds = None

    def WriteMask(self, out_path, **kwargs):
        self.WriteArray(self.Mask, out_path=out_path, **kwargs)

    def WriteMaskVector(self, out_vec, out_mask_img=None, **kwargs):
        if out_mask_img is None:
            out_mask_img = r'/vsimem/mask.tif'
        self.WriteMask(out_path=out_mask_img)
        gdal_polygonize(img=out_mask_img, out_vec=out_vec, **kwargs)

    def NDVI(self, out_path, red_num, nir_num):
        ndvi_arr = self.ndvi_array(red_num, nir_num)
        self.WriteArray(ndvi_arr, out_path, stacked=False)

    def mNDWI(self, out_path, green_num, swir_num):
            mndwi_arr = self.mndwi_array(green_num, swir_num)
            self.WriteArray(mndwi_arr, out_path, stacked=False)

    def create_brightness(self, bands: list, out_path: Union[str, pathlib.PurePath]):
        for i, b in enumerate(bands):
            a = self.GetBandAsArray(b)
            if i == 0:
                tot = copy.deepcopy(a)
            else:
                tot = np.ma.add(tot, a)
                a = None

        if out_path:
            self.WriteArray(tot, out_path)

        return tot

    def extract_bands(self, bands, out_path):
        arrs = []
        for b in bands:
            b_arr = self.GetBandAsArray(b, mask=True)
            arrs.append(b_arr)
        
        stacked = np.dstack([arrs])
        
        self.WriteArray(stacked, out_path=out_path, stacked=True)

        return stacked

    def SamplePoint(self, point):
        '''
        Samples the current raster object at the given point. Must be the
        sampe coordinate system used by the raster object.
        point: tuple of (y, x) in geocoordinates
        '''
        # Convert point geocoordinates to array coordinates
        py = int(np.around((point[0] - self.geotransform[3]) / self.geotransform[5]))
        px = int(np.around((point[1] - self.geotransform[0]) / self.geotransform[1]))
        # Handle point being out of raster bounds
        try:
            point_value = self.Array[py, px]
        except IndexError as e:
            logger.warning('Point not within raster bounds.')
            logger.warning(e)
            point_value = None
        return point_value

    def SampleWindow(self, center_point, window_size, agg='mean',
                     grow_window=False, max_grow=100000,
                     null_values=None):
        """
        Samples the current raster object using a window centered
        on center_point. Assumes 1 band raster.
        center_point: tuple of (y, x) in geocoordinates
        window_size: tuple of (y_size, x_size) as number of pixels (must be odd)
        agg: type of aggregation, default is mean, can also me sum, min, max
        grow_window: set to True to increase the size of the window until a valid value is
                        included in the window
        max_grow: the maximum area (x * y) the window will grow to
        """


        def window_bounds(window_size, py, px):
            """
            Takes a window size and center pixel coords and
            returns the window bounds as ymin, ymax, xmin, xmax
            window_size: tuple (3,3)
            py: int 125
            px: int 100
            """
            # Get window around center point
            # Get size in y, x directions
            y_sz = window_size[0]
            y_step = int(y_sz / 2)
            x_sz = window_size[1]
            x_step = int(x_sz / 2)

            # Get pixel locations of window bounds
            ymin = py - y_step
            ymax = py + y_step + 1  # slicing doesn't include stop val so add 1
            xmin = px - x_step
            xmax = px + x_step + 1

            return ymin, ymax, xmin, xmax

        # Convert center point geocoordinates to array coordinates
        py = int(np.around((center_point[0] - self.geotransform[3]) / self.geotransform[5]))
        px = int(np.around((center_point[1] - self.geotransform[0]) / self.geotransform[1]))

        # Handle window being out of raster bounds
        if null_values:
            null_values.append(self.nodata_val)
        else:
            null_values = [self.nodata_val]
        try:
            growing = True
            while growing:
                ymin, ymax, xmin, xmax = window_bounds(window_size, py, px)
                window = self.Array[ymin:ymax, xmin:xmax].astype(np.float32)
                window = np.where(np.isin(window, null_values), np.nan, window)

                # Test for window with all nans to avoid getting 0's for all nans
                # Returns an array of True/False where True is valid values
                window_valid = window == window

                if True in window_valid:
                    # Window contains at least one valid value, do aggregration
                    agg_lut = {
                        'mean': np.nanmean(window),
                        'sum': np.nansum(window),
                        'min': np.nanmin(window),
                        'max': np.nanmax(window)
                        }
                    window_agg = agg_lut[agg]

                    # Do not grow if valid values found
                    growing = False

                else:
                    # Window all nan's, return nan value (arbitratily picking -9999)
                    # If grow_window is True, increase window (y+2, x+2)
                    if grow_window:
                        window_size = (window_size[0] + 2, window_size[1] + 2)
                    # If grow_window is False, return no data and exit while loop
                    else:
                        window_agg = self.nodata_val
                        growing = False

        except IndexError as e:
            logger.warning('Window bounds not within raster bounds.')
            # logger.error(e)
            window_agg = None

        return window_agg


def same_srs(raster1, raster2):
    """
    Compare the spatial references of two rasters.

    Parameters
    ----------
    raster1 : os.path.abspath
        Path to the first raster.
    raster2 : os.path.abspath
        Path to the second raster.

    Returns
    -------
    BOOL : True is match.

    """
    r1 = Raster(raster1)
    r1_srs = r1.prj
    # r1 = None

    r2 = Raster(raster2)
    r2_srs = r2.prj
    # r2 = None

    result = r1_srs.IsSame(r2_srs)
    if result == 1:
        same = True
    elif result == 0:
        same = False
    else:
        logger.error('Unknown return value from IsSame, expected 0 or 1: {}'.format(result))
    return same


def stack_rasters(rasters, minbb=True, rescale=False):
    """
    Stack single band rasters into a multiband raster.

    Parameters
    ----------
    rasters : list
        List of rasters to stack. Reference raster for NoData value, projection, etc.
        is the first raster provided.
    rescale : bool
        True to rescale rasters to 0 to 1.

    Returns
    -------
    np.array : path.abspath : path to write multiband raster to.

    """
    # TODO: Add default clipping to reference window
    if minbb:
        logger.info('Clipping to overlap area...')
        rasters = clip_minbb(rasters, in_mem=True, out_format='vrt')

    # Determine a raster to use for reference
    ref = Raster(rasters[0])

    # Check for SRS match between reference and other rasters
    srs_matches = [same_srs(rasters[0], r) for r in rasters[1:]]
    if not all(srs_matches):
        logger.warning("""Spatial references do not match, match status between
                          reference and rest:\n{}""".format('\n'.join(srs_matches)))

    # Initialize the stacked array with just the reference array
    if ref.depth > 1:
        stacked = ref.GetBandAsArray(1, mask=True)
        if rescale:
            stacked = (stacked - stacked.min()) / (stacked.max() - stacked.min())

        for i in range(ref.depth-1):
            band = ref.GetBandAsArray(i+2, mask=True)
            if rescale:
                band = (band - band.min()) / (band.max() - band.min())
            stacked = np.ma.dstack([stacked, band])

    for i, rast in enumerate(rasters[1:]):
        ma = Raster(rast).MaskedArray
        if np.ma.isMaskedArray(ma):
            if rescale:
                # Rescale to between 0 and 1
                ma = (ma - ma.min())/ (ma.max() - ma.min())
            # Replace mask/nodata value in array with reference values
            # ma_mask = ma.mask
            # ma = np.ma.masked_where(ma_mask, ma)
            # ma.set_fill_value(ref.nodata_val)
            # ma = ma.filled(ma.fill_value)

        stacked = np.ma.dstack([stacked, ma])

        ma = None

    # Revert to original NoData value (stacking changes)
    # stacked.set_fill_value(ref.nodata_val)
    ref = None

    return stacked
