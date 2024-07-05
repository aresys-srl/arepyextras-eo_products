# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
SAOCOM product format reader
----------------------------
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
from arepytools.io import read_raster_with_raster_info
from arepytools.io.metadata import RasterInfo
from lxml import etree

import arepyextras.eo_products.saocom.l1_products.utilities as support
from arepyextras.eo_products.common.utilities import SARRadiometricQuantity


def read_channel_metadata(file_path: Path | str, channel_id: str) -> support.SAOCOMChannelMetadata:
    """Reading channel metadata info and storing them in a SAOCOMChannelMetadata dataclass.

    Parameters
    ----------
    file_path : Path | str
        Path to the metadata file, could be an .xml (GRD) or a .h5 file (SLC)

    Returns
    -------
    support.SAOCOMChannelMetadata
        SAOCOMChannelMetadata metadata dataclass
    """
    file_path = Path(file_path)
    root = etree.parse(file_path).getroot()
    root = root.find("Channel")

    # general info
    general_info = support.SAOCOMGeneralChannelInfo.from_metadata(node=root, channel_id=channel_id)

    # raster info
    raster_info = support.raster_info_from_metadata(node=root.find("RasterInfo"))

    # burst info
    burst_info = support.SAOCOMBurstInfo.from_metadata_node(node=root.find("BurstInfo"), raster_info=raster_info)

    # dataset info
    dataset_info = support.dataset_info_from_metadata(node=root.find("DataSetInfo"))

    # swath info
    swath_info = support.swath_info_from_metadata(node=root.find("SwathInfo"))

    # acquisition timeline
    acquisition_timeline = support.acquisition_timeline_from_metadata(node=root.find("AcquisitionTimeLine"))
    if acquisition_timeline is not None:
        acquisition_timeline.swl_changes = [1, [0], [raster_info.samples_step * (raster_info.samples - 1)]]

    # sampling constants
    sampling_constants = support.sampling_constants_from_metadata(node=root.find("SamplingConstants"))

    # pulse
    pulse = support.pulse_from_metadata(node=root.find("Pulse"))

    # state vectors
    state_vectors = support.SAOCOMStateVectors.from_metadata(node=root.find("StateVectorData"))
    gso = support.general_sar_orbit_from_saocom_state_vectors(state_vectors=state_vectors)

    # doppler centroid polynomial
    doppler_centroid_poly = support.doppler_poly_from_metadata(node=root, doppler_node_tag="DopplerCentroid")

    # doppler rate vector
    doppler_rate_poly = support.doppler_poly_from_metadata(node=root, doppler_node_tag="DopplerRate")

    # coordinates conversion
    coordinate_conversions = support.SAOCOMCoordinateConversions.from_metadata(node=root)

    return support.SAOCOMChannelMetadata(
        image_radiometric_quantity=SARRadiometricQuantity.SIGMA_NOUGHT,
        general_info=general_info,
        raster_info=raster_info,
        general_sar_orbit=gso,
        burst_info=burst_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        coordinate_conversions=coordinate_conversions,
        acquisition_timeline=acquisition_timeline,
        sampling_constants=sampling_constants,
        pulse=pulse,
        state_vectors=state_vectors,
    )


def read_channel_data(
    raster_file: Union[str, Path],
    raster_info: RasterInfo,
    block_to_read: list[int] = None,
) -> np.ndarray:
    """Reading SAOCOM channel raster files with raster info.

    Parameters
    ----------
    raster_file : Union[str, Path]
        Path to binary raster file to be read
    raster_info : RasterInfo
        channel raster info
    block_to_read : list[int], optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        by default None

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """

    return read_raster_with_raster_info(raster_file=raster_file, raster_info=raster_info, block_to_read=block_to_read)


def open_product(pf_path: Union[str, Path]) -> support.SAOCOMProduct:
    """Open a SAOCOM product.

    Parameters
    ----------
    pf_path : Union[str, Path]
        Path to the SAOCOM product

    Returns
    -------
    SAOCOMProduct
        SAOCOMProduct object corresponding to the input SAOCOM product
    """

    if not support.is_saocom_product(product=pf_path):
        raise support.InvalidSAOCOMProduct(f"{pf_path}")

    return support.SAOCOMProduct(path=pf_path)
