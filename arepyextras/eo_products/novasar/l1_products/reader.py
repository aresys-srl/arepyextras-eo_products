# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
NOVASAR product format reader
-----------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np
import zarr
from lxml import etree
from tifffile import imread

import arepyextras.eo_products.novasar.l1_products.utilities as support
from arepyextras.eo_products.common.utilities import (
    SARRadiometricQuantity,
    SARSamplingFrequencies,
)


def read_product_metadata(xml_path: Union[str, Path]) -> dict[str, support.NovaSAR1ChannelMetadata]:
    """Read NovaSAR-1 product metadata.

    Parameters
    ----------
    xml_path : Union[str, Path]
        Path to the .xml metadata file

    Returns
    -------
    dict[str, support.NovaSAR1ChannelMetadata]
        dictionary with channel id as keys and NovaSAR1ChannelMetadata dataclass as values
    """
    xml_path = Path(xml_path)

    _, product_type, channels_list, _ = support.get_basic_info_from_metadata(metadata_path=xml_path)
    acquisition_mode = support.get_acquisition_mode_from_product_type(prod_type=product_type)
    out_dict = dict.fromkeys(channels_list)

    # loading the xml file
    root = etree.parse(xml_path).getroot()

    # general info
    product_node = root.find("Product")
    source_attributes_node = root.find("Source_Attributes")
    image_generation_parameters_node = root.find("Image_Generation_Parameters")
    image_attributes_node = root.find("Image_Attributes")
    orbit_data_node = root.find("OrbitData")

    # CHANNEL INDEPENDENT INFO
    # raster info
    raster_info = support.raster_info_from_metadata_nodes(
        image_generation_parameters_node=image_generation_parameters_node,
        image_attributes_node=image_attributes_node,
        product_type=product_type,
    )

    # burst info
    burst_info = support.NovaSAR1BurstInfo(
        num=1,
        lines_per_burst=raster_info.lines,
        samples_per_burst=raster_info.samples,
        azimuth_start_times=np.array([raster_info.lines_start]),
        range_start_times=np.array([raster_info.samples_start]),
    )

    # dataset info
    dataset_info = support.dataset_info_from_metadata_node(
        source_attributes_node=source_attributes_node, prod_type=product_type
    )

    # swath info
    swath_info = support.NovaSAR1SwathInfo.from_metadata_nodes(
        source_attributes_node=source_attributes_node, acq_mode=acquisition_mode
    )

    # doppler centroid polynomial
    doppler_centroid_poly = support.doppler_centroid_poly_from_metadata_node(
        image_generation_parameters_node=image_generation_parameters_node, raster_info=raster_info
    )

    # incidence angles polynomial
    incidence_angle_poly = support.NovaSAR1IncidenceAnglePolynomial.from_metadata_node(
        image_generation_parameters_node=image_generation_parameters_node, raster_info=raster_info
    )

    # acquisition timeline
    acquisition_timeline = support.acquisition_timeline_from_metadata_node(
        image_generation_parameters_node=image_generation_parameters_node, raster_info=raster_info
    )

    # pulse
    pulse = support.pulse_info_from_metadata_nodes(
        image_generation_parameters_node=image_generation_parameters_node,
        source_attributes_node=source_attributes_node,
        samples_step=raster_info.samples_step,
    )

    # image calibration factor
    image_calibration_factor = 1 / np.sqrt(float(image_attributes_node.find("CalibrationConstant").text))

    # image radiometric quantity
    image_radiometric_quantity = image_generation_parameters_node.find("RadiometricScaling").text.lower()
    if "beta" in image_radiometric_quantity:
        image_radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT
    elif "gamma" in image_radiometric_quantity:
        image_radiometric_quantity = SARRadiometricQuantity.GAMMA_NOUGHT
    elif "sigma" in image_radiometric_quantity:
        image_radiometric_quantity = SARRadiometricQuantity.SIGMA_NOUGHT

    # sampling constants
    if product_type == support.NovaSAR1ProductType.SLC:
        sampling_constants = SARSamplingFrequencies(
            azimuth_freq_hz=1 / raster_info.lines_step,
            azimuth_bandwidth_freq_hz=float(
                image_generation_parameters_node.find("TotalProcessedAzimuthBandwidth").text
            ),
            range_freq_hz=1 / raster_info.samples_step,
            range_bandwidth_freq_hz=float(image_generation_parameters_node.find("TotalProcessedRangeBandwidth").text),
        )
    else:
        sampling_constants = SARSamplingFrequencies(
            azimuth_freq_hz=0,
            azimuth_bandwidth_freq_hz=0,
            range_freq_hz=0,
            range_bandwidth_freq_hz=0,
        )

    # state vectors and orbit
    state_vectors = support.NovaSAR1StateVectors.from_metadata_node(orbit_data_node=orbit_data_node)
    gso = support.general_sar_orbit_from_novasar1_state_vectors(state_vectors=state_vectors)

    # attitude info
    attitude = support.NovaSAR1Attitude.from_metadata_node(
        orbit_data_node=orbit_data_node, image_gen_params_node=image_generation_parameters_node
    )

    # coordinates conversion polynomials
    coordinates_conversions_poly = support.NovaSAR1CoordinateConversions.from_metadata_node(
        image_generation_parameters_node=image_generation_parameters_node, raster_info=raster_info
    )

    # checking orbit and attitude time axes equivalence
    assert attitude.time_step == state_vectors.time_step
    np.testing.assert_allclose(np.sum((attitude.time_axis - state_vectors.time_axis)), 0, atol=1e-10, rtol=0)

    # CHANNEL DEPENDENT INFO
    for channel in channels_list:
        # general info
        general_info = support.NovaSAR1GeneralChannelInfo.from_metadata_node(
            product_node=product_node,
            source_attributes_node=source_attributes_node,
            orbit_data_node=orbit_data_node,
            prod_type=product_type,
            channel_id=channel,
        )

        out_dict[channel] = support.NovaSAR1ChannelMetadata(
            general_info=general_info,
            image_calibration_factor=image_calibration_factor,
            image_radiometric_quantity=image_radiometric_quantity,
            raster_info=raster_info,
            burst_info=burst_info,
            dataset_info=dataset_info,
            swath_info=swath_info,
            acquisition_timeline=acquisition_timeline,
            doppler_centroid_poly=doppler_centroid_poly,
            incidence_angles_poly=incidence_angle_poly,
            coordinate_conversions=coordinates_conversions_poly,
            pulse=pulse,
            general_sar_orbit=gso,
            state_vectors=state_vectors,
            attitude=attitude,
            sampling_constants=sampling_constants,
        )

    return out_dict


def read_channel_data(
    raster_file: Union[str, Path],
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Reading NovaSAR-1 tif channel data file.

    Parameters
    ----------
    raster_file : Union[str, Path]
        Path to .tif raster file to be read
    block_to_read : list[int], optional
        data block to be read, to be specified as a list of 4 integers, in the form:
            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        by default None

    scaling_conversion : float, optional
        scaling conversion to be multiplied to the data read

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """
    img_store = imread(raster_file, aszarr=True)
    z = zarr.open(img_store, mode="r")
    if block_to_read is None:
        target_area = z[:]
    else:
        target_area = z[
            block_to_read[0] : block_to_read[0] + block_to_read[2],
            block_to_read[1] : block_to_read[1] + block_to_read[3],
        ]
    img_store.close()

    # SLC image is a two page tif, with real and imaginary part that must be recombined
    if target_area.ndim == 3:
        target_area = target_area[:, :, 0] + 1j * target_area[:, :, 1]

    # applying input scaling factor
    target_area = target_area * scaling_conversion

    return target_area


def open_product(pf_path: Union[str, Path]) -> support.NovaSAR1Product:
    """Open a NovaSAR-1 product.

    Parameters
    ----------
    pf_path : Union[str, Path]
        Path to the NovaSAR-1 product

    Returns
    -------
    NovaSAR1Product
        NovaSAR1Product object corresponding to the input NovaSAR-1 product
    """

    if not support.is_novasar_1_product(product=pf_path):
        raise support.InvalidNovaSAR1Product(f"{pf_path}")

    return support.NovaSAR1Product(path=pf_path)
