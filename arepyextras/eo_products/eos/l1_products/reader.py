# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
EOS04 product format reader
---------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr
from lxml import etree
from tifffile import imread

import arepyextras.eo_products.eos.l1_products.utilities as support
from arepyextras.eo_products.common.utilities import (
    SARRadiometricQuantity,
    SARSamplingFrequencies,
)


def read_product_metadata(xml_path: str | Path, channels: list[str]) -> dict[str, support.EOS04ChannelMetadata]:
    """Reading EOS04 product channel metadata.

    Parameters
    ----------
    xml_path : str | Path
        path to the annotation xml file
    channels : list[str]
        channels ids for the current product

    Returns
    -------
    dict[str, support.EOS04ChannelMetadata]
        dictionary of EOS04ChannelMetadata dataclasses as values, channel name as key
    """

    xml_path = Path(xml_path)

    # loading the xml file
    root = etree.parse(xml_path).getroot()
    product_type = root.find("ProductType").text
    source_attributes_node = root.find("SourceAttributes")
    orbit_node = source_attributes_node.find("OrbitAndAttitude/OrbitInformation")
    attitude_node = source_attributes_node.find("OrbitAndAttitude/AttitudeInformation")
    image_generation_node = root.find("ImageGenerationParameters")
    image_attributes_node = root.find("ImageAttributes")

    # channel independent info
    state_vectors = support.EOS04StateVectors.from_metadata_node(orbit_information_node=orbit_node)
    gso = support.general_sar_orbit_from_eos04_state_vectors(state_vectors=state_vectors)
    attitude = support.EOS04Attitude.from_metadata_node(attitude_information_node=attitude_node)
    # forcing radiometric input as Beta Nought
    radiometric_quantity = SARRadiometricQuantity.BETA_NOUGHT

    channels_dict = dict.fromkeys(channels)
    for channel in channels:

        beam, polarization = support.unpack_channel_name(channel)

        # general info
        general_info = support.EOS04GeneralChannelInfo.from_metadata_node(
            source_attributes_node=source_attributes_node, channel_id=channel, product_type=product_type
        )

        # dataset info
        dataset_info = support.dataset_info_from_metadata_node(
            source_attributes_node=source_attributes_node, projection=general_info.projection
        )

        # swath info
        # TODO: rank? steering rate poly?
        if general_info.product_type == support.EOS04ProductType.SLC:
            swath_info = support.EOS04SwathInfo.from_metadata_nodes(
                image_generation_parameters_node=image_generation_node, polarization=polarization, beam_id=beam
            )
        else:
            # TODO: prf for GRD?
            swath_info = support.EOS04SwathInfo(rank=0, azimuth_steering_rate_poly=(0, 0, 0), prf=0)

        # raster info
        raster_info = support.raster_info_from_metadata_nodes(
            image_generation_parameters_node=image_generation_node,
            image_attributes_node=image_attributes_node,
            beam_id=beam,
            polarization=polarization,
            product_type=general_info.product_type,
        )

        # burst info
        if general_info.product_type == support.EOS04ProductType.SLC:
            burst_info = support.EOS04BurstInfo.from_metadata_node(
                image_generation_parameters_node=image_generation_node, polarization=polarization, beam_id=beam
            )
        else:
            burst_info = support.EOS04BurstInfo(
                num=1,
                first_valid_lines=np.array([0]),
                first_valid_samples=np.array([0]),
                lines_per_burst=raster_info.lines,
                samples_per_burst=raster_info.samples,
                azimuth_start_times=np.array([raster_info.lines_start]),
                range_start_times=np.array([raster_info.samples_start]),
            )

        # pulse info (no chirp direction)
        # TODO: this is fake
        pulse = support.pulse_info_from_metadata_nodes(
            source_attributes_node=source_attributes_node, samples_step=raster_info.samples_step
        )

        # doppler centroid poly
        doppler_centroid_poly = support.doppler_centroid_poly_from_metadata_node(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # doppler rate poly
        doppler_rate_poly = support.doppler_rate_poly_from_metadata_node(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # acquisition timeline
        # TODO: this is fake
        acquisition_timeline = support.acquisition_timeline_from_metadata_node(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # coordinate conversion
        coordinate_conversion = support.EOS04CoordinateConversions.from_metadata_node(
            image_generation_parameters_node=image_generation_node, raster_info=raster_info
        )

        # sampling constants
        sampling_constants = SARSamplingFrequencies(
            azimuth_bandwidth_freq_hz=float(
                image_generation_node.find("SarProcessingInformation/TotalProcessedAzimuthBandwidth").text
            ),
            azimuth_freq_hz=1 / raster_info.lines_step,
            range_bandwidth_freq_hz=float(
                image_generation_node.find("SarProcessingInformation/TotalProcessedRangeBandwidth").text
            ),
            range_freq_hz=1 / raster_info.samples_step,
        )

        # image calibration factor
        calibration_constant_db = [
            float(c.text)
            for c in image_attributes_node.findall("CalibrationConstant_Beta0")
            if c.get("pol") == polarization.name
        ][0]
        calibration_factor = 1 / (10 ** (calibration_constant_db / 20))

        channels_dict[channel] = support.EOS04ChannelMetadata(
            channel_id=channel,
            general_info=general_info,
            raster_info=raster_info,
            burst_info=burst_info,
            state_vectors=state_vectors,
            general_sar_orbit=gso,
            dataset_info=dataset_info,
            sampling_constants=sampling_constants,
            swath_info=swath_info,
            attitude=attitude,
            image_calibration_factor=calibration_factor,
            image_radiometric_quantity=radiometric_quantity,
            doppler_centroid_poly=doppler_centroid_poly,
            doppler_rate_poly=doppler_rate_poly,
            acquisition_timeline=acquisition_timeline,
            coordinate_conversions=coordinate_conversion,
            pulse=pulse,
        )

    return channels_dict


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
) -> np.ndarray:
    """Reading EOS04 tif channel data file.

    Parameters
    ----------
    raster_file : str | Path
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


def open_product(path: str | Path) -> support.EOS04Product:
    """Open an EOS04 product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the EOS04 product

    Returns
    -------
    EOS04Product
        EOS04Product object corresponding to the input product
    """
    path = Path(path)

    if not support.is_eos04_product(product=path):
        raise support.InvalidEOS04Product(f"{path}")

    return support.EOS04Product(path=path)
