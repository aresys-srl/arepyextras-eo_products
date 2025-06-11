# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Envisat & ERS product format reader
-----------------------------------
"""

from __future__ import annotations

from pathlib import Path

import arepyextras.eo_products.asar.l1_products.utilities as support
import epr
import numpy as np
from arepyextras.eo_products.common.utilities import (
    SARRadiometricQuantity,
    StandardSARAcquisitionMode,
)


def read_channel_metadata(
    file_path: Path | str, channel_id: str
) -> support.ASARChannelMetadata:
    """Reading metadata for current ASAR input product related to the selected channel.

    Parameters
    ----------
    file_path : Path | str
        Path to the ASAR binary product
    channel_id : str
        channel of choice

    Returns
    -------
    support.ASARChannelMetadata
        metadata for the selected channel
    """
    file_path = Path(file_path)
    mph, sph = support.read_product_headers(product=file_path)
    acquisition_mode = support.ASARAcquisitionMode.from_str(file_path.name)
    if acquisition_mode == support.ASARAcquisitionMode.STRIPMAP:
        acquisition_mode_std = StandardSARAcquisitionMode.STRIPMAP
    else:
        acquisition_mode_std = StandardSARAcquisitionMode.WAVE
    product_type = support.ASARProductType.from_str(sph.sample_type)
    projection = support.get_projection_from_product_type(product_type)

    product = epr.open(str(file_path))
    main_params_dataset = product.get_dataset("MAIN_PROCESSING_PARAMS_ADS")
    doppler_centroid_coeffs_dataset = product.get_dataset("DOP_CENTROID_COEFFS_ADS")
    geolocation_grid_dataset = product.get_dataset("GEOLOCATION_GRID_ADS")

    # mds1_sq_dataset = product.get_dataset("MDS1_SQ_ADS")
    # mds1_dataset = product.get_dataset("MDS1")
    # chirp_params_dataset = product.get_dataset("CHIRP_PARAMS_ADS")

    raster_info = support.raster_info_from_record(
        main_params_dataset=main_params_dataset,
        geolocation_record=geolocation_grid_dataset.read_record(0),
        product_type=product_type,
    )

    state_vectors = support.ASARStateVectors.from_metadata(
        main_params_dataset=main_params_dataset, orbit_direction=sph.orbit_direction
    )
    orbit = support.orbit_from_state_vectors(state_vectors=state_vectors)

    doppler_centroid_poly = support.doppler_centroid_poly_from_dataset(
        dc_params_dataset=doppler_centroid_coeffs_dataset
    )
    doppler_rate_poly = support.doppler_rate_poly_from_dataset(
        main_params_dataset=main_params_dataset,
    )

    coordinate_conversion = None
    if "SR_GR_ADS" in product.get_dataset_names():
        sr_gr_dataset = product.get_dataset("SR_GR_ADS")
        coordinate_conversion = support.ASARCoordinateConversions.from_dataset(
            slant_to_ground_dataset=sr_gr_dataset, raster_info=raster_info
        )
    else:
        coordinate_conversion = support.ASARCoordinateConversions.from_orbit(
            orbit=orbit, raster_info=raster_info
        )

    ds_channel_index = [p.name.lower() for p in sph.mds_polarizations].index(
        channel_id.split("_")[-1]
    )
    for ds_id, main_processing_params in enumerate(main_params_dataset):
        if not ds_id == ds_channel_index:
            continue

        burst_info = support.ASARBurstInfo(
            num=1,
            lines_per_burst=raster_info.lines,
            samples_per_burst=raster_info.samples,
            azimuth_start_times=np.array([raster_info.lines_start]),
            range_start_times=np.array([raster_info.samples_start]),
        )

        dataset_info = support.dataset_info_from_record(
            product_type=product_type,
            acquisition_mode=acquisition_mode,
            main_params_record=main_processing_params,
            mph=mph,
            sph=sph,
        )
        sampling_constants = support.sampling_constants_from_record(
            main_params_record=main_processing_params
        )
        swath_info = support.ASARSwathInfo.from_record(
            main_params_record=main_processing_params
        )
        general_info = support.ASARGeneralChannelInfo(
            product_name=mph.product,
            channel_id=channel_id,
            swath=sph.swath,
            product_type=product_type,
            polarization=sph.mds_polarizations[ds_id],
            projection=projection,
            acquisition_mode=acquisition_mode,
            acquisition_mode_std=acquisition_mode_std,
            orbit_direction=sph.orbit_direction,
            signal_frequency=dataset_info.fc_hz,
            acq_start_time=mph.sensing_start,
            acq_stop_time=mph.sensing_stop,
        )

        calibration_factor = support.get_calibration_factor(
            index=ds_id, main_params_record=main_processing_params
        )

        pulse_info = support.pulse_info_from_record(
            main_params_record=main_processing_params
        )

    product.close()
    return support.ASARChannelMetadata(
        general_info=general_info,
        orbit=orbit,
        image_calibration_factor=calibration_factor,
        image_radiometric_quantity=SARRadiometricQuantity.BETA_NOUGHT,
        raster_info=raster_info,
        burst_info=burst_info,
        dataset_info=dataset_info,
        swath_info=swath_info,
        sampling_constants=sampling_constants,
        doppler_centroid_poly=doppler_centroid_poly,
        doppler_rate_poly=doppler_rate_poly,
        pulse=pulse_info,
        coordinate_conversions=coordinate_conversion,
        state_vectors=state_vectors,
    )


def read_channel_data(
    raster_file: str | Path,
    block_to_read: list[int] | None = None,
    scaling_conversion: float = 1,
):
    """Reading ASAR data file from binary.

    NOTE: the range axis is actually reversed. When reading data, the ROI range index should be provided relative to
    the end of the raster range axis and not to the start, i.e. range_roi_index = samples - range_roi_index.
    Data portion is the flip alongside range direction and returned to the user.

    Parameters
    ----------
    raster_file : str | Path
        Path to .N1 binary file
    block_to_read : list[int] | None, optional
        data block to be read, to be specified as a list of 4 integers, in the form:

            0. first line to be read
            1. first sample to be read
            2. total number of lines to be read
            3. total number of samples to be read

        by default None
    scaling_conversion : float, optional
        scaling conversion to be applied to the data read

    Returns
    -------
    np.ndarray
        numpy array containing the data read from raster file, with shape (lines, samples)
    """

    product = epr.open(str(raster_file))
    if "proc_data" not in product.get_band_names():
        i_data = product.get_band("i")
        q_data = product.get_band("q")
        if block_to_read is not None:
            i_data_area = i_data.read_as_array(
                block_to_read[3],
                block_to_read[2],
                xoffset=block_to_read[1],
                yoffset=block_to_read[0],
            )
            q_data_area = q_data.read_as_array(
                block_to_read[3],
                block_to_read[2],
                xoffset=block_to_read[1],
                yoffset=block_to_read[0],
            )
        else:
            i_data_area = i_data.read_as_array()
            q_data_area = q_data.read_as_array()
        target_area = q_data_area + 1j * i_data_area
    else:
        proc_data = product.get_band("proc_data")
        if block_to_read is not None:
            target_area = proc_data.read_as_array(
                block_to_read[3],
                block_to_read[2],
                xoffset=block_to_read[1],
                yoffset=block_to_read[0],
            )
        else:
            target_area = proc_data.read_as_array()

    product.close()
    # NOTE: flipping along range due to reversed range raster direction
    target_area = np.flip(target_area, axis=1)

    return target_area * scaling_conversion


def open_product(pf_path: str | Path) -> support.ASARProduct:
    """Open a ASAR product.

    Parameters
    ----------
    pf_path : str | Path
        Path to the ASAR product

    Returns
    -------
    ASARProduct
        ASARProduct object corresponding to the input ASAR product
    """

    if not support.is_asar_product(product=pf_path):
        raise support.InvalidASARProductError(f"{pf_path}")

    return support.ASARProduct(path=pf_path)
