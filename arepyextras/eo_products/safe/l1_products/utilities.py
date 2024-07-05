# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
SAFE reader support module
--------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import arepytools.io.metadata as meta
import numpy as np
import numpy.typing as npt
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.math.axis import Axis
from arepytools.math.genericpoly import SortedPolyList, create_sorted_poly_list
from arepytools.timing.precisedatetime import InvalidUtcString, PreciseDateTime
from lxml import etree
from numpy.polynomial.polynomial import Polynomial

from arepyextras.eo_products.common.utilities import (
    ConversionPolynomial,
    OrbitDirection,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
)

_IMAGE_FORMAT = ".tiff"
_METADATA_EXTENSION = ".xml"
_S1_TX_PULSE_LATCH_TIME = 1.439e-6  # [s]


class ChannelDataPairMismatch(RuntimeError):
    """Mismatch between channel data pair (raster data and corresponding metadata)"""


class InvalidChannelId(RuntimeError):
    """Invalid channel number"""


# camel case pattern
_cc_pattern = re.compile(r"(?<!^)(?=[A-Z])")


def convert_camel2snake(text: str) -> str:
    """CamelCase to snake_case converter.

    Parameters
    ----------
    text : str
        CamelCase text

    Returns
    -------
    str
        snake_case text
    """
    return _cc_pattern.sub("_", text).lower()


class InvalidSAFEProduct(RuntimeError):
    """Invalid SAFE product"""


class S1OrbitType(Enum):
    """EOrbitType class"""

    DOWNLINK = "DOWNLINK"
    PREDICTED = "PREDICTED"
    RESTITUTED = "RESTITUTED"
    PRECISE = "PRECISE"
    UNKNOWN = "UNKNOWN"


class S1DCEstimateMethod(Enum):
    """Doppler Centroid estimate method: data or geometry"""

    DATA = "DATA"
    GEOMETRY = "GEOMETRY"


class S1AcquisitionMode(Enum):
    """Sentinel-1 acquisition modes"""

    IW = "IW"  # Interferometric Wide swath (IW)
    EW = "EW"  # Extra-Wide swath (EW)
    WV = "WV"  # Wave (WV): single polarization only (HH or VV)
    SM = "SM"  # Stripmap (SM)
    EN = "EN"  # Elevation Notch (instrument calibration mode)
    AN = "AN"  # Azimuth Notch (instrument calibration mode)


class S1ResolutionClasses(Enum):
    """Sentinel-1 resolution classes"""

    FR = "FR"  # Full resolution
    HR = "HR"  # High resolution
    MR = "MR"  # Medium resolution


class S1Polarization(Enum):
    """Sentinel-1 signal polarization"""

    HH = "HH"
    VV = "VV"
    HV = "HV"
    VH = "VH"


class S1ReferenceFrameType(Enum):
    """Sentinel-1 available reference frames"""

    UNDEFINED = "Undefined"
    GALACTIC = "Galactic"
    BM1950 = "BM1950"
    BM2000 = "BM2000"
    HM2000 = "HM2000"
    GM2000 = "GM2000"
    MEAN_OF_DATE = "Mean Of Date"
    TRUE_OF_DATE = "True Of Date"
    PSEUDO_EARTH_FIXED = "Pseudo Earth Fixed"
    EARTH_FIXED = "Earth Fixed"
    TOPOCENTRIC = "Topocentric"
    SATELLITE_ORBITAL = "Satellite Orbital"
    SATELLITE_NORMAL = "Satellite Nominal"
    SATELLITE_ATTITUDE = "Satellite Attitude"
    INSTRUMENT_ATTITUDE = "Instrument Attitude"


class S1L1ProductType(Enum):
    """Sentinel-1 L1 product types"""

    SLC = "SLC"  # Slant Range, Single Look Complex (SLC, lvl 1)
    GRD = "GRD"  # Ground Range Multi Look Detected (GRD, lvl 1, phase lost)


def _convert_time_to_axis(axis_array: np.ndarray) -> Axis:
    """Converting time array to axis.

    Parameters
    ----------
    axis_array : np.ndarray
        time array to be converted

    Returns
    -------
    Axis
        Arepytools Axis representation of the input time array
    """

    axis_start = axis_array[0]
    relative_axis = (axis_array - axis_start).astype(float)

    return Axis(relative_axis, axis_start)


def data_scaling_factor_from_metadata_node(
    calibration_vector_node: etree._Element,
    radiometric_quantity: SARRadiometricQuantity = SARRadiometricQuantity.BETA_NOUGHT,
) -> float:
    """Extract data scaling factor from safe Calibration xml metadata file.

    Parameters
    ----------
    calibration_vector_node : etree._Element
        calibrationVectorList xml node
    radiometric_quantity : SARRadiometricQuantity, optional
        selected radiometric quantity, by default SARRadiometricQuantity.BETA_NOUGHT

    Returns
    -------
    float
        calibration scaling factor to be applied to raster data
    """
    # NOTE Beta Nought LUT is supposed to be equal to a constant value
    cal_vectors = calibration_vector_node.findall("calibrationVector")
    if radiometric_quantity == SARRadiometricQuantity.BETA_NOUGHT:
        return 1 / float(cal_vectors[0].find("betaNought").text.split(" ")[0])


def general_sar_orbit_from_s1_state_vectors(state_vectors: S1StateVectors) -> GeneralSarOrbit:
    """Creating a GeneralSarOrbit Arepytools element from S1StateVectors dataclass.

    Parameters
    ----------
    state_vectors : S1StateVectors
        state vectors S1StateVectors dataclass

    Returns
    -------
    GeneralSarOrbit
        channel orbit
    """
    return GeneralSarOrbit(time_axis=state_vectors.time_axis, state_vectors=state_vectors.positions.ravel())


def raster_info_from_metadata_node(
    image_information_node: etree._Element, samples_step: float | None = None
) -> meta.RasterInfo:
    """Creating a RasterInfo Arepytools metadata element from safe xml node.

    Parameters
    ----------
    image_information_node : etree._Element
        imageInformation metadata xml node
    samples_step : float | None, optional
        samples step in seconds, if no None it means that the product is SLC, otherwise it's GRD, by default None

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """
    image_information_dict = dict([(convert_camel2snake(p.tag), p.text) for p in image_information_node])
    raster_info = meta.RasterInfo(
        lines=int(image_information_dict["number_of_lines"]),
        samples=int(image_information_dict["number_of_samples"]),
        celltype="FLOAT_COMPLEX",
        filename=None,
        header_offset_bytes=0,
        row_prefix_bytes=0,
        byteorder="LITTLEENDIAN",
    )
    raster_info.set_lines_axis(
        lines_start=PreciseDateTime.from_utc_string(image_information_dict["product_first_line_utc_time"]),
        lines_start_unit="Mjd",
        lines_step=float(image_information_dict["azimuth_time_interval"]),
        lines_step_unit="s",
    )
    if samples_step is None:
        # GRD
        raster_info.set_samples_axis(
            samples_start=0,
            samples_start_unit="m",
            samples_step=float(image_information_dict["range_pixel_spacing"]),
            samples_step_unit="m",
        )
    else:
        # SLC
        raster_info.set_samples_axis(
            samples_start=float(image_information_dict["slant_range_time"]),
            samples_start_unit="s",
            samples_step=samples_step,
            samples_step_unit="s",
        )

    return raster_info


def dataset_info_from_metadata_nodes(
    header_node: etree._Element, product_info_node: etree._Element
) -> meta.DataSetInfo:
    """Creating a DataSetInfo Arepytools metadata element from safe xml nodes.

    Parameters
    ----------
    header_node : etree._Element
        adsHeader metadata xml node
    product_info_node : etree._Element
        productInformation metadata xml node

    Returns
    -------
    DataSetInfo
        DataSetInfo metadata object
    """
    product_info_dict = dict([(convert_camel2snake(p.tag), p.text) for p in product_info_node])
    header_dict = dict([(convert_camel2snake(p.tag), p.text) for p in header_node])
    dataset_info = meta.DataSetInfo(
        acquisition_mode_i=header_dict["mode"], fc_hz_i=float(product_info_dict["radar_frequency"])
    )
    dataset_info.sensor_name = header_dict["mission_id"]
    dataset_info.projection = product_info_dict["projection"].upper()
    dataset_info.side_looking = meta.ESideLooking("RIGHT")
    return dataset_info


def sampling_constants_from_metadata_nodes(
    swath_processing_node: etree._Element, product_info_node: etree._Element, image_info_node: etree._Element
) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata element from safe xml nodes.

    Parameters
    ----------
    swath_processing_node : etree._Element
        swathProcParams metadata xml node
    product_info_node : etree._Element
        productInformation metadata xml node
    image_info_node : etree._Element
        imageInformation metadata xml node

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    return SARSamplingFrequencies(
        azimuth_freq_hz=1 / float(image_info_node.find("azimuthTimeInterval").text),
        azimuth_bandwidth_freq_hz=float(swath_processing_node.find("azimuthProcessing/processingBandwidth").text),
        range_freq_hz=float(product_info_node.find("rangeSamplingRate").text),
        range_bandwidth_freq_hz=float(swath_processing_node.find("rangeProcessing/processingBandwidth").text),
    )


def acquisition_timeline_from_metadata_nodes(downlink_info_node: etree._Element) -> meta.AcquisitionTimeLine:
    """Creating a AcquisitionTimeLine Arepytools metadata element from safe xml nodes.

    Parameters
    ----------
    downlink_info_node : etree._Element
        downlinkInformation metadata xml node

    Returns
    -------
    AcquisitionTimeLine
        AcquisitionTimeLine metadata object
    """
    swst_list = downlink_info_node.find("downlinkValues/swstList")
    swl_list = downlink_info_node.find("downlinkValues/swlList")
    swst_num = int(swst_list.get("count"))
    swl_num = int(swl_list.get("count"))

    # missing elements set with these dummy values
    missing_lines_number = 0
    missing_lines_azimuth_times = None
    noise_packets_number = 0
    noise_packets_azimuth_times = None
    internal_calibration_number = 0
    internal_calibration_azimuth_times = None

    if swst_num > 1:
        swst_changes_number = swst_num
        swst_changes_azimuth_times = [
            PreciseDateTime.from_utc_string(s.text) for s in swst_list.findall("swst/azimuthTime")
        ]
        swst_changes_values = [float(s.text) for s in swst_list.findall("swst/value")]
    else:
        swst_changes_number = 1
        swst_changes_azimuth_times = [PreciseDateTime.from_utc_string(swst_list.find("swst/azimuthTime").text)]
        swst_changes_values = [float(swst_list.find("swst/value").text)]

    if swl_num > 1:
        swl_changes_number = swl_num
        swl_changes_azimuth_times = [
            PreciseDateTime.from_utc_string(s.text) for s in swl_list.findall("swl/azimuthTime")
        ]
        swl_changes_values = [float(s.text) for s in swl_list.findall("swl/value")]
    else:
        swl_changes_number = 1
        swl_changes_azimuth_times = [PreciseDateTime.from_utc_string(swl_list.find("swl/azimuthTime").text)]
        swl_changes_values = [float(swl_list.find("swl/value").text)]

    return meta.AcquisitionTimeLine(
        missing_lines_number_i=missing_lines_number,
        missing_lines_azimuth_times_i=missing_lines_azimuth_times,
        swst_changes_number_i=swst_changes_number,
        swst_changes_azimuth_times_i=swst_changes_azimuth_times,
        swst_changes_values_i=swst_changes_values,
        noise_packets_number_i=noise_packets_number,
        noise_packets_azimuth_times_i=noise_packets_azimuth_times,
        internal_calibration_number_i=internal_calibration_number,
        internal_calibration_azimuth_times_i=internal_calibration_azimuth_times,
        swl_changes_number_i=swl_changes_number,
        swl_changes_azimuth_times_i=swl_changes_azimuth_times,
        swl_changes_values_i=swl_changes_values,
    )


def doppler_centroid_vector_from_metadata_nodes(
    dc_estimate_node: etree._Element, estimate_method: S1DCEstimateMethod
) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler centroid polynomial wrapper from safe xml node.

    Parameters
    ----------
    dc_estimate_node : etree._Element
        dcEstimateList metadata xml node
    estimate_method : S1DCEstimateMethod
        doppler centroid estimate method

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerCentroidVector metadata object
    """
    doppler_poly = []
    for item in dc_estimate_node:
        if estimate_method == S1DCEstimateMethod.DATA:
            coefficients = [float(c) for c in item.find("dataDcPolynomial").text.split(" ")]
        else:
            coefficients = [float(c) for c in item.find("geometryDcPolynomial").text.split(" ")]

        coefficients = [coefficients[0], coefficients[1], 0, 0, coefficients[2], 0, 0, 0, 0, 0, 0]
        doppler_poly.append(
            meta.DopplerCentroid(
                i_ref_az=PreciseDateTime.from_utc_string(item.find("azimuthTime").text),
                i_ref_rg=float(item.find("t0").text),
                i_coefficients=coefficients,
            )
        )

    return create_sorted_poly_list(meta.DopplerCentroidVector(i_poly2d=doppler_poly))


def doppler_rate_vector_from_metadata_nodes(azimuth_fm_rate_node: etree._Element) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler rate vector polynomial wrapper from safe xml node.

    Parameters
    ----------
    azimuth_fm_rate_node : etree._Element
        azimuthFmRateList metadata xml node

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerRateVector metadata object
    """
    doppler_rate_poly = []
    for item in azimuth_fm_rate_node:
        coefficients = [float(c) for c in item.find("azimuthFmRatePolynomial").text.split(" ")]
        coefficients = [coefficients[0], coefficients[1], 0, 0, coefficients[2], 0, 0, 0, 0, 0, 0]
        doppler_rate_poly.append(
            meta.DopplerRate(
                i_ref_az=PreciseDateTime.from_utc_string(item.find("azimuthTime").text),
                i_ref_rg=float(item.find("t0").text),
                i_coefficients=coefficients,
            )
        )
    return create_sorted_poly_list(meta.DopplerRateVector(i_poly2d=doppler_rate_poly))


def noise_from_metadata_nodes(noise_list_node: etree._Element) -> dict[str, S1Noise]:
    """Creating a dictionary of S1Noise dataclass for each swath from safe xml node.

    Parameters
    ----------
    noise_list_node : etree._Element
        noiseList metadata xml node

    Returns
    -------
    dict[str, S1Noise]
        keys are the swaths, values are S1Noise dataclasses
    """

    dict_keys = {k.find("swath").text for k in noise_list_node}
    noise = {key: S1Noise.from_metadata_node(noise_list_node=noise_list_node, swath=key) for key in sorted(dict_keys)}

    return noise


def antenna_pattern_from_metadata_nodes(antenna_pattern_list_node: etree._Element) -> dict[str, S1AntennaPattern]:
    """Creating a dictionary of S1AntennaPattern dataclass for each swath from safe xml node.

    Parameters
    ----------
    antenna_pattern_list_node : etree._Element
        antennaPatternList metadata xml node

    Returns
    -------
    dict[str, S1AntennaPattern]
        keys are the swaths, values are S1AntennaPattern dataclasses
    """

    dict_keys = {k.find("swath").text for k in antenna_pattern_list_node}
    antenna = {
        key: S1AntennaPattern.from_metadata_node(antenna_pattern_list_node=antenna_pattern_list_node, swath=key)
        for key in sorted(dict_keys)
    }

    return antenna


@dataclass
class S1ChirpReplica:
    """Sentinel-1 PG chirp replica parameters derived from the calibration pulses at 100 MHz bandwidth dataclass"""

    swath: str  # swath of to the current chirp replica
    time_axis: np.ndarray  # Zero Doppler azimuth time at which replica applies
    cross_correlation_bandwidth: (
        np.ndarray
    )  # 3-dB pulse width of cross-correlation function between the reconstructed replica and the nominal replica
    cross_correlation_pslr: (
        np.ndarray
    )  # pslr of cross-correlation function between the reconstructed replica and the nominal replica
    cross_correlation_islr: (
        np.ndarray
    )  # islr of cross-correlation function between the reconstructed replica and the nominal replica
    cross_correlation_peak_location: (
        np.ndarray
    )  # peak location of cross-correlation function between the reconstructed replica and the nominal replica [samples]
    pg_product_amplitude: np.ndarray  # amplitude of the PG product derived from this replica
    pg_product_phase: np.ndarray  # phase of the PG product derived from this replica [radians]
    model_pg_product_amplitude: np.ndarray  # PG product amplitude value from the input PG product model
    model_pg_product_phase: np.ndarray  # PG product phase value from the input PG product model [radians]
    internal_time_delay: (
        np.ndarray
    )  # calculated deviation of the location of this PG replica from the location of the transmitted pulse
    reconstructed_replica_validity: list[
        bool
    ]  # if the cross-correlation between the nominal PG replica and this extracted PG replica resulted in a valid peak location
    relative_validity: list[bool]  # if the amplitude and phase of the PG product passed relative validation
    absolute_validity: list[bool]  # if the amplitude and phase of the PG product passed the absolute validation

    @staticmethod
    def from_metadata_node(replica_info_node: etree._Element) -> S1ChirpReplica:
        """Generating S1ChirpReplica object directly from metadata xml nodes.

        Parameters
        ----------
        replica_info_node : etree._Element
            replicaInformation metadata xml node

        Returns
        -------
        S1ChirpReplica
            chirp replica dataclass
        """

        # for item in replica_info_node:
        swath = replica_info_node.find("swath").text
        replica_list = replica_info_node.find("replicaList")
        time_axis = np.array([PreciseDateTime.from_utc_string(rep.find("azimuthTime").text) for rep in replica_list])
        cross_correlation_bandwidth = np.array(
            [float(rep.find("crossCorrelationBandwidth").text) for rep in replica_list]
        )
        cross_correlation_pslr = np.array([float(rep.find("crossCorrelationPslr").text) for rep in replica_list])
        cross_correlation_islr = np.array([float(rep.find("crossCorrelationIslr").text) for rep in replica_list])
        cross_correlation_peak_location = np.array(
            [float(rep.find("crossCorrelationPeakLocation").text) for rep in replica_list]
        )
        pg_product_amplitude = np.array([float(rep.find("pgProductAmplitude").text) for rep in replica_list])
        pg_product_phase = np.array([float(rep.find("pgProductPhase").text) for rep in replica_list])
        model_pg_product_amplitude = np.array([float(rep.find("modelPgProductAmplitude").text) for rep in replica_list])
        model_pg_product_phase = np.array([float(rep.find("modelPgProductPhase").text) for rep in replica_list])
        internal_time_delay = np.array([float(rep.find("internalTimeDelay").text) for rep in replica_list])
        reconstructed_replica_validity = [bool(rep.find("reconstructedReplicaValidFlag").text) for rep in replica_list]
        relative_validity = [bool(rep.find("relativePgProductValidFlag").text) for rep in replica_list]
        absolute_validity = [bool(rep.find("absolutePgProductValidFlag").text) for rep in replica_list]

        return S1ChirpReplica(
            swath=swath,
            time_axis=time_axis,
            cross_correlation_bandwidth=cross_correlation_bandwidth,
            cross_correlation_pslr=cross_correlation_pslr,
            cross_correlation_islr=cross_correlation_islr,
            cross_correlation_peak_location=cross_correlation_peak_location,
            pg_product_amplitude=pg_product_amplitude,
            pg_product_phase=pg_product_phase,
            model_pg_product_amplitude=model_pg_product_amplitude,
            model_pg_product_phase=model_pg_product_phase,
            internal_time_delay=internal_time_delay,
            reconstructed_replica_validity=reconstructed_replica_validity,
            relative_validity=relative_validity,
            absolute_validity=absolute_validity,
        )


@dataclass
class S1Noise:
    """Sentinel-1 thermal noise parameters derived from noise packets dataclass"""

    swath: str  # swath of to the current noise data
    time_axis: np.ndarray  # Zero Doppler azimuth time of the noise measurement
    noise_power_correction_factor: np.ndarray  # noise power correction factor
    noise_lines_num: np.ndarray  # number of noise lines used to calculate noise correction factor

    @staticmethod
    def from_metadata_node(noise_list_node: etree._Element, swath: str) -> S1Noise:
        """Generating S1Noise object directly from metadata xml nodes.

        Parameters
        ----------
        noise_list_node : etree._Element
            noiseList metadata xml node
        swath : str
            swath of interest

        Returns
        -------
        S1Noise
            noise correction factor dataclass
        """

        # taking only the nodes corresponding to the selected swath
        filtered_nodes = filter(lambda x: x.find("swath").text == swath, noise_list_node)

        time_axis = []
        noise_power_correction_factor = []
        noise_lines_num = []
        for item in filtered_nodes:
            time_axis.append(PreciseDateTime.from_utc_string(item.find("azimuthTime").text))
            noise_power_correction_factor.append(float(item.find("noisePowerCorrectionFactor").text))
            noise_lines_num.append(int(item.find("numberOfNoiseLines").text))

        return S1Noise(
            swath=swath,
            time_axis=np.array(time_axis),
            noise_power_correction_factor=np.array(noise_power_correction_factor),
            noise_lines_num=np.array(noise_lines_num),
        )


@dataclass
class S1AntennaPattern:
    """Sentinel-1 antenna pattern dataclass"""

    swath: str  # swath of to the current antenna pattern data
    time_axis: np.ndarray  # Zero Doppler azimuth time at which antenna pattern applies
    slant_range_time_array: np.ndarray  # two-way slant range time array for this antenna pattern [s]
    elevation_angle_array: np.ndarray  # corresponding elevation angle for this antenna pattern [degrees]
    elevation_pattern_array: np.ndarray  # corresponding two-way antenna elevation pattern value for this point
    incidence_angle: np.ndarray  # corresponding incidence angle value for this point [degrees]
    terrain_height: np.ndarray  # average terrain height in range for this antenna pattern [m]
    roll: np.ndarray  # estimated roll angle for this antenna pattern [degrees]

    @staticmethod
    def from_metadata_node(antenna_pattern_list_node: etree._Element, swath: str) -> S1AntennaPattern:
        """Generating S1AntennaPattern object directly from metadata xml nodes.

        Parameters
        ----------
        antenna_pattern_list_node : etree._Element
            antennaPatternList metadata xml node
        swath : str
            swath of interest

        Returns
        -------
        S1Noise
            antenna pattern dataclass
        """

        # taking only the nodes corresponding to the selected swath
        filtered_nodes = filter(lambda x: x.find("swath").text == swath, antenna_pattern_list_node)

        time_axis = []
        slant_range_time_array = []
        elevation_angle_array = []
        elevation_pattern_array = []
        incidence_angle = []
        terrain_height = []
        roll = []
        for item in filtered_nodes:
            time_axis.append(PreciseDateTime.from_utc_string(item.find("azimuthTime").text))
            slant_range_time_array.append(np.array([float(c) for c in item.find("slantRangeTime").text.split(" ")]))
            elevation_angle_array.append(np.array([float(c) for c in item.find("elevationAngle").text.split(" ")]))
            elevation_pattern_array.append(np.array([float(c) for c in item.find("elevationPattern").text.split(" ")]))
            incidence_angle.append(np.array([float(c) for c in item.find("incidenceAngle").text.split(" ")]))
            terrain_height.append(float(item.find("terrainHeight").text))
            roll.append(float(item.find("roll").text))

        return S1AntennaPattern(
            swath=swath,
            time_axis=np.array(time_axis),
            slant_range_time_array=np.stack(slant_range_time_array),
            elevation_angle_array=np.stack(elevation_angle_array),
            elevation_pattern_array=np.stack(elevation_pattern_array),
            incidence_angle=np.stack(incidence_angle),
            terrain_height=np.array(terrain_height),
            roll=np.array(roll),
        )


@dataclass
class S1GeneralChannelInfo:
    """Sentinel-1 general channel info representation dataclass"""

    mission_id: str
    channel_id: int
    swath: str
    product_type: S1L1ProductType
    polarization: SARPolarization
    projection: SARProjection
    mode: S1AcquisitionMode
    orbit_direction: OrbitDirection
    range_sampling_rate: float
    signal_frequency: float
    start_time: PreciseDateTime
    stop_time: PreciseDateTime

    @staticmethod
    def from_metadata_node(header_node: etree._Element, product_info_node: etree._Element) -> S1GeneralChannelInfo:
        """Generating S1GeneralChannelInfo object directly from metadata xml nodes.

        Parameters
        ----------
        header_node : etree._Element
            adsHeader metadata xml node
        product_info_node : etree._Element
            generalAnnotation/productInformation metadata xml node

        Returns
        -------
        S1GeneralChannelInfo
            general channel info dataclass
        """

        header_dict = dict([(convert_camel2snake(h.tag), h.text) for h in header_node])
        product_info_dict = dict([(convert_camel2snake(p.tag), p.text) for p in product_info_node])

        polarization = header_dict["polarisation"]
        # converting the polarization string value to a proper input format for SARPolarization enum
        if len(polarization) == 2:
            polarization = polarization[0] + "/" + polarization[1]

        general_info = S1GeneralChannelInfo(
            mission_id=header_dict["mission_id"],
            channel_id=int(header_dict["image_number"]),
            swath=header_dict["swath"],
            product_type=S1L1ProductType(header_dict["product_type"]),
            polarization=SARPolarization(polarization),
            projection=SARProjection(product_info_dict["projection"].upper()),
            mode=S1AcquisitionMode(header_dict["mode"]),
            orbit_direction=OrbitDirection(product_info_dict["pass"].lower()),
            range_sampling_rate=float(product_info_dict["range_sampling_rate"]),
            signal_frequency=float(product_info_dict["radar_frequency"]),
            start_time=PreciseDateTime.from_utc_string(header_dict["start_time"]),
            stop_time=PreciseDateTime.from_utc_string(header_dict["stop_time"]),
        )

        return general_info


@dataclass
class S1StateVectors:
    """Sentinel-1 orbit's state vectors"""

    num: int  # attitude data numerosity
    frame: S1ReferenceFrameType  # reference frame of the orbit state data
    positions: np.ndarray  # platform position data with respect to the Earth-fixed reference frame
    velocities: np.ndarray  # platform velocity data with respect to the Earth-fixed reference frame
    time_axis: np.ndarray  # PreciseDateTime axis at which orbit state vectors apply
    time_step: float  # time axis step
    orbit_direction: OrbitDirection | None = None  # orbit direction: ascending or descending
    orbit_type: S1OrbitType | None = None  # orbit type

    @staticmethod
    def _get_xyz_from_orbit_node(node: etree._Element, tag: str) -> np.ndarray:
        """Extracting an array of shape (N, 3) with x, y, z components for the desired tag inside orbit node.
        Tag can be "position" or "velocity".

        Parameters
        ----------
        node : etree._Element
            orbitList xml node
        tag : str
            tag of the node where to extract x, y, z components

        Returns
        -------
        np.ndarray
            array of components along x, y, z, in the form (N, 3)
        """
        x_component = np.array([float(p.text) for p in node.findall("./orbit/" + tag + "/x")])
        y_component = np.array([float(p.text) for p in node.findall("./orbit/" + tag + "/y")])
        z_component = np.array([float(p.text) for p in node.findall("./orbit/" + tag + "/z")])
        return np.stack([x_component, y_component, z_component], axis=1)

    @staticmethod
    def from_metadata_node(orbit_node: etree._Element, orbit_type: S1OrbitType | None = None) -> S1StateVectors:
        """Generating a S1StateVectors object directly from metadata xml node.

        Parameters
        ----------
        orbit_node : etree._Element
            orbitList xml node
        orbit_type : S1OrbitType, optional
            S1OrbitType orbit type, by default None

        Returns
        -------
        S1StateVectors
            orbit's state vectors dataclass
        """

        numerosity = int(orbit_node.values()[0])
        positions = S1StateVectors._get_xyz_from_orbit_node(node=orbit_node, tag="position")
        velocities = S1StateVectors._get_xyz_from_orbit_node(node=orbit_node, tag="velocity")
        frames = [frame.text for frame in orbit_node.findall("./orbit/frame")]
        times = np.array([PreciseDateTime.from_utc_string(time.text) for time in orbit_node.findall("./orbit/time")])

        assert len(set(frames)) == 1
        assert positions.shape[0] == velocities.shape[0] == times.size == len(frames) == numerosity

        state_vectors = S1StateVectors(
            num=numerosity,
            frame=S1ReferenceFrameType(set(frames).pop()),
            positions=positions,
            velocities=velocities,
            time_axis=times,
            time_step=times[1] - times[0],
            orbit_type=orbit_type if orbit_type is not None else None,
        )
        return state_vectors


@dataclass
class S1Attitude:
    """Sentinel-1 sensor's attitude"""

    num: int  # attitude data numerosity
    frame: S1ReferenceFrameType  # reference frame of the attitude data.
    quaternions: np.ndarray  # attitude quaternion as extracted from ancillary attitude data (N, 4)
    angular_velocities: np.ndarray  # angular velocity as extracted from ancillary attitude data [degrees/s] (N, 3)
    roll: np.ndarray  # platform roll
    pitch: np.ndarray  # platform pitch
    yaw: np.ndarray  # platform yaw
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies

    @staticmethod
    def _get_quaternions_from_node(node: etree._Element) -> np.ndarray:
        """Extracting quaternions array shape (N, 4) with q0, q1, q2 and q3 components from xml attitude node.

        Parameters
        ----------
        node : etree._Element
            attitudeList xml node

        Returns
        -------
        np.ndarray
            quaternions array, (N, 4)
        """
        q_0 = np.array([float(q.text) for q in node.findall("./attitude/q0")])
        q_1 = np.array([float(q.text) for q in node.findall("./attitude/q1")])
        q_2 = np.array([float(q.text) for q in node.findall("./attitude/q2")])
        q_3 = np.array([float(q.text) for q in node.findall("./attitude/q3")])
        return np.stack([q_0, q_1, q_2, q_3], axis=1)

    @staticmethod
    def _get_angular_velocities_from_node(node: etree._Element) -> np.ndarray:
        """Extracting angular velocities array shape (N, 3) with wx, wy and wz components from xml attitude node.

        Parameters
        ----------
        node : etree._Element
            attitudeList xml node

        Returns
        -------
        np.ndarray
            angular velocities array, (N, 3)
        """
        w_x = np.array([float(q.text) for q in node.findall("./attitude/wx")])
        w_y = np.array([float(q.text) for q in node.findall("./attitude/wy")])
        w_z = np.array([float(q.text) for q in node.findall("./attitude/wz")])
        return np.stack([w_x, w_y, w_z], axis=1)

    @staticmethod
    def from_metadata_node(attitude_node: etree._Element) -> S1Attitude:
        """Generating S1Attitude object directly from metadata xml node.

        Parameters
        ----------
        attitude_node : etree._Element
            attitudeList xml node

        Returns
        -------
        S1Attitude
            sensor's attitude dataclass
        """

        numerosity = int(attitude_node.values()[0])
        quaternions = S1Attitude._get_quaternions_from_node(node=attitude_node)
        angular_velocities = S1Attitude._get_angular_velocities_from_node(node=attitude_node)
        yaw = np.array([float(y.text) for y in attitude_node.findall("./attitude/yaw")])
        pitch = np.array([float(p.text) for p in attitude_node.findall("./attitude/pitch")])
        roll = np.array([float(r.text) for r in attitude_node.findall("./attitude/roll")])
        frames = [frame.text for frame in attitude_node.findall("./attitude/frame")]
        times = np.array(
            [PreciseDateTime.from_utc_string(time.text) for time in attitude_node.findall("./attitude/time")]
        )
        time_axis = _convert_time_to_axis(times)

        assert len(set(frames)) == 1
        assert (
            quaternions.shape[0]
            == angular_velocities.shape[0]
            == yaw.size
            == pitch.size
            == roll.size
            == times.size
            == len(frames)
            == numerosity
        )

        attitude = S1Attitude(
            num=numerosity,
            quaternions=quaternions,
            angular_velocities=angular_velocities,
            frame=S1ReferenceFrameType(set(frames).pop()),
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            time_axis=time_axis,
        )

        return attitude


@dataclass
class S1BurstInfo:
    """Sentinel-1 swath burst info"""

    num: int  # number of bursts in this swath
    lines_per_burst: int  # number of azimuth lines within each burst
    samples_per_burst: int  # number of range samples within each burst
    azimuth_start_times: np.ndarray  # zero doppler azimuth time of the first line of this burst
    range_start_times: np.ndarray  # zero doppler range time of the first sample of this burst
    azimuth_start_times_anx: (
        np.ndarray
    )  # zero doppler azimuth time of the first line of this burst relative to the Ascending Node Crossing (ANX) time

    @staticmethod
    def from_metadata_node(burst_node: etree._Element, samples_start: float) -> S1BurstInfo:
        """Generating S1BurstInfo object directly from metadata xml node.

        Parameters
        ----------
        burst_node : etree._Element
            swathTiming xml node

        Returns
        -------
        S1BurstInfo
            swath's burst info dataclass
        """
        lines_per_burst = int(burst_node.find("linesPerBurst").text)
        samples_per_burst = int(burst_node.find("samplesPerBurst").text)
        bursts = burst_node.find("burstList")
        numerosity = int(bursts.values()[0])
        azimuth_start_times = np.array(
            [PreciseDateTime.from_utc_string(t.text) for t in bursts.findall("./burst/azimuthTime")]
        )
        azimuth_start_times_anx = np.array([float(t.text) for t in bursts.findall("./burst/azimuthAnxTime")])

        assert numerosity == azimuth_start_times.size == azimuth_start_times_anx.size

        burst_info = S1BurstInfo(
            num=numerosity,
            lines_per_burst=lines_per_burst,
            samples_per_burst=samples_per_burst,
            azimuth_start_times=azimuth_start_times,
            range_start_times=np.repeat(samples_start, numerosity),
            azimuth_start_times_anx=azimuth_start_times_anx,
        )
        return burst_info


@dataclass
class S1SwathInfo:
    """Sentinel-1 swath info"""

    rank: int
    azimuth_steering_rate_poly: tuple[float, float, float]
    prf: float

    @staticmethod
    def from_metadata_nodes(product_info: etree._Element, downlink_info_node: etree._Element) -> S1SwathInfo:
        """Generating S1SwathInfo object directly from metadata xml nodes.

        Parameters
        ----------
        product_info : etree._Element
            productInformation xml node
        downlink_info_node : etree._Element
            downlinkInformation xml node

        Returns
        -------
        S1SwathInfo
            swath info dataclass
        """
        return S1SwathInfo(
            rank=int(downlink_info_node.find("downlinkValues/rank").text),
            prf=float(downlink_info_node.find("prf").text),
            azimuth_steering_rate_poly=(float(product_info.find("azimuthSteeringRate").text) * np.pi / 180, 0, 0),
        )


@dataclass
class S1Pulse:
    """Sentinel-1 pulse info"""

    length: float
    bandwidth: float
    energy: float
    start_frequency: float
    start_phase: float
    direction: meta.EPulseDirection
    tx_pulse_latch_time: float = _S1_TX_PULSE_LATCH_TIME

    @staticmethod
    def from_metadata_nodes(swath_processing_node: etree._Element, downlink_info_node: etree._Element) -> S1Pulse:
        """Generating S1Pulse object directly from metadata xml nodes.

        Parameters
        ----------
        swath_processing_node : etree._Element
            swathProcParams xml node
        downlink_info_node : etree._Element
            downlinkInformation xml node

        Returns
        -------
        S1Pulse
            pulse info dataclass
        """
        return S1Pulse(
            length=float(downlink_info_node.find("downlinkValues/txPulseLength").text),
            bandwidth=float(swath_processing_node.find("rangeProcessing/totalBandwidth").text),
            energy=1.0,
            start_frequency=float(downlink_info_node.find("downlinkValues/txPulseStartFrequency").text),
            start_phase=0.0,
            direction=meta.EPulseDirection.up,
        )


@dataclass
class S1CoordinateConversions:
    """Sentinel-1 coordinate conversion"""

    ground_to_slant: list[ConversionPolynomial] | None = None
    slant_to_ground: list[ConversionPolynomial] | None = None
    azimuth_reference_times: np.ndarray | None = None

    def _detect_right_polynomial_index(self, azimuth_time: PreciseDateTime) -> int:
        """Detecting the index of the right polynomial to be used given an input azimuth time.
        The polynomial to be used is the one with reference azimuth time closest to the input value but with
        reference_azimuth_time < input_azimuth_time.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            selected azimuth time

        Returns
        -------
        int
            index corresponding to the polynomial to be used
        """
        diff = np.array(azimuth_time - self.azimuth_reference_times).astype("float")
        return np.ma.masked_where(diff < 0, diff).argmin()

    @staticmethod
    def from_metadata_node(coord_conversion_node: etree._Element) -> S1CoordinateConversions:
        """Generating S1CoordinateConversions object directly from metadata xml node.

        Parameters
        ----------
        coord_conversion_node : etree._Element
            coordinateConversion xml node

        Returns
        -------
        S1CoordinateConversions
            polynomial for coordinate conversion dataclass
        """
        if coord_conversion_node.get("count") == "0":
            return S1CoordinateConversions()

        # polynomials written in metadata are supposed to be always already sorted in time, i.e. azimuth validity
        # reference time will be strictly increasing (no need to be sorted afterwards)
        num_poly = int(coord_conversion_node.get("count"))
        slant_to_ground_coeff = []
        ground_to_slant_coeff = []
        az_time_axis = []
        rng_time_axis = []
        slant_range_origin = []
        ground_range_origin = []
        for item in coord_conversion_node:
            az_time_axis.append(PreciseDateTime.from_utc_string(item.find("azimuthTime").text))
            rng_time_axis.append(float(item.find("slantRangeTime").text))
            slant_range_origin.append(float(item.find("sr0").text))
            ground_range_origin.append(float(item.find("gr0").text))
            slant_to_ground_coeff.append([float(c) for c in item.find("srgrCoefficients").text.split(" ")])
            ground_to_slant_coeff.append([float(c) for c in item.find("grsrCoefficients").text.split(" ")])

        # creating polynomial dataclasses
        ground_to_slant_poly = [
            ConversionPolynomial(
                azimuth_reference_time=az_time_axis[p],
                origin=ground_range_origin[p],
                polynomial=Polynomial(ground_to_slant_coeff[p]),
            )
            for p in range(num_poly)
        ]
        slant_to_ground_poly = [
            ConversionPolynomial(
                azimuth_reference_time=az_time_axis[p],
                origin=slant_range_origin[p],
                polynomial=Polynomial(slant_to_ground_coeff[p]),
            )
            for p in range(num_poly)
        ]

        return S1CoordinateConversions(
            azimuth_reference_times=np.array(az_time_axis),
            ground_to_slant=ground_to_slant_poly,
            slant_to_ground=slant_to_ground_poly,
        )

    def evaluate_ground_to_slant(
        self, azimuth_time: PreciseDateTime, ground_range: Union[float, npt.ArrayLike]
    ) -> float:
        """Compute ground to slant conversion.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time to select the proper polynomial to be used for conversion
        ground_range :  Union[float, npt.ArrayLike]
            ground range value(s) in meters

        Returns
        -------
        float
            slant range value
        """
        poly_index = self._detect_right_polynomial_index(azimuth_time=azimuth_time)
        poly = self.ground_to_slant[poly_index]
        return poly.polynomial(ground_range - poly.origin)

    def evaluate_slant_to_ground(
        self, azimuth_time: PreciseDateTime, slant_range: Union[float, npt.ArrayLike]
    ) -> float:
        """Compute slant to ground conversion.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time to select the proper polynomial to be used for conversion
        slant_range :  Union[float, npt.ArrayLike]
            slant range value(s) in meters

        Returns
        -------
        float
            ground range value
        """
        poly_index = self._detect_right_polynomial_index(azimuth_time=azimuth_time)
        poly = self.slant_to_ground[poly_index]
        return poly.polynomial(slant_range - poly.origin)


@dataclass
class S1ChannelMetadata:
    """Sentinel-1 channel metadata xml file wrapper"""

    general_info: S1GeneralChannelInfo
    general_sar_orbit: GeneralSarOrbit
    attitude: S1Attitude
    burst_info: S1BurstInfo
    raster_info: meta.RasterInfo
    dataset_info: meta.DataSetInfo
    swath_info: S1SwathInfo
    sampling_constants: SARSamplingFrequencies
    acquisition_timeline: meta.AcquisitionTimeLine
    doppler_centroid_poly: SortedPolyList
    doppler_rate_vector: SortedPolyList
    pulse: S1Pulse
    coordinate_conversions: S1CoordinateConversions
    state_vectors: S1StateVectors
    chirp_replica: dict[str, S1ChirpReplica]  # dictionary key is the swath
    noise: dict[str, S1Noise]  # dictionary key is the swath
    antenna_pattern: dict[str, S1AntennaPattern]  # dictionary key is the swath


class S1Manifest:
    """Sentinel-1 SAFE manifest parser"""

    def __init__(self, manifest_path: Path) -> None:
        """Parsing the SAFE product manifest to gather information about data and metadata files.

        Parameters
        ----------
        manifest_path : Path
            path to the manifest file
        """
        self._path = manifest_path
        self._channel_list = []
        self._data_paths = []
        self._metadata_paths = []
        self._root = self._read_file()

    @staticmethod
    def _extract_relative_path_from_nodes(nodes: list[etree._Element]) -> list:
        """Extracting relative paths from location element for each input dataObject node.

        Parameters
        ----------
        nodes : list[etree._Element]
            list of dataObject nodes

        Returns
        -------
        list
            list of paths for each input node
        """
        file_locations = [n.find("byteStream").find("fileLocation") for n in nodes]
        relative_file_paths = [dict(loc.items())["href"] for loc in file_locations]
        return relative_file_paths

    def _read_file(self) -> etree._Element:
        """Parsing manifest .xml file.

        Returns
        -------
        etree._Element
            XML manifest root
        """
        tree = etree.parse(self._path)
        root = tree.getroot()
        return root

    def parse_manifest_document(
        self,
    ) -> tuple[list[str], list[str], list[str], PreciseDateTime, tuple[float, float, float, float]]:
        """Parsing SAFE manifest .xml document to gather information to available data, metadata, calibrations and
        product acquisition time.

        Returns
        -------
        tuple[list[str], list[str], list[str], PreciseDateTime, tuple[float, float, float, float]]
            list of data relative paths with respect to SAFE product folder,
            list of metadata relative paths with respect to SAFE product folder,
            list of calibration relative paths with respect to SAFE product folder,
            acquisition start time,
            product footprint [min lat, max lat, min lon, max lon]
        """
        data_object_section = self._root.find("dataObjectSection")
        data_objects = data_object_section.findall("dataObject")

        # extracting paths to each raster data inside the SAFE product
        data_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1MeasurementSchema"]
        relative_data_paths = self._extract_relative_path_from_nodes(data_elements)

        # extracting paths to each metadata inside the SAFE product
        metadata_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1ProductSchema"]
        relative_metadata_paths = self._extract_relative_path_from_nodes(metadata_elements)

        # extracting paths to each calibration file inside the SAFE product
        calibration_elements = [d for d in data_objects if dict(d.items())["repID"] == "s1Level1CalibrationSchema"]
        relative_calibration_paths = self._extract_relative_path_from_nodes(calibration_elements)

        # extracting acquisition start time
        metadata_objects = self._root.find("metadataSection").findall("metadataObject")
        acquisition_period = [m for m in metadata_objects if dict(m.items())["ID"] == "acquisitionPeriod"]
        period = [m for m in acquisition_period[0].find("metadataWrap/xmlData") if "acquisitionPeriod" in m.tag][0]
        date = [p.text for p in period if "startTime" in p.tag][0]
        try:
            acq_start_time = PreciseDateTime.from_utc_string(date)
        except InvalidUtcString:
            acq_start_time = PreciseDateTime.fromisoformat(date)

        # extracting product footprint
        meas_frame_set_node = [m for m in metadata_objects if dict(m.items())["ID"] == "measurementFrameSet"][0]
        footprint_str = (
            meas_frame_set_node.xpath(
                ".//metadataWrap//xmlData//safe:frameSet//safe:frame//safe:footPrint//gml:coordinates",
                namespaces=self._root.nsmap,
            )[0]
            .text.replace(",", " ")
            .split()
        )
        longitudes = [float(f) for f in footprint_str[1::2]]
        latitudes = [float(f) for f in footprint_str[::2]]
        footprint = (min(latitudes), max(latitudes), min(longitudes), max(longitudes))

        return relative_data_paths, relative_metadata_paths, relative_calibration_paths, acq_start_time, footprint


class SAFEFolderLayout:
    """SAFE file main directory architecture"""

    def __init__(self, path: Path) -> None:
        """Definition of internal architecture of a SAFE product folder.

        Parameters
        ----------
        path : Path
            path to the SAFE base folder
        """
        self._safe_path = path
        self._manifest_file = path.joinpath("manifest.safe")
        self._annotations_dir = path.joinpath("annotation")
        self._measurements_dir = path.joinpath("measurement")
        self._preview_dir = path.joinpath("preview")
        self._calibrations_dir = self._annotations_dir.joinpath("calibration")

    @property
    def manifest(self) -> Path:
        """Location of manifest file"""
        return self._manifest_file

    @property
    def annotations(self) -> Path:
        """Location of annotation folder"""
        return self._annotations_dir

    @property
    def measurements(self) -> Path:
        """Location of measurement folder"""
        return self._measurements_dir

    @property
    def preview(self) -> Path:
        """Location of preview folder"""
        return self._preview_dir

    @property
    def calibration(self) -> Path:
        """Location of calibration folder in annotations"""
        return self._calibrations_dir


class S1Product:
    """Sentinel-1 product object"""

    def __init__(self, path: Union[str, Path]) -> None:
        self._product_path = Path(path)
        self._product_name = self._product_path.name.strip(".SAFE")
        self._layout = SAFEFolderLayout(self._product_path)
        self._manifest = S1Manifest(manifest_path=self._layout.manifest)

        # locating relative data and metadata paths inside SAFE product folder
        (data_rel_paths, metadata_rel_paths, calibration_rel_paths, acquisition_time, footprint) = (
            self._manifest.parse_manifest_document()
        )
        # validating channel data pairs (raster + metadata)
        self._validate_channel_pairs(data_rel_paths, metadata_rel_paths)

        # extracting full path to raster and metadata files
        self._data_paths = self._get_full_file_paths(data_rel_paths)
        self._metadata_paths = self._get_full_file_paths(metadata_rel_paths)
        self._calibration_paths = self._get_full_file_paths(calibration_rel_paths)
        self._channels_number = len(self._data_paths)

        # acquisition time
        self._acq_time = acquisition_time

        # footprint
        self._footprint = footprint

        # computing channel list
        self._channel_list_by_swath_id = ["-".join(m.split("/")[-1].split("-")[:4]) for m in metadata_rel_paths]

    @staticmethod
    def _validate_channel_pairs(data_list: list[str], metadata_list: list[str]) -> None:
        """Checking validity of metadata and raster pairs.

        Parameters
        ----------
        data_list : list[str]
            list of raster data files
        metadata_list : list[str]
            list of metadata files
        """
        if not len(data_list) == len(metadata_list):
            raise ChannelDataPairMismatch(
                f"Metadata number != raster data number ({len(metadata_list)} != {len(data_list)})"
            )
        data_names = [d.split("/")[-1].strip(_IMAGE_FORMAT) for d in data_list]
        metadata_names = [m.split("/")[-1].strip(_METADATA_EXTENSION) for m in metadata_list]
        if not metadata_names == data_names:
            raise ChannelDataPairMismatch("Metadata file names are not the same as the rasters'")

    def _get_full_file_paths(self, rel_paths: list[str]) -> list[Path]:
        """Generating full paths from relative ones by prepending the SAFE product folder path.

        Parameters
        ----------
        rel_paths : list[str]
            list of relative paths inside the SAFE product folder

        Returns
        -------
        list[Path]
            list of full paths
        """
        abs_paths = [self._product_path.joinpath(rel.replace("./", "", 1)) for rel in rel_paths]
        return abs_paths

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def data_list(self) -> list[Path]:
        """Returning the list of raster data files in SAFE product"""
        return self._data_paths

    @property
    def metadata_list(self) -> list[Path]:
        """Returning the list of metadata files in SAFE product"""
        return self._metadata_paths

    @property
    def calibration_list(self) -> list[Path]:
        """Returning the list of calibration files in SAFE product"""
        return self._calibration_paths

    @property
    def channels_number(self) -> int:
        """Returning the number of channels in the SAFE product"""
        return self._channels_number

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (swath-polarization)"""
        return self._channel_list_by_swath_id

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_files_from_channel_name(self, channel_name: str) -> tuple[Path, Path, Path]:
        """Get metadata, raster and calibration file paths associated to input channel name.

        Parameters
        ----------
        channel_name : str
            selected channel name

        Returns
        -------
        tuple[Path, Path, Path]
            metadata file path,
            raster file path,
            calibration file path
        """
        metadata = [m for m in self.metadata_list if channel_name in m.name][0]
        raster = [r for r in self.data_list if channel_name in r.name][0]
        calibration = [c for c in self.calibration_list if channel_name in c.name][0]
        return metadata, raster, calibration


def is_s1_safe_product(product: Union[str, Path]) -> bool:
    """Check if input path corresponds to a valid S1 Safe product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - path ends with SAFE
        - S1Product can be instantiated

    Parameters
    ----------
    product : Union[str, Path]
        path to the product to be checked

    Returns
    -------
    bool
        True if it is a valid S1 Safe, else False
    """
    product = Path(product)

    if not product.exists() or not product.is_dir():
        return False

    if not product.name.endswith(".SAFE"):
        return False

    try:
        S1Product(product)
    except Exception:
        return False

    return True
