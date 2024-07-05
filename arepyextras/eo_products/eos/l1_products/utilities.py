# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
EOS04 reader support module
---------------------------
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Union

import arepytools.io.metadata as meta
import numpy as np
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.math.genericpoly import SortedPolyList, create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime
from lxml import etree
from numpy.polynomial.polynomial import Polynomial
from scipy.constants import speed_of_light as LIGHT_SPEED

from arepyextras.eo_products.common.utilities import (
    ConversionPolynomial,
    OrbitDirection,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
)

RASTER_EXTENSION = ".tif"
METADATA_EXTENSION = ".xml"


class InvalidEOS04Product(RuntimeError):
    """Invalid EOS04 Product"""


class EOS04TimeOrdering(Enum):
    """EOS04 available Time Ordering"""

    INCREASING = auto()
    DECREASING = auto()


class EOS04AcquisitionMode(Enum):
    """EOS04 Acquisition Modes"""

    SCANSAR = "SCANSAR"


class EOS04ProductType(Enum):
    """EOS04 Product Types"""

    SLC = "SLC"
    GRD = "GROUND RANGE"


def _parse_timestamp(timestamp: float) -> PreciseDateTime:
    """Parsing UTC timestamp referred to 01/01/1970.

    Parameters
    ----------
    timestamp : float
        float timestamp

    Returns
    -------
    PreciseDateTime
        PreciseDateTime date format
    """
    return PreciseDateTime.fromisoformat(
        datetime.fromtimestamp(timestamp, tz=timezone.utc).replace(tzinfo=None).isoformat()
    )


def _get_basic_info_from_metadata(
    metadata_path: Path,
) -> tuple[PreciseDateTime, list[int], list[SARPolarization], EOS04ProductType]:
    """Get the product acquisition time, polarizations and beams from metadata file.

    Parameters
    ----------
    metadata_path : Path
        path to the EOS04 metadata file

    Returns
    -------
    PreciseDateTime
        acquisition time
    list[int]
        list of beams
    list[SARPolarization]
        list of channels polarizations
    EOS04ProductType
        Product type
    """
    mtd = metadata_path.read_text(encoding="UTF-8")
    acq_time_re = re.compile("(?<=<StartTime>).*(?=</StartTime>)")
    beams_re = re.compile("(?<=<BeamID>).*(?=</BeamID>)")
    polarizations_re = re.compile("(?<=<Polarizations>).*(?=</Polarizations>)")
    product_type_re = re.compile("(?<=<ProductType>).*(?=</ProductType>)")
    acquisition_time = acq_time_re.findall(mtd)[0]
    product_type = EOS04ProductType(product_type_re.findall(mtd)[0])
    beams = list(range(0, len(beams_re.findall(mtd)[0].split())))
    polarizations = [SARPolarization[p.upper()] for p in polarizations_re.findall(mtd)[0].split()]
    return PreciseDateTime.fromisoformat(acquisition_time), beams, polarizations, product_type


def _retrieve_scene_footprint(metadata_path: Path) -> tuple[float, float, float, float]:
    """Product footprint as tuple of (min lat, max lat, min lon, max lon).

    Parameters
    ----------
    metadata_path : Path
        Path to the metadata .xml file

    Returns
    -------
    tuple[float, float, float, float]
        (min lat, max lat, min lon, max lon)
    """
    mtd = metadata_path.read_text(encoding="UTF-8")
    footprint_re = re.compile('(?<=<SourceDataGeometry type="WKT">).*(?=</SourceDataGeometry>)')
    footprint = footprint_re.findall(mtd)[0]
    footprint = [float(f) for f in footprint.replace("Polygon", "").strip("() ").replace(" ", ",").split(",")]
    longitudes, latitudes = footprint[0::2], footprint[1::2]
    return (min(latitudes), max(latitudes), min(longitudes), max(longitudes))


def compose_channel_name(polarization: SARPolarization, beam: int) -> str:
    """Composing channel name from polarization and beam.

    Parameters
    ----------
    polarization : SARPolarization
        channel polarization
    beam : int
        channel beam

    Returns
    -------
    str
        channel name, as B{beam}_POL
    """
    return "_".join([f"B{str(beam)}", polarization.name])


def unpack_channel_name(channel_name: str) -> tuple[int, SARPolarization]:
    """Recovering beam id and polarization value from channel name.

    Parameters
    ----------
    channel_name : str
        channel name string

    Returns
    -------
    int
        channel beam id
    SARPolarization
        channel polarization
    """
    beam_pol = channel_name.split("_")
    return int(beam_pol[0].strip("B")), SARPolarization[beam_pol[1]]


def general_sar_orbit_from_eos04_state_vectors(state_vectors: EOS04StateVectors) -> GeneralSarOrbit:
    """Creating a GeneralSarOrbit from product state vectors.

    Parameters
    ----------
    state_vectors : EOS04StateVectors
        state vectors of EOS04 product

    Returns
    -------
    GeneralSarOrbit
        General SAR Orbit from State Vectors
    """
    return GeneralSarOrbit(time_axis=state_vectors.time_axis, state_vectors=state_vectors.positions.ravel())


def raster_info_from_metadata_nodes(
    image_generation_parameters_node: etree._Element,
    image_attributes_node: etree._Element,
    beam_id: int,
    polarization: SARPolarization,
    product_type: EOS04ProductType,
) -> meta.RasterInfo:
    """Creating a RasterInfo Arepytools metadata element from xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    image_attributes_node : etree._Element
        ImageAttributes metadata xml node
    beam_id : int
        swath beam id
    product_type : EOS04ProductType
        product type

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """

    # swath timing for this channel
    if product_type == EOS04ProductType.SLC:
        swath_timing_node = [
            s
            for s in image_generation_parameters_node.findall("swathTiming")
            if s.find("swath").get("pol") == polarization.name
        ][beam_id]
        bursts_num = int(swath_timing_node.find("burstList").get("count"))
        bursts = swath_timing_node.findall("burstList/burst")

        # azimuth
        lines = int(swath_timing_node.find("linesPerBurst").text) * bursts_num
        lines_step = 1 / float(swath_timing_node.find("swathPRF").text)
        lines_start = _parse_timestamp(timestamp=float(bursts[0].find("firstValidLineTime").text))
        lines_start_unit = "Utc"
        lines_step_unit = "s"

        # samples
        samples = int(swath_timing_node.find("samplesPerBurst").text)
        samples_step = 2 / LIGHT_SPEED * float(swath_timing_node.find("swathRangeSampling").text)
        samples_start = 2 / LIGHT_SPEED * float(bursts[0].find("firstSampleRange").text)
        samples_start_unit = "s"
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"
    else:
        # TODO: check all this
        # azimuth
        lines = int(image_attributes_node.find("RasterAttributes/NumberOfLines").text)
        lines_start = PreciseDateTime.fromisoformat(
            image_generation_parameters_node.find("SarProcessingInformation/ZeroDopplerTimeFirstLine").text
        )
        lines_stop = PreciseDateTime.fromisoformat(
            image_generation_parameters_node.find("SarProcessingInformation/ZeroDopplerTimeLastLine").text
        )
        lines_step = (lines_stop - lines_start) / (lines - 1)
        lines_start_unit = "Utc"
        lines_step_unit = "s"

        # ground range
        samples = int(image_attributes_node.find("RasterAttributes/NumberOfSamplesPerLine").text)
        samples_start = 0
        samples_start_unit = "m"
        samples_step = float(image_generation_parameters_node.find("SarProcessingInformation/RangePixelSpacing").text)
        samples_step_unit = "m"
        celltype = "FLOAT32"

    raster_info = meta.RasterInfo(
        lines=lines,
        samples=samples,
        celltype=celltype,
        filename=None,
        header_offset_bytes=0,
        row_prefix_bytes=0,
        byteorder="LITTLEENDIAN",
    )
    raster_info.set_lines_axis(
        lines_start=lines_start,
        lines_start_unit=lines_start_unit,
        lines_step=lines_step,
        lines_step_unit=lines_step_unit,
    )
    raster_info.set_samples_axis(
        samples_start=samples_start,
        samples_start_unit=samples_start_unit,
        samples_step=samples_step,
        samples_step_unit=samples_step_unit,
    )

    return raster_info


def dataset_info_from_metadata_node(
    source_attributes_node: etree._Element, projection: SARProjection
) -> meta.DataSetInfo:
    """Creating a DataSetInfo Arepytools metadata element from safe xml nodes.

    Parameters
    ----------
    source_attributes_node : etree._Element
        SourceAttributes metadata xml node
    projection : SARProjection
        product projection

    Returns
    -------
    DataSetInfo
        DataSetInfo metadata object
    """
    sensor_name = source_attributes_node.find("Satellite").text
    fc_hz = float(source_attributes_node.find("SourceDataAcquisitionParameters/RadarCenterFrequency").text)
    acq_mode = EOS04AcquisitionMode(source_attributes_node.find("SourceDataAcquisitionParameters/ObservationMode").text)

    dataset_info = meta.DataSetInfo(acquisition_mode_i=acq_mode.value, fc_hz_i=fc_hz)
    dataset_info.sensor_name = sensor_name
    dataset_info.image_type = (
        "MULTILOOK" if projection == SARProjection.GROUND_RANGE else "AZIMUTH FOCUSED RANGE COMPENSATED"
    )
    dataset_info.projection = projection.value
    dataset_info.side_looking = meta.ESideLooking(
        source_attributes_node.find("SourceDataAcquisitionParameters/AntennaPointing").text.upper()
    )

    return dataset_info


def doppler_centroid_poly_from_metadata_node(
    image_generation_parameters_node: etree._Element, raster_info: meta.RasterInfo
) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler centroid polynomial wrapper from safe xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    raster_info : meta.RasterInfo
        product raster info

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerCentroidVector metadata object
    """

    coeff_raw = [
        [float(c) for c in cc.text.split()]
        for cc in image_generation_parameters_node.findall("DopplerCentroid/DopplerCentroidCoefficients")
    ]
    ref_times = [
        PreciseDateTime.fromisoformat(tt.text)
        for tt in image_generation_parameters_node.findall("DopplerCentroid/TimeOfDopplerCentroidEstimate")
    ]
    coefficients = [
        [
            c[0],
            c[1] / raster_info.samples_step,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        for c in coeff_raw
    ]
    coefficients_max_num = np.amin(np.array([len(coefficients[0]) - 2, len(coeff_raw[0])]))
    for index, _ in enumerate(coefficients):
        for idx in range(2, coefficients_max_num):
            coefficients[index][idx + 2] = coeff_raw[index][idx] / (raster_info.samples_step**idx)

    doppler_centroids = [
        meta.DopplerCentroid(i_ref_az=ref_times[c], i_ref_rg=raster_info.samples_start, i_coefficients=coefficients[c])
        for c in range(len(coefficients))
    ]

    return create_sorted_poly_list(poly2d_vector=meta.DopplerCentroidVector(doppler_centroids))


def doppler_rate_poly_from_metadata_node(
    image_generation_parameters_node: etree._Element, raster_info: meta.RasterInfo
) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler centroid rate polynomial wrapper from safe xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        ImageGenerationParameters metadata xml node
    raster_info : meta.RasterInfo
        product raster info

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerCentroidVector metadata object
    """

    coeff_raw = [
        [float(c) for c in cc.text.split()]
        for cc in image_generation_parameters_node.findall("DopplerRateValues/DopplerRateValuesCoefficients")
    ]
    ref_times = [
        [PreciseDateTime.fromisoformat(t) for t in tt.text.split()]
        for tt in image_generation_parameters_node.findall("DopplerRateValues/DopplerRateReferenceTime")
    ]
    coefficients = [
        [
            c[0],
            c[1] / raster_info.samples_step,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
        for c in coeff_raw
    ]
    coefficients_max_num = np.amin(np.array([len(coefficients[0]) - 2, len(coeff_raw[0])]))
    for index, _ in enumerate(coefficients):
        for idx in range(2, coefficients_max_num):
            coefficients[index][idx + 2] = coeff_raw[index][idx] / (raster_info.samples_step**idx)

    doppler_rates = [
        meta.DopplerRate(i_ref_az=ref_times[c][0], i_ref_rg=raster_info.samples_start, i_coefficients=coefficients[c])
        for c in range(len(coefficients))
    ]

    return create_sorted_poly_list(poly2d_vector=meta.DopplerRateVector(doppler_rates))


def acquisition_timeline_from_metadata_node(
    image_generation_parameters_node: etree._Element, raster_info: meta.RasterInfo
) -> meta.AcquisitionTimeLine:
    """Creating a AcquisitionTimeLine Arepytools metadata element from safe xml nodes.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        downlinkInformation metadata xml node
    raster_info : meta.RasterInfo
        product raster info

    Returns
    -------
    AcquisitionTimeLine
        AcquisitionTimeLine metadata object
    """

    return meta.AcquisitionTimeLine(
        swst_changes_number_i=1,
        swst_changes_azimuth_times_i=[0],
        swst_changes_values_i=[0],  # TODO: which is the SWST value?
        swl_changes_number_i=1,
        swl_changes_azimuth_times_i=[0],
        swl_changes_values_i=[raster_info.samples_step * (raster_info.samples - 1)],
    )


def pulse_info_from_metadata_nodes(source_attributes_node: etree._Element, samples_step: float) -> meta.Pulse:
    """Creating a Pulse Arepytools dataclass from xml nodes.

    Parameters
    ----------
    source_attributes_node : etree._Element
        SourceAttributes metadata xml node
    samples_step : float
        raster info samples step

    Returns
    -------
    meta.Pulse
        Pulse info dataclass
    """

    # TODO: forcing this to be UPCHIRP but this is just to generate the proper Pulse metadata, this is not TRUE!!
    pulse_direction = meta.EPulseDirection.up.value
    pulse_bandwidth = float(source_attributes_node.find("SourceDataAcquisitionParameters/PulseBandwidth").text)
    pulse_length = float(source_attributes_node.find("SourceDataAcquisitionParameters/PulseLength").text)
    pulse_start_frequency = -pulse_bandwidth / 2

    return meta.Pulse(
        i_pulse_length=pulse_length,
        i_bandwidth=pulse_bandwidth,
        i_pulse_sampling_rate=1 / samples_step,
        i_pulse_energy=1,
        i_pulse_start_frequency=pulse_start_frequency,
        i_pulse_start_phase=0,
        i_pulse_direction=pulse_direction,
    )


@dataclass
class EOS04BurstInfo:
    """EOS04 swath burst info"""

    num: int  # number of bursts in this swath
    lines_per_burst: int  # number of azimuth lines within each burst
    samples_per_burst: int  # number of range samples within each burst
    first_valid_lines: np.ndarray  # first valid azimuth line within each burst
    first_valid_samples: np.ndarray  # first valid azimuth line within each burst
    azimuth_start_times: np.ndarray  # zero doppler azimuth time of the first line for each burst
    range_start_times: np.ndarray  # zero doppler range time of the first sample for each burst

    @staticmethod
    def from_metadata_node(
        image_generation_parameters_node: etree._Element, polarization: SARPolarization, beam_id: int
    ) -> EOS04BurstInfo:
        """Generating EOS04BurstInfo object directly from metadata xml nodes.

        Parameters
        ----------
        image_generation_parameters_node : etree._Element
            ImageGenerationParameters xml node
        polarization : SARPolarization
            product acquisition mode
        beam_id : int
            channel beam id

        Returns
        -------
        EOS04BurstInfo
            burst info dataclass
        """
        swath_timing_node = [
            s
            for s in image_generation_parameters_node.findall("swathTiming")
            if s.find("swath").get("pol") == polarization.name
        ][beam_id]
        range_start_times = [
            2 / LIGHT_SPEED * float(s.text) for s in swath_timing_node.findall("burstList/burst/firstSampleRange")
        ]
        azimuth_start_times = [
            _parse_timestamp(float(s.text)) for s in swath_timing_node.findall("burstList/burst/firstValidLineTime")
        ]
        first_valid_lines = [int(s.text) for s in swath_timing_node.findall("burstList/burst/firstValidLine")]
        first_valid_samples = [int(s.text) for s in swath_timing_node.findall("burstList/burst/firstValidSample")]
        lines = int(swath_timing_node.find("linesPerBurst").text)
        samples = int(swath_timing_node.find("samplesPerBurst").text)
        return EOS04BurstInfo(
            num=len(range_start_times),
            lines_per_burst=lines,
            samples_per_burst=samples,
            first_valid_lines=first_valid_lines,
            first_valid_samples=first_valid_samples,
            azimuth_start_times=np.array(azimuth_start_times),
            range_start_times=np.array(range_start_times),
        )


@dataclass
class EOS04SwathInfo:
    """EOS04 swath info"""

    rank: int
    azimuth_steering_rate_poly: tuple[float, float, float]
    prf: float

    @staticmethod
    def from_metadata_nodes(
        image_generation_parameters_node: etree._Element, polarization: SARPolarization, beam_id: int
    ) -> EOS04SwathInfo:
        """Generating EOS04SwathInfo object directly from metadata xml nodes.

        Parameters
        ----------
        image_generation_parameters_node : etree._Element
            ImageGenerationParameters xml node
        polarization : SARPolarization
            product acquisition mode
        beam_id : int
            channel beam id

        Returns
        -------
        EOS04SwathInfo
            swath info dataclass
        """
        rank = 0
        swath_timing_node = [
            s
            for s in image_generation_parameters_node.findall("swathTiming")
            if s.find("swath").get("pol") == polarization.name
        ][beam_id]
        prf = float(swath_timing_node.find("swathPRF").text)

        return EOS04SwathInfo(rank=rank, azimuth_steering_rate_poly=(0, 0, 0), prf=prf)


@dataclass
class EOS04StateVectors:
    """EOS04 orbit's state vectors"""

    num: int  # attitude data numerosity
    orbit_direction: OrbitDirection  # orbit direction: ascending or descending
    positions: np.ndarray  # platform position data with respect to the Earth-fixed reference frame
    velocities: np.ndarray  # platform velocity data with respect to the Earth-fixed reference frame
    time_axis: np.ndarray  # PreciseDateTime axis at which orbit state vectors apply
    time_step: float  # time axis step

    @staticmethod
    def _unpack_state_vector_from_orbit_node(
        orbit_data_node: etree._Element,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracting an array of shape (N, 3) with x, y, z components for positions and velocities of state vectors.
        Recovering also the time axis in PreciseDateTime format.

        Parameters
        ----------
        orbit_data_node : etree._Element
            OrbitData node from .xml metadata

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            positions (N, 3) array,
            velocities (N, 3) array,
            time axis (N,) array
        """
        pos_x = np.array([float(p.text) for p in orbit_data_node.findall("StateVectorECEF/xPosition")])
        pos_y = np.array([float(p.text) for p in orbit_data_node.findall("StateVectorECEF/yPosition")])
        pos_z = np.array([float(p.text) for p in orbit_data_node.findall("StateVectorECEF/zPosition")])
        positions = np.stack([pos_x, pos_y, pos_z], axis=1)

        vel_x = np.array([float(p.text) for p in orbit_data_node.findall("StateVectorECEF/xVelocity")])
        vel_y = np.array([float(p.text) for p in orbit_data_node.findall("StateVectorECEF/yVelocity")])
        vel_z = np.array([float(p.text) for p in orbit_data_node.findall("StateVectorECEF/zVelocity")])
        velocities = np.stack([vel_x, vel_y, vel_z], axis=1)

        time_axis = np.array(
            [PreciseDateTime.fromisoformat(p.text) for p in orbit_data_node.findall("StateVectorECEF/TimeStamp")]
        )

        return positions, velocities, time_axis

    @staticmethod
    def from_metadata_node(orbit_information_node: etree._Element) -> EOS04StateVectors:
        """Generating a EOS04StateVectors object directly from metadata xml node.

        Parameters
        ----------
        orbit_data_node : etree._Element
            OrbitInformation xml node

        Returns
        -------
        EOS04StateVectors
            orbit's state vectors dataclass
        """

        positions, velocities, times = EOS04StateVectors._unpack_state_vector_from_orbit_node(
            orbit_data_node=orbit_information_node
        )
        numerosity = times.size
        assert positions.shape == velocities.shape == (numerosity, 3)

        mean_delta_time = np.diff(times).mean()

        return EOS04StateVectors(
            num=numerosity,
            orbit_direction=OrbitDirection[orbit_information_node.find("PassDirection").text],
            positions=positions,
            velocities=velocities,
            time_axis=times,
            time_step=mean_delta_time,
        )


@dataclass
class EOS04Attitude:
    """EOS04 sensor's attitude"""

    num: int  # attitude data numerosity
    yaw: np.ndarray  # platform yaw
    pitch: np.ndarray  # platform pitch
    roll: np.ndarray  # platform roll
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies
    time_step: float  # time axis step

    @staticmethod
    def from_metadata_node(attitude_information_node: etree._Element) -> EOS04Attitude:
        """Generating EOS04Attitude object directly from metadata xml node.

        Parameters
        ----------
        attitude_information_node : etree._Element
            AttitudeInformation xml node

        Returns
        -------
        EOS04Attitude
            sensor's attitude dataclass
        """

        time_axis = np.array(
            [
                PreciseDateTime.fromisoformat(t.text)
                for t in attitude_information_node.findall("AttitudeAngles/TimeStamp")
            ]
        )
        yaw = np.array([float(t.text) for t in attitude_information_node.findall("AttitudeAngles/yaw")])
        roll = np.array([float(t.text) for t in attitude_information_node.findall("AttitudeAngles/roll")])
        pitch = np.array([float(t.text) for t in attitude_information_node.findall("AttitudeAngles/pitch")])

        assert time_axis.size == yaw.size == roll.size == pitch.size

        return EOS04Attitude(
            num=time_axis.size,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            time_axis=time_axis,
            time_step=time_axis[1] - time_axis[0],
        )


@dataclass
class EOS04CoordinateConversions:
    """EOS04 coordinate conversion"""

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
    def from_metadata_node(
        image_generation_parameters_node: etree._Element, raster_info: meta.RasterInfo
    ) -> EOS04CoordinateConversions:
        """Generating EOS04CoordinateConversions object directly from metadata xml node.

        About coefficients annotated in metadata:

        Coefficients of polynomial fit to the "Ground to Slant Range" transform applied. Fixed along all image slices.
        Defined with respect to (slant range in m)/(pixel no.)^n where first pixel in line is 0.

        Values output in order A0, A1, ...,  An, in order of increasing degree.
        Polynomial to be evaluated is:  A_0+ A_1 x + ... + A_n x^n
        where x is pixel number in the line, starting from 0.

        Evaluated polynomial gives Slant Range in meters.

        Parameters
        ----------
        image_generation_parameters_node : etree._Element
            ImageGenerationParameters metadata xml node
        raster_info : meta.RasterInfo
            product raster info

        Returns
        -------
        EOS04CoordinateConversions
            polynomial for coordinate conversion dataclass
        """

        # <SlantRangeToGroundRange>
        #   <ZeroDopplerAzimuthTime>2022-04-04T01:55:47.811000000Z</ZeroDopplerAzimuthTime>
        #   <SlantRangeTimeToFirstRangeSample units="s">0.004845699792411</SlantRangeTimeToFirstRangeSample>
        #   <SlantToGroundRangeCoefficients units="m">726351.800000 12.763530 0.000128</SlantToGroundRangeCoefficients>
        # </SlantRangeToGroundRange>

        if image_generation_parameters_node.find("SlantRangeToGroundRange") is None:
            return EOS04CoordinateConversions()

        node = image_generation_parameters_node.find("SlantRangeToGroundRange")
        # recovering coefficients and applying conversion factor meters to seconds
        az_ref_time = PreciseDateTime.fromisoformat(node.find("ZeroDopplerAzimuthTime").text)
        m2s_conversion_factor = 1 / (LIGHT_SPEED / 2)
        coeff_raw = [m2s_conversion_factor * float(c) for c in node.find("SlantToGroundRangeCoefficients").text.split()]
        ground_to_slant_coeff = [c / raster_info.samples_step**idx for idx, c in enumerate(coeff_raw)]
        ground_to_slant_poly = Polynomial(ground_to_slant_coeff)

        # slant to ground poly is not given, so it must be evaluated by inverting the ground to slant poly
        rng_axis = np.arange(0, (raster_info.samples + 1) * raster_info.samples_step, raster_info.samples_step)
        ground_to_slant_poly_evaluated = ground_to_slant_poly(rng_axis)
        slant_to_ground_poly = Polynomial.fit(
            x=ground_to_slant_poly_evaluated, y=rng_axis, deg=ground_to_slant_poly.degree()
        )

        ground_to_slant_poly_list = [
            ConversionPolynomial(
                azimuth_reference_time=raster_info.lines_start,
                origin=0,
                polynomial=ground_to_slant_poly,
            )
        ]
        slant_to_ground_poly_list = [
            ConversionPolynomial(
                azimuth_reference_time=raster_info.lines_start,
                origin=0,
                polynomial=slant_to_ground_poly,
            )
        ]

        return EOS04CoordinateConversions(
            azimuth_reference_times=az_ref_time,
            ground_to_slant=ground_to_slant_poly_list,
            slant_to_ground=slant_to_ground_poly_list,
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
class EOS04GeneralChannelInfo:
    """EOS04 general channel info representation dataclass"""

    channel_id: str
    product_name: str
    satellite: str
    swath: str
    acq_start_time: PreciseDateTime
    product_type: EOS04ProductType
    acquisition_mode: EOS04AcquisitionMode
    polarization: SARPolarization
    projection: SARProjection
    orbit_direction: OrbitDirection

    @staticmethod
    def from_metadata_node(
        source_attributes_node: etree._Element, product_type: str, channel_id: str
    ) -> EOS04GeneralChannelInfo:
        """Generating EOS04GeneralChannelInfo object directly from metadata xml nodes.

        Parameters
        ----------
        source_attributes_node : etree._Element
            SourceAttributes metadata xml node
        prod_type : str
            product type
        channel_id : str
            channel id

        Returns
        -------
        EOS04GeneralChannelInfo
            general channel info dataclass
        """

        start_time = PreciseDateTime.fromisoformat(
            source_attributes_node.find("SourceDataAcquisitionTime/StartTime").text
        )
        product_type = EOS04ProductType(product_type)
        acq_mode = EOS04AcquisitionMode(
            source_attributes_node.find("SourceDataAcquisitionParameters/ObservationMode").text
        )
        projection = SARProjection.SLANT_RANGE if product_type == EOS04ProductType.SLC else SARProjection.GROUND_RANGE
        orbit_direction = OrbitDirection[
            source_attributes_node.find("OrbitAndAttitude/OrbitInformation/PassDirection").text
        ]
        beam, polarization = unpack_channel_name(channel_name=channel_id)

        return EOS04GeneralChannelInfo(
            channel_id=channel_id,
            swath=f"B{str(beam)}",
            product_name=source_attributes_node.find("ProductID").text,
            satellite=source_attributes_node.find("Satellite").text,
            acq_start_time=start_time,
            product_type=product_type,
            acquisition_mode=acq_mode,
            projection=projection,
            polarization=polarization,
            orbit_direction=orbit_direction,
        )


@dataclass
class EOS04ChannelMetadata:
    """EOS04 channel metadata xml file wrapper"""

    channel_id: str
    general_info: EOS04GeneralChannelInfo
    general_sar_orbit: GeneralSarOrbit
    attitude: EOS04Attitude
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: EOS04BurstInfo
    raster_info: meta.RasterInfo
    dataset_info: meta.DataSetInfo
    swath_info: EOS04SwathInfo
    sampling_constants: SARSamplingFrequencies
    acquisition_timeline: meta.AcquisitionTimeLine
    doppler_centroid_poly: SortedPolyList
    doppler_rate_poly: SortedPolyList
    pulse: meta.Pulse
    coordinate_conversions: EOS04CoordinateConversions
    state_vectors: EOS04StateVectors


class EOS04FolderLayout:
    """EOS04 file main directory architecture"""

    def __init__(self, path: Path) -> None:
        """Definition of internal architecture of a EOS04 product folder.

        Parameters
        ----------
        path : Path
            path to the EOS04 product base folder
        """
        self._product_path = path
        self._product_name = path.name
        self._band_meta_file = path.joinpath("BAND_META.txt")
        self._metadata_file = path.joinpath("product" + METADATA_EXTENSION)

    @property
    def band_meta_file(self) -> Path:
        """Path to the BAND_META.txt file"""
        return self._band_meta_file

    @property
    def metadata_file(self) -> Path:
        """Path to the product.xml metadata file"""
        return self._metadata_file

    def get_slant_range_grid_file(self, polarization: str | SARPolarization) -> Path:
        """Retrieving the _L1_SlantRange_grid.txt file for the input polarization.

        Parameters
        ----------
        polarization : str | SARPolarization
            polarization value

        Returns
        -------
        Path
            Path to the _L1_SlantRange_grid.txt for the selected polarization
        """
        pol = SARPolarization(polarization).name.upper()
        return self._product_path.joinpath("_".join([self._product_name, pol, "L1_Slant_Range_grid.txt"]))

    def get_beam_raster_file(
        self, polarization: str | SARPolarization, beam: int, product_type: EOS04ProductType
    ) -> Path:
        """Retrieving the raster file for the selected polarization and beam number.

        Parameters
        ----------
        polarization : str | SARPolarization
            polarization value
        beam : int
            selected beam number
        product_type : EOS04ProductType
            product type

        Returns
        -------
        Path
            Path to the beam raster file
        """
        pol = SARPolarization(polarization).name.upper()
        scene_folder = self._product_path.joinpath(f"scene_{pol}")
        if product_type == EOS04ProductType.GRD:
            return scene_folder.joinpath(f"imagery_{pol}" + RASTER_EXTENSION)
        return scene_folder.joinpath(f"imagery_{pol}_b{beam}" + RASTER_EXTENSION)


class EOS04Product:
    """EOS04 Product"""

    def __init__(self, path: str | Path) -> None:
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._layout = EOS04FolderLayout(self._product_path)

        # acquisition time, beams and polarizations
        self._acq_time, self._beams, self._pol_list, self._product_type = _get_basic_info_from_metadata(
            self._layout.metadata_file
        )

        if self._product_type == EOS04ProductType.GRD:
            # GRD products still presents beams in metadata but actually there is no dependency on beams! taking the
            # first one only to keep the name for the only available raster
            self._beams = [self._beams[0]]

        self._footprint = _retrieve_scene_footprint(self._layout.metadata_file)

        self._channels = [compose_channel_name(p, b) for p in self._pol_list for b in self._beams]

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def metadata_file(self) -> Path:
        """Returning the Path to the product metadata file"""
        return self._layout.metadata_file

    @property
    def channels_number(self) -> int:
        """Returning the number of channels for this product"""
        return len(self._channels)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (beam-polarization)"""
        return self._channels

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_raster_file_from_channel_name(self, channel_name: str) -> Path:
        """Get raster file path associated to input channel name.

        Parameters
        ----------
        channel_name : str
            selected channel name

        Returns
        -------
        Path
            raster file path
        """
        beam, pol = unpack_channel_name(channel_name)
        return self._layout.get_beam_raster_file(polarization=pol, beam=beam, product_type=self._product_type)


def is_eos04_product(product: Union[str, Path]) -> bool:
    """Check if input path corresponds to a valid EOS04 product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - metadata file exists
        - metadata basic info extraction works

    Parameters
    ----------
    product : Union[str, Path]
        path to the product to be checked

    Returns
    -------
    bool
        True if it is a valid product, else False
    """
    product = Path(product)

    if not product.exists() or not product.is_dir():
        return False

    try:
        layout = EOS04FolderLayout(path=product)
    except Exception:
        return False

    if not layout.metadata_file.is_file():
        return False

    try:
        _, _, _, _ = _get_basic_info_from_metadata(layout.metadata_file)
    except Exception:
        return False

    return True
