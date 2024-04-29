# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
NOVASAR reader support module
-----------------------------
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Union

import arepytools.io.metadata as meta
import numpy as np
import numpy.typing as npt
from arepytools.constants import LIGHT_SPEED
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.math.genericpoly import SortedPolyList, create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime
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

_IMAGE_FORMAT = ".tif"
_METADATA_EXTENSION = ".xml"


class NovaSAR1ProductType(Enum):
    """NovaSAR-1 L1 product types"""

    SLC = "SLC"  # stripmap, single look, complex, slant range
    SRD = "SRD"  # stripmap, multi-look, detected, slant range
    GRD = "GRD"  # stripmap, multi-look, detected, ground range
    SCD = "SCD"  # scanSAR, multi-look, detected, ground range


class NovaSAR1AcquisitionModes(Enum):
    """NovaSAR-1 L1 acquisition modes"""

    STRIPMAP = auto()
    SCANSAR = auto()


class NovaSAR1ReferenceFrameType(Enum):
    """NovaSAR-1 available reference frames"""

    ZERO_DOPPLER = auto()


class NovaSAR1TimeOrdering(Enum):
    """NovaSAR-1 available Time Ordering"""

    INCREASING = auto()
    DECREASING = auto()


def get_basic_info_from_metadata(
    metadata_path: Union[str, Path]
) -> tuple[PreciseDateTime, NovaSAR1ProductType, list[str], tuple[float, float, float, float]]:
    """Recovering acquisition time and list of channels.

    Parameters
    ----------
    metadata_path : Union[str, Path]
        Path to NovaSAR-1 metadata file

    Returns
    -------
    tuple[PreciseDateTime, NovaSAR1ProductTypes, list[str]]
        acquisition time in PreciseDateTime format,
        product type,
        list of channels ids,
        scene footprint [min lat, max lat, min lon, max lon]

    Raises
    ------
    RuntimeError
        if acquisition mode is not stripmap nor scansar
    """
    metadata_path = Path(metadata_path)
    mtd = metadata_path.read_text(encoding="UTF-8")

    # regex init
    acq_time_re = re.compile("(?<=<RawDataStartTime>).*(?=</RawDataStartTime>)")
    type_re = re.compile("(?<=<ProductType>).*(?=</ProductType>)")
    pols_re = re.compile("(?<=<Polarisations>).*(?=</Polarisations>)")
    footprint_lat_re = re.compile('(?<=<Latitude units="deg">).*(?=</Latitude>)')
    footprint_lon_re = re.compile('(?<=<Longitude units="deg">).*(?=</Longitude>)')

    # info extraction
    acq_time = acq_time_re.findall(mtd)[0]
    acq_type = type_re.findall(mtd)[0].lower()
    acq_pols = pols_re.findall(mtd)[0].lower()

    # generating channels names
    pol_list = acq_pols.split()
    channels_list = [acq_type + "_" + pol for pol in pol_list]

    # recovering scene footprint
    footprint_lat = [float(f) for f in footprint_lat_re.findall(mtd)]
    footprint_lon = [float(f) for f in footprint_lon_re.findall(mtd)]
    footprint = (min(footprint_lat), max(footprint_lat), min(footprint_lon), max(footprint_lon))

    return PreciseDateTime.from_utc_string(acq_time), NovaSAR1ProductType(acq_type.upper()), channels_list, footprint


def get_acquisition_mode_from_product_type(prod_type: NovaSAR1ProductType) -> NovaSAR1AcquisitionModes:
    """Get product acquisition mode from product type.

    Parameters
    ----------
    prod_type : NovaSAR1ProductTypes
        product type

    Returns
    -------
    NovaSAR1AcquisitionModes
        product acquisition mode
    """
    if prod_type == NovaSAR1ProductType.SCD:
        return NovaSAR1AcquisitionModes.SCANSAR

    return NovaSAR1AcquisitionModes.STRIPMAP


def get_projection_from_product_type(prod_type: NovaSAR1ProductType) -> SARProjection:
    """Get product projection from product type.

    Parameters
    ----------
    prod_type : NovaSAR1ProductTypes
        product type

    Returns
    -------
    SARProjection
        product projection
    """
    if prod_type in (NovaSAR1ProductType.SLC, NovaSAR1ProductType.SRD):
        return SARProjection.SLANT_RANGE

    return SARProjection.GROUND_RANGE


def raster_info_from_metadata_nodes(
    image_generation_parameters_node: etree._Element,
    image_attributes_node: etree._Element,
    product_type: NovaSAR1ProductType,
) -> meta.RasterInfo:
    """Creating a RasterInfo Arepytools metadata element from xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        Image_Generation_Parameters metadata xml node
    image_attributes_node : etree._Element
        Image_Attributes metadata xml node
    product_type : NovaSAR1ProductTypes
        product type

    Returns
    -------
    RasterInfo
        RasterInfo metadata object
    """
    # lines
    lines = int(image_attributes_node.find("NumberOfLinesInImage").text)
    lines_start = PreciseDateTime.from_utc_string(
        image_generation_parameters_node.find("ZeroDopplerTimeFirstLine").text
    )
    lines_end = PreciseDateTime.from_utc_string(image_generation_parameters_node.find("ZeroDopplerTimeLastLine").text)
    lines_step = (lines_end - lines_start) / (lines - 1)
    lines_start_unit = "Utc"
    lines_step_unit = "s"

    # samples
    samples = int(image_attributes_node.find("NumberOfSamplesPerLine").text)
    if product_type == NovaSAR1ProductType.SLC or product_type == NovaSAR1ProductType.SRD:
        # slant range
        samples_start = float(image_generation_parameters_node.find("SWST").text)
        samples_start_unit = "s"
        samples_step = float(image_attributes_node.find("SampledPixelSpacing").text) / (LIGHT_SPEED / 2)
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"
    else:
        # ground range
        samples_start = 0
        samples_start_unit = "m"
        samples_step = float(image_attributes_node.find("SampledPixelSpacing").text)
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
    source_attributes_node: etree._Element, prod_type: NovaSAR1ProductType
) -> meta.DataSetInfo:
    """Creating a DataSetInfo Arepytools metadata element from safe xml nodes.

    Parameters
    ----------
    source_attributes_node : etree._Element
        Source_Attributes metadata xml node
    prod_type : NovaSAR1ProductTypes
        product type

    Returns
    -------
    DataSetInfo
        DataSetInfo metadata object
    """
    sensor_name = source_attributes_node.find("Satellite").text
    fc_hz = float(source_attributes_node.find("RadarCentreFrequency").text)
    acq_mode = get_acquisition_mode_from_product_type(prod_type=prod_type)

    if prod_type in (NovaSAR1ProductType.SLC, NovaSAR1ProductType.SRD):
        projection = "SLANT RANGE"
    else:
        projection = "GROUND RANGE"
    if prod_type == NovaSAR1ProductType.SLC:
        image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
    else:
        image_type = "MULTILOOK"

    dataset_info = meta.DataSetInfo(acquisition_mode_i=acq_mode.value, fc_hz_i=fc_hz)
    dataset_info.sensor_name = sensor_name
    dataset_info.image_type = image_type
    dataset_info.projection = projection
    dataset_info.side_looking = meta.ESideLooking(source_attributes_node.find("AntennaPointing").text.upper())

    return dataset_info


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
        swst_changes_values_i=[
            float(image_generation_parameters_node.find("SWST").text)
        ],  # for multiple subswaths, only the first is taken
        swl_changes_number_i=1,
        swl_changes_azimuth_times_i=[0],
        swl_changes_values_i=[raster_info.samples_step * (raster_info.samples - 1)],
    )


def doppler_centroid_poly_from_metadata_node(
    image_generation_parameters_node: etree._Element, raster_info: meta.RasterInfo
) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler centroid polynomial wrapper from safe xml node.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        Image_Generation_Parameters metadata xml node
    raster_info : meta.RasterInfo
        product raster info

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerCentroidVector metadata object
    """

    coeff_raw = [float(c) for c in image_generation_parameters_node.find("DopplerCentroid").text.split()]
    coefficients = [
        coeff_raw[0],
        coeff_raw[1] / raster_info.samples_step,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    coefficients_max_num = np.amin(np.array([len(coefficients) - 2, len(coeff_raw)]))
    for idx in range(2, coefficients_max_num):
        coefficients[idx + 2] = coeff_raw[idx] / (raster_info.samples_step**idx)

    doppler_centroid = meta.DopplerCentroid(
        i_ref_az=raster_info.lines_start, i_ref_rg=raster_info.samples_start, i_coefficients=coefficients
    )

    return create_sorted_poly_list(poly2d_vector=meta.DopplerCentroidVector([doppler_centroid]))


def pulse_info_from_metadata_nodes(
    image_generation_parameters_node: etree._Element, source_attributes_node: etree._Element, samples_step: float
) -> meta.Pulse:
    """Creating a Pulse Arepytools dataclass from xml nodes.

    Parameters
    ----------
    image_generation_parameters_node : etree._Element
        Image_Generation_Parameters metadata xml node
    source_attributes_node : etree._Element
        Source_Attributes metadata xml node
    samples_step : float
        raster info samples step

    Returns
    -------
    meta.Pulse
        Pulse info dataclass

    Raises
    ------
    RuntimeError
        unsupported chirp direction
    """

    chirp_direction = image_generation_parameters_node.find("ChirpDirection").text
    pulse_bandwidth = float(
        source_attributes_node.find("ChirpBandwidth").text
    )  # for multiple subswaths, only the first is taken
    if chirp_direction == "UPCHIRP":
        pulse_direction = meta.EPulseDirection.up.value
        pulse_start_frequency = -pulse_bandwidth / 2
    elif chirp_direction == "DOWNCHIRP":
        pulse_direction = meta.EPulseDirection.down.value
        pulse_start_frequency = pulse_bandwidth / 2
    else:
        raise RuntimeError(f"Chirp direction {chirp_direction} not recognized, expected: UPCHIRP or DOWNCHIRP")

    return meta.Pulse(
        i_pulse_length=float(
            source_attributes_node.find("PulseLength").text
        ),  # for multiple subswaths, only the first is taken
        i_bandwidth=pulse_bandwidth,
        i_pulse_sampling_rate=1 / samples_step,
        i_pulse_energy=1,
        i_pulse_start_frequency=pulse_start_frequency,
        i_pulse_start_phase=0,
        i_pulse_direction=pulse_direction,
    )


def general_sar_orbit_from_novasar1_state_vectors(state_vectors: NovaSAR1StateVectors) -> GeneralSarOrbit:
    """Creating a GeneralSarOrbit from product state vectors.

    Parameters
    ----------
    state_vectors : NovaSAR1StateVectors
        state vectors of NovaSAR-1 product

    Returns
    -------
    GeneralSarOrbit
        General SAR Orbit from State Vectors
    """
    return GeneralSarOrbit(time_axis=state_vectors.time_axis, state_vectors=state_vectors.positions.ravel())


class InvalidNovaSAR1Product(RuntimeError):
    """Invalid NovaSAR-1 product"""


@dataclass
class NovaSAR1SwathInfo:
    """NovaSAR-1 swath info"""

    rank: int
    azimuth_steering_rate_poly: tuple[float, float, float]
    prf: float

    @staticmethod
    def from_metadata_nodes(
        source_attributes_node: etree._Element, acq_mode: NovaSAR1AcquisitionModes
    ) -> NovaSAR1SwathInfo:
        """Generating NovaSAR1SwathInfo object directly from metadata xml nodes.

        Parameters
        ----------
        source_attributes_node : etree._Element
            Source_Attributes xml node
        acq_mode : etree._Element
            product acquisition mode

        Returns
        -------
        NovaSAR1SwathInfo
            swath info dataclass
        """
        rank = 0
        acquisition_prf = 0
        if acq_mode == NovaSAR1AcquisitionModes.STRIPMAP:
            rank = int(source_attributes_node.find("Rank").text)
            acquisition_prf = float(source_attributes_node.find("PulseRepetitionFrequency").text)

        return NovaSAR1SwathInfo(rank=rank, azimuth_steering_rate_poly=(0, 0, 0), prf=acquisition_prf)


@dataclass
class NovaSAR1BurstInfo:
    """NovaSAR-1 swath burst info"""

    num: int  # number of bursts in this swath
    lines_per_burst: int  # number of azimuth lines within each burst
    samples_per_burst: int  # number of range samples within each burst
    azimuth_start_times: np.ndarray  # zero doppler azimuth time of the first line of this burst
    range_start_times: np.ndarray  # zero doppler range time of the first sample of this burst


@dataclass
class NovaSAR1StateVectors:
    """NovaSAR-1 orbit's state vectors"""

    num: int  # attitude data numerosity
    frame: NovaSAR1ReferenceFrameType  # reference frame of the attitude data, always Zero Doppler
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
        pos_x = np.array([float(p.text) for p in orbit_data_node.findall("./StateVector/xPosition")])
        pos_y = np.array([float(p.text) for p in orbit_data_node.findall("./StateVector/yPosition")])
        pos_z = np.array([float(p.text) for p in orbit_data_node.findall("./StateVector/zPosition")])
        positions = np.stack([pos_x, pos_y, pos_z], axis=1)

        vel_x = np.array([float(p.text) for p in orbit_data_node.findall("./StateVector/xVelocity")])
        vel_y = np.array([float(p.text) for p in orbit_data_node.findall("./StateVector/yVelocity")])
        vel_z = np.array([float(p.text) for p in orbit_data_node.findall("./StateVector/zVelocity")])
        velocities = np.stack([vel_x, vel_y, vel_z], axis=1)

        time_axis = np.array(
            [PreciseDateTime.from_utc_string(p.text) for p in orbit_data_node.findall("./StateVector/Time")]
        )

        return positions, velocities, time_axis

    @staticmethod
    def from_metadata_node(orbit_data_node: etree._Element) -> NovaSAR1StateVectors:
        """Generating a NovaSAR1StateVectors object directly from metadata xml node.

        Parameters
        ----------
        orbit_data_node : etree._Element
            OrbitData xml node

        Returns
        -------
        NovaSAR1StateVectors
            orbit's state vectors dataclass
        """

        positions, velocities, times = NovaSAR1StateVectors._unpack_state_vector_from_orbit_node(
            orbit_data_node=orbit_data_node
        )
        rel_times = (times - times[0]).astype(float)
        numerosity = int(orbit_data_node.find("NumberOfStateVectorSets").text)
        assert positions.shape == velocities.shape == (numerosity, 3)

        # interpolating state vectors using a constant time step axis (original time axis step is not constant)
        # extrapolating also one time step before and one time step after the original time axis to extend it by
        # two points
        mean_delta_time = np.diff(times).mean()
        poly_order = 6 if numerosity > 6 else (numerosity - 1)
        extended_time_relative_axis = np.arange(-1, numerosity + 1, 1) * mean_delta_time
        extended_numerosity = extended_time_relative_axis.size  # this is just default numerosity + 2

        # fitting a poly_order polynomial to the actual positions values in the metadata file
        pos_x_poly = Polynomial.fit(rel_times, positions[:, 0], poly_order)
        pos_y_poly = Polynomial.fit(rel_times, positions[:, 1], poly_order)
        pos_z_poly = Polynomial.fit(rel_times, positions[:, 2], poly_order)

        # evaluating interpolated new positions
        pos_x_interp = pos_x_poly(extended_time_relative_axis)
        pos_y_interp = pos_y_poly(extended_time_relative_axis)
        pos_z_interp = pos_z_poly(extended_time_relative_axis)
        positions_interp = np.stack([pos_x_interp, pos_y_interp, pos_z_interp], axis=1)

        # evaluating interpolated new velocities
        vel_x_interp = pos_x_poly.deriv()(extended_time_relative_axis)
        vel_y_interp = pos_y_poly.deriv()(extended_time_relative_axis)
        vel_z_interp = pos_z_poly.deriv()(extended_time_relative_axis)
        velocities_interp = np.stack([vel_x_interp, vel_y_interp, vel_z_interp], axis=1)

        return NovaSAR1StateVectors(
            num=extended_numerosity,
            frame=NovaSAR1ReferenceFrameType.ZERO_DOPPLER,
            orbit_direction=OrbitDirection(orbit_data_node.find("Pass_Direction").text.lower()),
            positions=positions_interp,
            velocities=velocities_interp,
            time_axis=extended_time_relative_axis + times[0],
            time_step=mean_delta_time,
        )


@dataclass
class NovaSAR1Attitude:
    """NovaSAR-1 sensor's attitude"""

    num: int  # attitude data numerosity (same as interpolated orbit)
    frame: NovaSAR1ReferenceFrameType  # reference frame of the attitude data, always Zero Doppler
    yaw: np.ndarray  # platform yaw
    pitch: np.ndarray  # platform pitch
    roll: np.ndarray  # platform roll
    time_axis: np.ndarray  # PreciseDateTime axis to which attitude data applies
    time_step: float  # time axis step

    @staticmethod
    def from_metadata_node(orbit_data_node: etree._Element, image_gen_params_node: etree._Element) -> NovaSAR1Attitude:
        """Generating NovaSAR1Attitude object directly from metadata xml node.

        Parameters
        ----------
        orbit_data_node : etree._Element
            OrbitData xml node
        image_gen_params_node : etree._Element
            Image_Generation_Parameters metadata xml node

        Returns
        -------
        NovaSAR1Attitude
            sensor's attitude dataclass
        """

        # using the same extended axis defined for orbit interpolation, aka 2 points more than those annotated,
        # in particular one point before and one after the annotated ones (minimal extrapolation)
        numerosity = int(orbit_data_node.find("NumberOfStateVectorSets").text)
        extended_numerosity = numerosity + 2

        yaw = float(orbit_data_node.find("PlatformYaw").text)
        yaw_rate = float(orbit_data_node.find("PlatformYawRate").text)
        pitch = float(orbit_data_node.find("PlatformPitch").text)
        pitch_rate = float(orbit_data_node.find("PlatformPitchRate").text)
        roll = float(orbit_data_node.find("PlatformRoll").text)
        roll_rate = float(orbit_data_node.find("PlatformRollRate").text)

        # creating time axis
        state_vectors_time_axis = np.array(
            [PreciseDateTime.from_utc_string(p.text) for p in orbit_data_node.findall("./StateVector/Time")]
        )
        mean_delta_time_sv = np.diff(state_vectors_time_axis).mean()

        # recovering doppler lines times
        start_time_extended_sv = state_vectors_time_axis[0] - mean_delta_time_sv
        first_zero_doppler_line = PreciseDateTime.from_utc_string(
            image_gen_params_node.find("ZeroDopplerTimeFirstLine").text
        )
        last_zero_doppler_line = PreciseDateTime.from_utc_string(
            image_gen_params_node.find("ZeroDopplerTimeLastLine").text
        )
        half_zero_doppler_line = first_zero_doppler_line + (last_zero_doppler_line - first_zero_doppler_line) / 2
        t0 = half_zero_doppler_line - start_time_extended_sv

        # creating extended time axis
        extended_relative_time_axis = np.arange(0, extended_numerosity, 1) * mean_delta_time_sv
        extended_relative_time_axis_start = state_vectors_time_axis[0] - mean_delta_time_sv

        # creating yaw, pitch and roll vectors
        yaw_array = yaw_rate * (extended_relative_time_axis - t0) + yaw
        pitch_array = pitch_rate * (extended_relative_time_axis - t0) + pitch
        roll_array = roll_rate * (extended_relative_time_axis - t0) + roll

        return NovaSAR1Attitude(
            num=extended_numerosity,
            frame=NovaSAR1ReferenceFrameType.ZERO_DOPPLER,
            yaw=yaw_array,
            pitch=pitch_array,
            roll=roll_array,
            time_axis=extended_relative_time_axis + extended_relative_time_axis_start,
            time_step=mean_delta_time_sv,
        )


@dataclass
class NovaSAR1IncidenceAnglePolynomial:
    """NovaSAR-1 Incidence Angle Polynomial"""

    incidence_angles: list[ConversionPolynomial] | None = None  # polynomial is f: pixels -> deg
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
    ) -> NovaSAR1IncidenceAnglePolynomial:
        """Generating NovaSAR1IncidenceAnglePolynomial object directly from metadata xml node.

        About coefficients annotated in metadata:

        Polynomial coefficients for incidence angle at pixel position. Fixed along all image slices.
        Defined with respect to (angles in deg)/(pixel no.)^n where first pixel in line is 0.

        Values output in order A0, A1, ...,  An, in order of increasing degree.
        Polynomial to be evaluated is:  A_0+ A_1 x + ... + A_n x^n
        where x is pixel number in the line, starting from 0.

        Evaluated polynomial gives Incidence Angle in deg.

        Parameters
        ----------
        image_generation_parameters_node : etree._Element
            Image_Generation_Parameters metadata xml node
        raster_info : meta.RasterInfo
            product raster info

        Returns
        -------
        NovaSAR1IncidenceAnglePolynomial
            polynomial for incidence angle computation
        """

        # recovering coefficients and applying conversion factor meters to seconds
        incidence_angle_coeff = [float(c) for c in image_generation_parameters_node.find("IncAngleCoeffs").text.split()]
        incidence_angle_poly = Polynomial(incidence_angle_coeff)

        incidence_angle_poly_list = [
            ConversionPolynomial(
                azimuth_reference_time=raster_info.lines_start,
                origin=0,
                polynomial=incidence_angle_poly,
            )
        ]

        return NovaSAR1IncidenceAnglePolynomial(
            azimuth_reference_times=raster_info.lines_start,
            incidence_angles=incidence_angle_poly_list,
        )

    def evaluate_incidence_angle(self, azimuth_time: PreciseDateTime, range_pixels: Union[int, npt.ArrayLike]) -> float:
        """Compute incidence angle at given time.

        Parameters
        ----------
        azimuth_time : PreciseDateTime
            azimuth time to select the proper polynomial to be used for conversion
        range_pixels :  Union[int, npt.ArrayLike]
            range pixel index

        Returns
        -------
        float
            incidence angle in degrees
        """
        poly_index = self._detect_right_polynomial_index(azimuth_time=azimuth_time)
        poly = self.incidence_angles[poly_index]
        return poly.polynomial(range_pixels - poly.origin)


@dataclass
class NovaSAR1CoordinateConversions:
    """NovaSAR-1 coordinate conversion"""

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
    ) -> NovaSAR1CoordinateConversions:
        """Generating NovaSAR1CoordinateConversions object directly from metadata xml node.

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
            Image_Generation_Parameters metadata xml node
        raster_info : meta.RasterInfo
            product raster info

        Returns
        -------
        NovaSAR1CoordinateConversions
            polynomial for coordinate conversion dataclass
        """

        if image_generation_parameters_node.find("GroundToSlantRangeCoefficients") is None:
            return NovaSAR1CoordinateConversions()

        # recovering coefficients and applying conversion factor meters to seconds
        m2s_conversion_factor = 1 / (LIGHT_SPEED / 2)
        coeff_raw = [
            m2s_conversion_factor * float(c)
            for c in image_generation_parameters_node.find("GroundToSlantRangeCoefficients").text.split()
        ]
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

        return NovaSAR1CoordinateConversions(
            azimuth_reference_times=raster_info.lines_start,
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
class NovaSAR1GeneralChannelInfo:
    """NovaSAR-1 general channel info representation dataclass"""

    product_name: str
    product_id: int
    channel_id: str
    swath: str
    satellite: str
    acq_start_time: PreciseDateTime
    product_type: NovaSAR1ProductType
    acquisition_mode: NovaSAR1AcquisitionModes
    polarization: SARPolarization
    projection: SARProjection
    orbit_direction: OrbitDirection

    @staticmethod
    def from_metadata_node(
        product_node: etree._Element,
        source_attributes_node: etree._Element,
        orbit_data_node: etree._Element,
        prod_type: NovaSAR1ProductType,
        channel_id: str,
    ) -> NovaSAR1GeneralChannelInfo:
        """Generating S1GeneralChannelInfo object directly from metadata xml nodes.

        Parameters
        ----------
        product_node : etree._Element
            Product metadata xml node
        source_attributes_node : etree._Element
            Source_Attributes metadata xml node
        orbit_data_node : etree._Element
            OrbitData metadata xml node
        prod_type : NovaSAR1ProductTypes
            product type
        channel_id : str
            channel id

        Returns
        -------
        NovaSAR1GeneralChannelInfo
            general channel info dataclass
        """

        pol_from_channel = channel_id.split("_")[-1]
        projection = get_projection_from_product_type(prod_type=prod_type)
        acq_mode = get_acquisition_mode_from_product_type(prod_type=prod_type)
        swath = source_attributes_node.find("Rank").get("Subswath")
        if acq_mode != NovaSAR1AcquisitionModes.STRIPMAP:
            swath = swath[0]

        return NovaSAR1GeneralChannelInfo(
            product_name=product_node.find("ProductName").text,
            product_id=int(product_node.find("Product_ID").text),
            channel_id=channel_id,
            swath=swath,
            satellite=source_attributes_node.find("Satellite").text,
            acq_start_time=PreciseDateTime.from_utc_string(source_attributes_node.find("RawDataStartTime").text),
            product_type=prod_type,
            acquisition_mode=acq_mode,
            projection=projection,
            polarization=SARPolarization.HH if pol_from_channel == "hh" else SARPolarization.VV,
            orbit_direction=OrbitDirection(orbit_data_node.find("Pass_Direction").text.lower()),
        )


@dataclass
class NovaSAR1ChannelMetadata:
    """NovaSAR-1 channel metadata xml file wrapper"""

    general_info: NovaSAR1GeneralChannelInfo
    general_sar_orbit: GeneralSarOrbit
    attitude: NovaSAR1Attitude
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    lines_time_ordering: NovaSAR1TimeOrdering
    samples_time_ordering: NovaSAR1TimeOrdering
    burst_info: NovaSAR1BurstInfo
    raster_info: meta.RasterInfo
    dataset_info: meta.DataSetInfo
    swath_info: NovaSAR1SwathInfo
    sampling_constants: SARSamplingFrequencies
    acquisition_timeline: meta.AcquisitionTimeLine
    doppler_centroid_poly: SortedPolyList
    incidence_angles_poly: NovaSAR1IncidenceAnglePolynomial
    pulse: meta.Pulse
    coordinate_conversions: NovaSAR1CoordinateConversions
    state_vectors: NovaSAR1StateVectors


class NovaSAR1Product:
    """NovaSAR-1 product object"""

    def __init__(self, path: Union[str, Path]) -> None:
        """NovaSAR Product init from directory path.

        Parameters
        ----------
        path : Union[str, Path]
            path to NovaSAR product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name

        # extracting full path to raster and metadata files
        self._data_paths = [
            f for f in self._product_path.iterdir() if f.name.startswith("image") and f.name.endswith(_IMAGE_FORMAT)
        ]
        self._metadata_path = self._product_path.joinpath("metadata").with_suffix(_METADATA_EXTENSION)
        self._channels_number = len(self._data_paths)

        # acquisition time and channels list
        (
            self._acq_time,
            self._product_type,
            self._channel_list_by_swath_id,
            self._footprint,
        ) = get_basic_info_from_metadata(metadata_path=self._metadata_path)

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def data_list(self) -> list[Path]:
        """Returning the list of raster data files of NovaSAR product"""
        return self._data_paths

    @property
    def metadata_file(self) -> Path:
        """Returning the product metadata file path of NovaSAR product"""
        return self._metadata_path

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of NovaSAR product"""
        return self._channels_number

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels in terms of SwathID (swath-polarization)"""
        return self._channel_list_by_swath_id

    @property
    def product_type(self) -> NovaSAR1ProductType:
        """Returning the product type"""
        return self._product_type

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_raster_files_from_channel_name(self, channel_name: str) -> Path:
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
        return [r for r in self.data_list if channel_name.split("_")[-1] in r.name.lower()][0]


def is_novasar_1_product(product: Union[str, Path]) -> bool:
    """Check if input path corresponds to a valid NovaSAR-1 product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - metadata file exists
        - basic info from metadata file can be read

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

    if product.joinpath("metadata.xml").is_file():
        try:
            get_basic_info_from_metadata(product.joinpath("metadata.xml"))
        except Exception:
            return False
    else:
        return False

    return True
