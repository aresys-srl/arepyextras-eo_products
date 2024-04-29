# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
ICEYE reader support module
---------------------------
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import arepytools.io.metadata as meta
import h5py
import numpy as np
import numpy.typing as npt
from arepytools.constants import LIGHT_SPEED
from arepytools.geometry.generalsarorbit import GeneralSarOrbit
from arepytools.math.genericpoly import SortedPolyList, create_sorted_poly_list
from arepytools.timing.precisedatetime import PreciseDateTime
from lxml import etree
from numpy.polynomial import Polynomial

from arepyextras.eo_products.common.utilities import (
    ConversionPolynomial,
    OrbitDirection,
    SARPolarization,
    SARProjection,
    SARRadiometricQuantity,
    SARSamplingFrequencies,
)

_GRD_IMAGE_FORMAT = ".tif"
_GRD_METADATA_EXTENSION = ".xml"
_SLC_DATA_EXTENSION = ".h5"


class InvalidICEYEProduct(RuntimeError):
    """Invalid ICEYE product"""


class ICEYEProductLevel(Enum):
    """ICEYE L1 product level"""

    SLC = "SLC"  # Slant Range, Single Look Complex (SLC, lvl 1)
    GRD = "GRD"  # Ground Range Multi Look Detected (GRD, lvl 1, phase lost)


class ICEYEAcquisitionMode(Enum):
    """ICEYE L1 acquisition modes"""

    SPOTLIGHT = "spotlight"
    STRIPMAP = "stripmap"
    TOPSAR = "scan"


class ICEYEStateVectorsReferenceSystem(Enum):
    """Orbit's state vectors reference system"""

    WGS84 = "WGS84"


class ICEYEOrbitProcessingLevel(Enum):
    """Orbit's processing level type"""

    PREDICTED = "predicted"
    RAPID = "rapid"
    PRECISE = "precise"
    SCIENTIFIC = "scientific"


def raster_info_from_metadata(root: etree._Element | h5py.File) -> meta.RasterInfo:
    """Creating a RasterInfo Arepytools metadata object from metadata file.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object

    Returns
    -------
    meta.RasterInfo
        RasterInfo metadata object
    """

    if isinstance(root, h5py.File):
        # assuming SLC only
        filename = root["product_file"][()].decode()

        # lines
        lines = int(root["number_of_azimuth_samples"][()])
        lines_start = PreciseDateTime.fromisoformat(root["zerodoppler_start_utc"][()].decode())
        lines_step = root["azimuth_time_interval"][()]
        lines_start_unit = "Utc"
        lines_step_unit = "s"

        # samples
        samples = int(root["number_of_range_samples"][()])
        samples_start = root["first_pixel_time"][()]
        samples_step = 1 / root["range_sampling_rate"][()]
        samples_start_unit = "s"
        samples_step_unit = "s"
        celltype = "FLOAT_COMPLEX"

    else:
        # assuming GRD only
        filename = root.find("product_file").text

        # lines
        lines = int(root.find("number_of_azimuth_samples").text)
        lines_start = PreciseDateTime.fromisoformat(root.find("zerodoppler_start_utc").text)
        lines_step = float(root.find("azimuth_time_interval").text)
        lines_start_unit = "Utc"
        lines_step_unit = "s"

        # samples
        samples = int(root.find("number_of_range_samples").text)
        samples_start = 0
        samples_step = float(root.find("range_spacing").text)
        samples_start_unit = "m"
        samples_step_unit = "m"
        celltype = "FLOAT32"

    # assembling RasterInfo
    raster_info = meta.RasterInfo(
        lines=lines,
        samples=samples,
        filename=filename,
        celltype=celltype,
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


def dataset_info_from_metadata(root: etree._Element | h5py.File) -> meta.DataSetInfo:
    """Creating a DataSetInfo Arepytools metadata object from metadata file.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object

    Returns
    -------
    DataSetInfo
        DataSetInfo metadata object
    """
    if isinstance(root, h5py.File):
        # assuming SLC only
        sensor_name = root["satellite_name"][()].decode()
        fc_hz = root["carrier_frequency"][()]
        acquisition_mode = ICEYEAcquisitionMode(root["acquisition_mode"][()].decode().lower())
        look_side = meta.ESideLooking(root["look_side"][()].decode().upper())
        projection = "SLANT RANGE"
        image_type = "AZIMUTH FOCUSED RANGE COMPENSATED"
    else:
        # assuming GRD only
        sensor_name = root.find("satellite_name").text
        fc_hz = float(root.find("carrier_frequency").text)
        acquisition_mode = ICEYEAcquisitionMode(root.find("acquisition_mode").text.lower())
        look_side = meta.ESideLooking(root.find("look_side").text.upper())
        projection = "GROUND RANGE"
        image_type = "MULTILOOK"

    dataset_info = meta.DataSetInfo(acquisition_mode_i=acquisition_mode.value, fc_hz_i=fc_hz)
    dataset_info.sensor_name = sensor_name
    dataset_info.image_type = image_type
    dataset_info.projection = projection
    dataset_info.side_looking = look_side

    return dataset_info


def acquisition_timeline_from_metadata(root: etree._Element | h5py.File) -> meta.AcquisitionTimeLine:
    """Creating a AcquisitionTimeLine Arepytools metadata object from metadata file.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object

    Returns
    -------
    AcquisitionTimeLine
        AcquisitionTimeLine metadata object
    """
    if isinstance(root, h5py.File):
        swst_changes_val = root["slant_range_to_first_pixel"][()] / (LIGHT_SPEED / 2)
        swl_changes_val = root["number_of_range_samples"][()] / root["range_sampling_rate"][()]
    else:
        swst_changes_val = float(root.find("slant_range_to_first_pixel").text) / (LIGHT_SPEED / 2)
        swl_changes_val = float(root.find("number_of_range_samples").text) / float(
            root.find("range_sampling_rate").text
        )

    return meta.AcquisitionTimeLine(
        swst_changes_number_i=1,
        swst_changes_azimuth_times_i=[0],
        swst_changes_values_i=[swst_changes_val],
        swl_changes_number_i=1,
        swl_changes_azimuth_times_i=[0],
        swl_changes_values_i=[swl_changes_val],
    )


def doppler_centroid_poly_from_metadata(
    root: etree._Element | h5py.File, raster_info: meta.RasterInfo
) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler centroid polynomial wrapper from metadata.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object
    raster_info : meta.RasterInfo
        channel raster info

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerCentroidVector metadata object
    """

    doppler_poly = []
    if isinstance(root, h5py.File):
        # SLC case
        acquisition_mode = ICEYEAcquisitionMode(root["acquisition_mode"][()].decode().lower())
        coeff_raw = root["dc_estimate_coeffs"][()]
        time_axis = np.array(
            [PreciseDateTime.fromisoformat(t.item().decode()) for t in root["dc_estimate_time_utc"][()]]
        )
        origin = np.repeat(
            raster_info.samples_start + (raster_info.samples_step * raster_info.samples) / 2, time_axis.size
        )  # mid range time
        if acquisition_mode != ICEYEAcquisitionMode.STRIPMAP:
            # Topsar and Spotlight doppler polynomial is forced to be a single average polynomial for the whole acquisition
            # because the annotated one is the antenna steering
            coeff_raw = np.atleast_2d(coeff_raw.mean(axis=0))
            time_axis = np.array([PreciseDateTime.fromisoformat(root["zerodoppler_start_utc"][()].decode())])
            origin = np.array([origin[0]])
    else:
        # GRD case
        acquisition_mode = ICEYEAcquisitionMode(root.find("acquisition_mode").text.lower())
        time_axis = [
            PreciseDateTime.fromisoformat(t.text)
            for t in root.findall("Doppler_Centroid_Coefficients/dc_coefficients_list/zero_doppler_time")
        ]
        origin = np.array(
            [
                float(t.text)
                for t in root.findall("Doppler_Centroid_Coefficients/dc_coefficients_list/reference_pixel_time")
            ]
        )
        coeff_raw = np.array(
            [
                float(t.text)
                for t in root.findall("Doppler_Centroid_Coefficients/dc_coefficients_list/coefficient/value")
            ]
        ).reshape(-1, 4)
        if acquisition_mode != ICEYEAcquisitionMode.STRIPMAP:
            # Topsar and Spotlight doppler polynomial is forced to be a single average polynomial for the whole acquisition
            # because the annotated one is the antenna steering
            coeff_raw = np.atleast_2d(coeff_raw.mean(axis=0))
            time_axis = np.array([PreciseDateTime.fromisoformat(root.find("zerodoppler_start_utc").text)])
            origin = np.array([origin[0]])

    for coeff, time, t_ref in zip(coeff_raw, time_axis, origin):
        doppler_poly.append(
            meta.DopplerCentroid(
                i_ref_az=time,
                i_ref_rg=t_ref,
                i_coefficients=[coeff[0], coeff[1], 0, 0, coeff[2], coeff[3], 0, 0, 0, 0, 0],
            )
        )

    return create_sorted_poly_list(meta.DopplerCentroidVector(i_poly2d=doppler_poly))


def doppler_rate_poly_from_metadata(root: etree._Element | h5py.File, raster_info: meta.RasterInfo) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools doppler rate vector polynomial wrapper from metadata.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object
    raster_info : meta.RasterInfo
        channel raster info

    Returns
    -------
    SortedPolyList
        SortedPolyList wrapper on DopplerRateVector metadata object
    """
    doppler_rate_poly = []
    if isinstance(root, h5py.File):
        ref_rng_time = (
            raster_info.samples_start + (raster_info.samples_step * raster_info.samples) / 2
        )  # mid range time
        ref_az_time = [PreciseDateTime.fromisoformat(t.item().decode()) for t in root["dc_estimate_time_utc"][()]][0]
        coeff_raw = root["doppler_rate_coeffs"][()]
    else:
        ref_rng_time = float(root.find("Doppler_Rate/reference_pixel_time").text)
        coeff_raw = [float(c.text) for c in root.findall("Doppler_Rate/coefficient/value")]
        ref_az_time = [
            PreciseDateTime.fromisoformat(t.text)
            for t in root.findall("Doppler_Centroid_Coefficients/dc_coefficients_list/zero_doppler_time")
        ][0]

    doppler_rate_poly = [
        meta.DopplerRate(
            i_ref_az=ref_az_time,
            i_ref_rg=ref_rng_time,
            i_coefficients=[coeff_raw[0], coeff_raw[1], 0, 0, coeff_raw[2], coeff_raw[3], 0, 0, 0, 0, 0],
        )
    ]

    return create_sorted_poly_list(meta.DopplerRateVector(i_poly2d=doppler_rate_poly))


def sampling_constants_from_metadata(root: etree._Element | h5py.File) -> SARSamplingFrequencies:
    """Creating a SARSamplingFrequencies metadata object from metadata file.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    if isinstance(root, h5py.File):
        azimuth_freq_hz = 1 / root["azimuth_time_interval"][()]
        azimuth_bandwidth_freq_hz = root["total_processed_bandwidth_azimuth"][()]
        range_freq_hz = root["range_sampling_rate"][()]
        range_bandwidth_freq_hz = root["chirp_bandwidth"][()]
    else:
        azimuth_freq_hz = 1 / float(root.find("azimuth_time_interval").text)
        azimuth_bandwidth_freq_hz = float(root.find("total_processed_bandwidth_azimuth").text)
        range_freq_hz = float(root.find("range_sampling_rate").text)
        range_bandwidth_freq_hz = float(root.find("chirp_bandwidth").text)

    return SARSamplingFrequencies(
        azimuth_freq_hz=azimuth_freq_hz,
        azimuth_bandwidth_freq_hz=azimuth_bandwidth_freq_hz,
        range_freq_hz=range_freq_hz,
        range_bandwidth_freq_hz=range_bandwidth_freq_hz,
    )


def pulse_info_from_metadata(root: etree._Element | h5py.File) -> meta.Pulse:
    """Creating a Pulse Arepytools metadata object from metadata.

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object

    Returns
    -------
    meta.Pulse
        Pulse metadata object
    """
    if isinstance(root, h5py.File):
        # SLC case
        pulse_length = root["chirp_duration"][()]
        chirp_bandwidth = root["chirp_bandwidth"][()]
        sampling_rate = root["range_sampling_rate"][()]
    else:
        # GRD case
        pulse_length = float(root.find("chirp_duration").text)
        chirp_bandwidth = float(root.find("chirp_bandwidth").text)
        sampling_rate = float(root.find("range_sampling_rate").text)

    return meta.Pulse(
        i_pulse_length=pulse_length,
        i_bandwidth=chirp_bandwidth,
        i_pulse_sampling_rate=sampling_rate,
        i_pulse_energy=1,
        i_pulse_start_frequency=-chirp_bandwidth / 2,
        i_pulse_start_phase=0,
        i_pulse_direction=meta.EPulseDirection.up.value,
    )


def calibration_factor_and_radiometric_quantity_from_metadata(
    root: etree._Element | h5py.File,
) -> tuple[float, SARRadiometricQuantity]:
    """Image calibration factor and radiometric quantity from metadata file.

    SLC case: binary image * calibration factor =  beta nought

    GRD case: binary image (already sigma nought) * calibration factor = sigma nought

    Parameters
    ----------
    root : etree._Element | h5py.File
        metadata root object

    Returns
    -------
    tuple[float, SARRadiometricQuantity]
        image calibration factor,
        radiometric quantity
    """
    if isinstance(root, h5py.File):
        # SLC case
        return np.sqrt(root["calibration_factor"])[()], SARRadiometricQuantity.BETA_NOUGHT

    # GRD case
    return np.sqrt(float(root.find("calibration_factor").text)), SARRadiometricQuantity.SIGMA_NOUGHT


def general_sar_orbit_from_iceye_state_vectors(state_vectors: ICEYEStateVectors) -> GeneralSarOrbit:
    """Creating a GeneralSarOrbit from product state vectors.

    Parameters
    ----------
    state_vectors : ICEYEStateVectors
        state vectors of ICEYE product

    Returns
    -------
    GeneralSarOrbit
        General SAR Orbit from State Vectors
    """
    return GeneralSarOrbit(time_axis=state_vectors.time_axis, state_vectors=state_vectors.positions.ravel())


@dataclass
class ICEYECoordinateConversions:
    """ICEYE coordinate conversion"""

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
    def from_metadata(root: etree._Element | h5py.File) -> ICEYECoordinateConversions:
        """Generating ICEYECoordinateConversions object directly from metadata.

        Parameters
        ----------
        root : etree._Element | h5py.File
            metadata root object

        Returns
        -------
        ICEYECoordinateConversions
            polynomial for coordinate conversion dataclass
        """
        if isinstance(root, h5py.File):
            # SLC case, no poly
            return ICEYECoordinateConversions()

        # GRD case
        m2s_conversion_factor = 1 / (LIGHT_SPEED / 2)  # conversion factor meters to seconds
        az_ref_time = PreciseDateTime.fromisoformat(root.find("GRSR_Coefficients/zero_doppler_time").text)
        origin = float(root.find("GRSR_Coefficients/ground_range_origin").text)
        coefficients = [
            float(c.text) * m2s_conversion_factor for c in root.findall("GRSR_Coefficients/coefficient/value")
        ]
        ground_to_slant_poly = Polynomial(coefficients)

        # slant to ground poly is not given, so it must be evaluated by inverting the ground to slant poly
        samples = int(root.find("number_of_range_samples").text)
        samples_step = float(root.find("range_spacing").text)
        rng_axis = np.arange(0, (samples + 1) * samples_step, samples_step)
        ground_to_slant_poly_evaluated = ground_to_slant_poly(rng_axis)
        slant_to_ground_poly = Polynomial.fit(
            x=ground_to_slant_poly_evaluated, y=rng_axis, deg=ground_to_slant_poly.degree()
        )

        # assembling list of polynomials
        ground_to_slant_poly_list = [
            ConversionPolynomial(
                azimuth_reference_time=az_ref_time,
                origin=origin,
                polynomial=ground_to_slant_poly,
            )
        ]
        slant_to_ground_poly_list = [
            ConversionPolynomial(
                azimuth_reference_time=az_ref_time,
                origin=0,
                polynomial=slant_to_ground_poly,
            )
        ]

        return ICEYECoordinateConversions(
            azimuth_reference_times=np.array([az_ref_time]),
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
class ICEYEStateVectors:
    """ICEYE orbit's state vectors"""

    num: int  # attitude data numerosity
    frame: ICEYEStateVectorsReferenceSystem  # reference frame
    positions: np.ndarray  # platform position data with respect to the Earth-fixed reference frame
    velocities: np.ndarray  # platform velocity data with respect to the Earth-fixed reference frame
    time_axis: np.ndarray  # PreciseDateTime axis at which orbit state vectors apply
    time_step: float  # time axis step
    orbit_direction: OrbitDirection  # orbit direction: ascending or descending
    orbit_type: ICEYEOrbitProcessingLevel  # orbit processing level type

    @staticmethod
    def from_metadata(root: etree._Element | h5py.File) -> ICEYEStateVectors:
        """Generating a ICEYEStateVectors object directly from metadata file.

        Parameters
        ----------
        root : etree._Element | h5py.File
            metadata root object

        Returns
        -------
        ICEYEStateVectors
            orbit's state vectors dataclass
        """

        if isinstance(root, h5py.File):
            # SLC case
            numerosity = root["number_of_state_vectors"][()]
            orbit_direction = OrbitDirection(root["orbit_direction"][()].decode().lower())
            orbit_type = ICEYEOrbitProcessingLevel(root["orbit_processing_level"][()].decode().lower())
            frame = ICEYEStateVectorsReferenceSystem(root["geo_ref_system"][()].decode().upper())
            positions = np.stack([root["posX"][()], root["posY"][()], root["posZ"][()]], axis=1)
            velocities = np.stack([root["velX"][()], root["velY"][()], root["velZ"][()]], axis=1)
            time_axis = np.array(
                [PreciseDateTime.fromisoformat(t.item().decode()) for t in root["state_vector_time_utc"][()]]
            )
        else:
            # GRD case
            numerosity = int(root.find("Orbit_State_Vectors/count").text)
            orbit_direction = OrbitDirection(root.find("orbit_direction").text.lower())
            orbit_type = ICEYEOrbitProcessingLevel(root.find("orbit_processing_level").text.lower())
            frame = ICEYEStateVectorsReferenceSystem(root.find("geo_ref_system").text.upper())
            positions = np.stack(
                [
                    [float(p.text) for p in root.findall("Orbit_State_Vectors/orbit_vector/posX")],
                    [float(p.text) for p in root.findall("Orbit_State_Vectors/orbit_vector/posY")],
                    [float(p.text) for p in root.findall("Orbit_State_Vectors/orbit_vector/posZ")],
                ],
                axis=1,
            )
            velocities = np.stack(
                [
                    [float(p.text) for p in root.findall("Orbit_State_Vectors/orbit_vector/velX")],
                    [float(p.text) for p in root.findall("Orbit_State_Vectors/orbit_vector/velY")],
                    [float(p.text) for p in root.findall("Orbit_State_Vectors/orbit_vector/velZ")],
                ],
                axis=1,
            )
            time_axis = np.array(
                [PreciseDateTime.fromisoformat(t.text) for t in root.findall("Orbit_State_Vectors/orbit_vector/time")]
            )

        assert positions.shape[0] == velocities.shape[0] == time_axis.size == numerosity

        return ICEYEStateVectors(
            num=numerosity,
            frame=frame,
            positions=positions,
            velocities=velocities,
            time_axis=time_axis,
            time_step=time_axis[1] - time_axis[0],
            orbit_direction=orbit_direction,
            orbit_type=orbit_type,
        )


@dataclass
class ICEYEIncidenceAnglePolynomial:
    """ICEYE Incidence Angle Polynomial"""

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
    def from_metadata(root: etree._Element | h5py.File) -> ICEYEIncidenceAnglePolynomial:
        """Generating ICEYEIncidenceAnglePolynomial object directly from metadata xml node.

        Evaluated polynomial gives Incidence Angle in deg.

        Parameters
        ----------
        root : etree._Element | h5py.File
            metadata root object

        Returns
        -------
        ICEYEIncidenceAnglePolynomial
            polynomial for incidence angle computation
        """

        if isinstance(root, h5py.File):
            # SLC case
            return ICEYEIncidenceAnglePolynomial()
        else:
            # GRD case
            coeff = [float(c.text) for c in root.findall("Incidence_Angle_Coefficients/coefficient/value")]
            ref_time = PreciseDateTime.fromisoformat(root.find("Incidence_Angle_Coefficients/zero_doppler_time").text)
            origin = float(root.find("Incidence_Angle_Coefficients/ground_range_origin").text)

            incidence_angle_poly_list = [
                ConversionPolynomial(
                    azimuth_reference_time=ref_time,
                    origin=origin,
                    polynomial=Polynomial(coeff),
                )
            ]

            return ICEYEIncidenceAnglePolynomial(
                azimuth_reference_times=np.array([ref_time]),
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
class ICEYESwathInfo:
    """ICEYE swath info"""

    rank: int
    azimuth_steering_rate_poly: tuple[float, float, float]
    prf: float

    @staticmethod
    def from_metadata(root: etree._Element | h5py.File) -> ICEYESwathInfo:
        """Generating ICEYESwathInfo object directly from metadata.

        Parameters
        ----------
        root : etree._Element | h5py.File
            metadata root object

        Returns
        -------
        ICEYESwathInfo
            swath info dataclass
        """
        if isinstance(root, h5py.File):
            prf = root["acquisition_prf"][()]
            rank = np.floor(root["slant_range_to_first_pixel"][()] / (LIGHT_SPEED / 2) * prf)
        else:
            prf = float(root.find("acquisition_prf").text)
            rank = np.floor(float(root.find("slant_range_to_first_pixel").text) / (LIGHT_SPEED / 2) * prf)

        # TODO check for Spotlight case
        azimuth_steering_rate_poly = (0, 0, 0)

        return ICEYESwathInfo(rank=rank.astype(int), azimuth_steering_rate_poly=azimuth_steering_rate_poly, prf=prf)


@dataclass
class ICEYEBurstInfo:
    """ICEYE swath burst info"""

    num: int  # number of bursts in this swath
    lines_per_burst: int  # number of azimuth lines within each burst
    samples_per_burst: int  # number of range samples within each burst
    azimuth_start_times: np.ndarray  # zero doppler azimuth time of the first line of this burst
    range_start_times: np.ndarray  # zero doppler range time of the first sample of this burst


@dataclass
class ICEYEGeneralChannelInfo:
    """ICEYE general channel info dataclass"""

    product_name: str
    channel_id: str
    swath: str
    product_level: ICEYEProductLevel
    polarization: SARPolarization
    projection: SARProjection
    acquisition_mode: ICEYEAcquisitionMode
    orbit_direction: OrbitDirection
    signal_frequency: float
    acq_start_time: PreciseDateTime
    acq_stop_time: PreciseDateTime

    @staticmethod
    def from_metadata(root: etree._Element | h5py.File, channel_id: str) -> ICEYEGeneralChannelInfo:
        """Generating ICEYEGeneralChannelInfo object directly from metadata.

        Parameters
        ----------
        root : etree._Element | h5py.File
            metadata root object
        channel_id : str
            channel id

        Returns
        -------
        ICEYEGeneralChannelInfo
            general channel info dataclass
        """

        polarization_dict = {
            "vv": SARPolarization.VV,
            "hh": SARPolarization.HH,
            "hv": SARPolarization.HV,
            "vh": SARPolarization.VH,
        }

        if isinstance(root, h5py.File):
            product_name = root["product_name"][()].decode()
            product_level = ICEYEProductLevel(root["product_level"][()].decode().upper())
            polarization = polarization_dict[root["polarization"][()].decode().lower()]
            acquisition_mode = ICEYEAcquisitionMode(root["acquisition_mode"][()].decode().lower())
            orbit_direction = OrbitDirection(root["orbit_direction"][()].decode().lower())
            signal_frequency = float(root["carrier_frequency"][()])
            acq_start_time = PreciseDateTime.fromisoformat(root["acquisition_start_utc"][()].decode())
            acq_stop_time = PreciseDateTime.fromisoformat(root["acquisition_end_utc"][()].decode())

        else:
            product_name = root.find("product_name").text
            product_level = ICEYEProductLevel(root.find("product_level").text.upper())
            polarization = polarization_dict[root.find("polarization").text.lower()]
            acquisition_mode = ICEYEAcquisitionMode(root.find("acquisition_mode").text.lower())
            orbit_direction = OrbitDirection(root.find("orbit_direction").text.lower())
            signal_frequency = float(root.find("carrier_frequency").text)
            acq_start_time = PreciseDateTime.fromisoformat(root.find("acquisition_start_utc").text)
            acq_stop_time = PreciseDateTime.fromisoformat(root.find("acquisition_end_utc").text)

        return ICEYEGeneralChannelInfo(
            product_name=product_name,
            channel_id=channel_id,
            swath="S1",
            product_level=product_level,
            polarization=polarization,
            projection=(
                SARProjection.GROUND_RANGE if product_level == ICEYEProductLevel.GRD else SARProjection.SLANT_RANGE
            ),
            acquisition_mode=acquisition_mode,
            orbit_direction=orbit_direction,
            signal_frequency=signal_frequency,
            acq_start_time=acq_start_time,
            acq_stop_time=acq_stop_time,
        )


@dataclass
class ICEYEChannelMetadata:
    """ICEYE channel metadata dataclass"""

    general_info: ICEYEGeneralChannelInfo
    general_sar_orbit: GeneralSarOrbit
    image_calibration_factor: float
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: ICEYEBurstInfo
    raster_info: meta.RasterInfo
    dataset_info: meta.DataSetInfo
    swath_info: ICEYESwathInfo
    sampling_constants: SARSamplingFrequencies
    acquisition_timeline: meta.AcquisitionTimeLine
    doppler_centroid_poly: SortedPolyList
    doppler_rate_poly: SortedPolyList
    incidence_angles_poly: ICEYEIncidenceAnglePolynomial
    pulse: meta.Pulse
    coordinate_conversions: ICEYECoordinateConversions
    state_vectors: ICEYEStateVectors


class ICEYEProduct:
    """ICEYE product object"""

    def __init__(self, path: Union[str, Path]) -> None:
        """ICEYE Product init from directory path.

        Parameters
        ----------
        path : Union[str, Path]
            path to ICEYE product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name

        # extracting full path to raster and metadata files
        channels = _find_valid_channels(self._product_path)
        self._channels_data = channels.copy()
        if channels is None:
            raise InvalidICEYEProduct("no channels found")
        data_paths = [v for val in channels.values() for v in val]
        self._data_paths = [
            v for v in data_paths if str(v).endswith(_GRD_IMAGE_FORMAT) or str(v).endswith(_SLC_DATA_EXTENSION)
        ]

        # channels list and number
        self._channels_number = len(channels)
        self._channels_list = list(channels.keys())

        # retrieve acquisition time
        self._acq_time = _retrieve_acquisition_time(data_paths)

        # retrieve scene footprint
        self._footprint = _retrieve_scene_footprint(data_paths)

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._acq_time

    @property
    def data_list(self) -> list[Path]:
        """Returning the list of raster data files of ICEYE product"""
        return self._data_paths

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of ICEYE product"""
        return self._channels_number

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels"""
        return self._channels_list

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_files_from_channel_name(self, channel_name: str) -> list[Path]:
        """Get files associated to a given channel name.

        GRD channels will return a list of two paths: .xml metadata file, .tif GeoTiff raster file

        SLC channels will return a list of a single path: .h5 HDF5 file

        Parameters
        ----------
        channel_name : str
            channel id name

        Returns
        -------
        list[Path]
            path to grd/slc files
        """
        return self._channels_data[channel_name]


def _find_valid_channels(product_path: Path) -> dict[str, list[Path]] | None:
    """Finding valid channels data from input product.

    GRD channels: two files [GeoTiff .tif file, metadata .xml file]

    SLC channels: one file [dataset HDF5 .h5 file]

    ICEYE product consists in a GRD channel and an SLC channel: if a single valid channel is found, the product is
    still considered valid

    Parameters
    ----------
    product_path : Path
        path to the input product

    Returns
    -------
    dict[str, list[Path]] | None
        dictionary with channel names as keys, channel files as values
    """

    # searching for valid GRD channel: GeoTif .tif file + .xml metadata
    grd_dict = _find_grd_channel(product_path=product_path)

    # searching for valid SLC channel: HDF5 .h5 file
    slc_dict = _find_slc_channel(product_path=product_path)

    out_dict = dict()
    if grd_dict is not None:
        out_dict.update(grd_dict)
    if slc_dict is not None:
        out_dict.update(slc_dict)

    return out_dict if out_dict else None


def _find_grd_channel(product_path: Path) -> dict[str, list[Path]] | None:
    """Find valid GRD ICEYE channel in input product.

    Parameters
    ----------
    product_path : Path
        Path to the ICEYE product folder

    Returns
    -------
    dict[str, list[Path]] | None
        dictionary of channel name as key, list of .tif and .xml files as value,
        None if no valid channel has been found
    """
    grd_file = list(product_path.glob("*" + _GRD_IMAGE_FORMAT))
    if len(grd_file) != 1:
        # no GRD channel found or more than a single GRD channel in product
        return None
    grd_file = grd_file[0]
    grd_metadata = product_path.joinpath(grd_file.stem).with_suffix(_GRD_METADATA_EXTENSION)
    if not grd_metadata.is_file():
        # GRD metadata .xml file not found
        return None
    grd_channel_files = [grd_file, grd_metadata]
    grd_channel_name = _grd_channel_name_definition(grd_metadata)

    return {grd_channel_name: grd_channel_files}


def _find_slc_channel(product_path: Path) -> dict[str, list[Path]] | None:
    """Find valid SLC ICEYE channel in input product.

    Parameters
    ----------
    product_path : Path
        Path to the ICEYE product folder

    Returns
    -------
    dict[str, Path] | None
        dictionary of channel name as key, .h5 file as value,
        None if no valid channel has been found
    """
    slc_file = list(product_path.glob("*" + _SLC_DATA_EXTENSION))
    if len(slc_file) != 1:
        # no SLC channel found or more than one in product
        return None

    slc_channel_name = _slc_channel_name_definition(slc_file[0])

    return {slc_channel_name: slc_file}


def _grd_channel_name_definition(metadata_file: Path) -> str:
    """Generating GRD channel name from metadata file.
    Name is composed as: grd + acq_mode + polarization

    Parameters
    ----------
    metadata_file : Path
        Path to the .xml metadata file

    Returns
    -------
    str
        grd channel name
    """
    # reading metadata file
    mtd = metadata_file.read_text()

    # regex init
    product_level_re = re.compile("(?<=<product_level>).*(?=</product_level>)")
    polarization_re = re.compile("(?<=<polarization>).*(?=</polarization>)")
    acquisition_mode_re = re.compile("(?<=<acquisition_mode>).*(?=</acquisition_mode>)")

    # info extraction
    product_level = product_level_re.findall(mtd)[0].lower()
    assert product_level == "grd"
    polarization = polarization_re.findall(mtd)[0]
    acquisition_mode = acquisition_mode_re.findall(mtd)[0]

    return "_".join([product_level, acquisition_mode.lower(), polarization.lower()])


def _slc_channel_name_definition(metadata_file: Path) -> str:
    """Generating SLC channel name from metadata file.
    Name is composed as: slc + acq_mode + polarization

    Parameters
    ----------
    metadata_file : Path
        Path to the .h5 file

    Returns
    -------
    str
        slc channel name
    """
    mtd = h5py.File(name=metadata_file)
    product_level = mtd["product_level"][()].decode().lower()
    assert product_level == "slc"
    acquisition_mode = mtd["acquisition_mode"][()].decode()
    polarization = mtd["polarization"][()].decode()
    mtd.close()

    return "_".join([product_level.lower(), acquisition_mode.lower(), polarization.lower()])


def _retrieve_acquisition_time(arg_in: list[Path]) -> PreciseDateTime:
    """Retrieving acquisition time for each channel in the ICEYE product.

    Parameters
    ----------
    arg_in : list[Path]
        list of relevant files in ICEYE product

    Returns
    -------
    PreciseDateTime
        acquisition time in PreciseDateTime format
    """
    xml_file = [f for f in arg_in if str(f).endswith(_GRD_METADATA_EXTENSION)]
    h5_file = [f for f in arg_in if str(f).endswith(_SLC_DATA_EXTENSION)]
    acq_time_grd, acq_time_slc = None, None
    if xml_file:
        # reading metadata file
        mtd = xml_file[0].read_text()
        acq_time_re = re.compile("(?<=<acquisition_start_utc>).*(?=</acquisition_start_utc>)")
        acq_time_grd = PreciseDateTime.fromisoformat(acq_time_re.findall(mtd)[0])

    if h5_file:
        mtd = h5py.File(name=h5_file[0])
        acq_time_slc = PreciseDateTime.fromisoformat(mtd["acquisition_start_utc"][()].decode())
        mtd.close()

    if acq_time_grd is not None and acq_time_slc is not None:
        # checking that acquisition times in both GRD and SLC are very close
        assert abs(acq_time_grd.sec85 - acq_time_slc.sec85) < 100
        return acq_time_grd

    if acq_time_grd is not None:
        return acq_time_grd

    if acq_time_slc is not None:
        return acq_time_slc


def _retrieve_scene_footprint(arg_in: list[Path]) -> tuple[float, float, float, float]:
    xml_file = [f for f in arg_in if str(f).endswith(_GRD_METADATA_EXTENSION)]
    h5_file = [f for f in arg_in if str(f).endswith(_SLC_DATA_EXTENSION)]
    footprint_grd, footprint_slc = None, None

    if xml_file:
        # reading metadata file
        mtd = xml_file[0].read_text()
        # scene coordinates boundaries are expressed as [x(col), y(row),lat,lon]
        corner_first_near_re = re.compile("(?<=<coord_first_near>).*(?=</coord_first_near>)")
        corner_first_far_re = re.compile("(?<=<coord_first_far>).*(?=</coord_first_far>)")
        corner_last_near_re = re.compile("(?<=<coord_last_near>).*(?=</coord_last_near>)")
        corner_last_far_re = re.compile("(?<=<coord_last_far>).*(?=</coord_last_far>)")
        corners = [
            [float(f) for f in corner_first_near_re.findall(mtd)[0].split()[2:]],
            [float(f) for f in corner_first_far_re.findall(mtd)[0].split()[2:]],
            [float(f) for f in corner_last_near_re.findall(mtd)[0].split()[2:]],
            [float(f) for f in corner_last_far_re.findall(mtd)[0].split()[2:]],
        ]
        latitudes = [c[0] for c in corners]
        longitudes = [c[1] for c in corners]
        footprint_grd = (min(latitudes), max(latitudes), min(longitudes), max(longitudes))

    if h5_file:
        mtd = h5py.File(name=h5_file[0])
        corners = np.vstack(
            [
                mtd["coord_first_near"][()][2:],
                mtd["coord_first_far"][()][2:],
                mtd["coord_last_near"][()][2:],
                mtd["coord_last_far"][()][2:],
            ]
        )
        mtd.close()
        footprint_slc = (corners.min(axis=0)[0], corners.max(axis=0)[0], corners.min(axis=0)[1], corners.max(axis=0)[1])

    if footprint_grd is not None and footprint_slc is not None:
        # checking that footprints in both GRD and SLC are very close
        assert np.abs(np.max(np.array(footprint_grd) - np.array(footprint_slc))) < 0.5
        return footprint_grd

    if footprint_grd is not None:
        return footprint_grd

    if footprint_slc is not None:
        return footprint_slc


def is_iceye_product(product: Union[str, Path]) -> bool:
    """Check if input path corresponds to a valid ICEYE product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - valid channels (GRD and/or SLC) can be found

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

    # check for GRD GeoTiff + .xml metadata
    try:
        channels = _find_valid_channels(product_path=product)
        if channels is None:
            return False
    except Exception:
        return False

    return True
