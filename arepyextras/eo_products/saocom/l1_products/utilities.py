# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
SAOCOM reader support module
----------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import arepytools.io.metadata as meta
import numpy as np
import numpy.typing as npt
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

_MANIFEST_EXTENSION = ".xemt"
_RASTER_METADATA_EXTENSION = ".xml"


polarization_dict = {
    "vv": SARPolarization.VV,
    "hh": SARPolarization.HH,
    "hv": SARPolarization.HV,
    "vh": SARPolarization.VH,
}


class InvalidSAOCOMProduct(RuntimeError):
    """Invalid SAOCOM product"""


class SAOCOMProductType(Enum):
    """SAOCOM L1 product level"""

    SLC = "SLC"  # Slant Range, Single Look Complex (SLC, lvl 1)
    GRD = "GRD"  # Ground Range Multi Look Detected (GRD, lvl 1, phase lost)


class SAOCOMAcquisitionMode(Enum):
    """SAOCOM L1 acquisition modes"""

    STRIPMAP = "STRIPMAP"
    TOPSAR = "TOPSAR"


def raster_info_from_metadata(node: etree._Element) -> meta.RasterInfo:
    """Creating a RasterInfo Arepytools metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        RasterInfo metadata node

    Returns
    -------
    meta.RasterInfo
        RasterInfo metadata object
    """

    # assembling RasterInfo
    raster_info = meta.RasterInfo(
        lines=int(node.find("Lines").text),
        samples=int(node.find("Samples").text),
        filename=node.find("FileName").text,
        celltype=node.find("CellType").text,
        header_offset_bytes=int(node.find("HeaderOffsetBytes").text),
        row_prefix_bytes=int(node.find("RowPrefixBytes").text),
        byteorder=node.find("ByteOrder").text,
    )

    # lines
    lines_start = PreciseDateTime.from_utc_string(node.find("LinesStart").text)
    lines_step = float(node.find("LinesStep").text)
    lines_start_unit = node.find("LinesStart").get("unit")
    lines_step_unit = node.find("LinesStep").get("unit")

    # samples
    samples_start = float(node.find("SamplesStart").text)
    samples_step = float(node.find("SamplesStep").text)
    samples_start_unit = node.find("SamplesStart").get("unit")
    samples_step_unit = node.find("SamplesStep").get("unit")

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


def dataset_info_from_metadata(node: etree._Element) -> meta.DataSetInfo:
    """Creating a DataSetInfo Arepytools metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        DataSetInfo metadata node

    Returns
    -------
    meta.DataSetInfo
        DataSetInfo metadata object
    """

    # assembling DataSetInfo
    dataset_info = meta.DataSetInfo(
        acquisition_mode_i=node.find("AcquisitionMode").text, fc_hz_i=float(node.find("fc_hz").text)
    )
    dataset_info.sensor_name = node.find("SensorName").text
    dataset_info.image_type = node.find("ImageType").text
    dataset_info.projection = node.find("Projection").text
    dataset_info.side_looking = meta.ESideLooking(node.find("SideLooking").text)

    return dataset_info


def swath_info_from_metadata(node: etree._Element) -> meta.SwathInfo:
    """Creating a SwathInfo Arepytools metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        SwathInfo metadata node

    Returns
    -------
    meta.SwathInfo
        SwathInfo metadata object
    """

    swath_info = meta.SwathInfo(
        swath_i=node.find("Swath").text,
        polarization_i=node.find("Polarization").text,
        acquisition_prf_i=float(node.find("AcquisitionPRF").text),
    )
    try:
        swath_info.rank = int(node.find("Rank").text)
        swath_info.range_delay_bias = float(node.find("RangeDelayBias").text)
        swath_info.range_delay_bias_unit = "s"
        swath_info.acquisition_start_time = PreciseDateTime.from_utc_string(node.find("AcquisitionStartTime").text)
        swath_info.acquisition_start_time_unit = "Utc"
        swath_info.azimuth_steering_rate_reference_time = float(node.find("AzimuthSteeringRateReferenceTime").text)
        swath_info.az_steering_rate_ref_time_unit = node.find("AzimuthSteeringRateReferenceTime").get("unit")
        swath_info.echoes_per_burst = int(node.find("EchoesPerBurst").text)
        swath_info.azimuth_steering_rate_pol = tuple(
            [float(p.text) for p in node.findall("AzimuthSteeringRatePol/val")]
        )
        swath_info.rx_gain = float(node.find("RxGain").text)
    except Exception:
        # some of these info are optional in the metadata but all of them are not used so much
        pass

    return swath_info


def acquisition_timeline_from_metadata(node: etree._Element) -> meta.AcquisitionTimeLine | None:
    """Creating a AcquisitionTimeLine Arepytools metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        AcquisitionTimeLine metadata node

    Returns
    -------
    meta.AcquisitionTimeLine | None
        AcquisitionTimeLine metadata object or None if not found
    """
    if node is None:
        return None

    return meta.AcquisitionTimeLine(
        swst_changes_number_i=int(node.find("Swst_changes_number").text),
        swst_changes_azimuth_times_i=[float(t.text) for t in node.findall("Swst_changes_azimuthtimes/val")],
        swst_changes_values_i=[float(t.text) for t in node.findall("Swst_changes_values/val")],
    )


def sampling_constants_from_metadata(node: etree._Element) -> SARSamplingFrequencies:
    """Creating a SamplingConstants Arepytools metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        SamplingConstants metadata node

    Returns
    -------
    SARSamplingFrequencies
        SARSamplingFrequencies metadata object
    """
    return SARSamplingFrequencies(
        azimuth_freq_hz=float(node.find("faz_hz").text),
        azimuth_bandwidth_freq_hz=float(node.find("Baz_hz").text),
        range_freq_hz=float(node.find("frg_hz").text),
        range_bandwidth_freq_hz=float(node.find("Brg_hz").text),
    )


def pulse_from_metadata(node: etree._Element) -> meta.Pulse | None:
    """Creating a Pulse Arepytools metadata object from metadata file.

    Parameters
    ----------
    node : etree._Element
        Pulse metadata node

    Returns
    -------
    meta.Pulse | None
        Pulse metadata object or None if node not found
    """
    if node is None:
        return None

    return meta.Pulse(
        i_bandwidth=float(node.find("Bandwidth").text),
        i_pulse_direction=node.find("Direction").text,
        i_pulse_energy=float(node.find("PulseEnergy").text),
        i_pulse_length=float(node.find("PulseLength").text),
        i_pulse_sampling_rate=float(node.find("PulseSamplingRate").text),
        i_pulse_start_frequency=float(node.find("PulseStartFrequency").text),
        i_pulse_start_phase=float(node.find("PulseStartPhase").text),
    )


def doppler_poly_from_metadata(node: etree._Element, doppler_node_tag: str) -> SortedPolyList:
    """Creating a SortedPolyList Arepytools object for Doppler Polynomial from metadata file.

    Parameters
    ----------
    node : etree._Element
        Channel metadata node
    doppler_node_tag : str
        doppler polynomial node tag, it could be "DopplerCentroid" or "DopplerRate"

    Returns
    -------
    SortedPolyList
        Doppler polynomial SortedPolyList object
    """

    azimuth_ref_times, range_ref_times, coefficients = _extract_poly_info_from_node(node, doppler_node_tag)
    if not azimuth_ref_times and not range_ref_times and not coefficients:
        # GRD does not have these polynomials, so a set of 0-valued coefficients is created
        coefficients = [[0] * 7]
        azimuth_ref_times = [PreciseDateTime.from_utc_string(node.find("RasterInfo/LinesStart").text)]
        range_ref_times = [float(node.find("RasterInfo/SamplesStart").text)]

    doppler_poly_list = []
    for az_t_ref, rng_t_ref, coeffs in zip(azimuth_ref_times, range_ref_times, coefficients):
        if doppler_node_tag == "DopplerCentroid":
            doppler_poly_list.append(
                meta.DopplerCentroid(
                    i_ref_az=az_t_ref,
                    i_ref_rg=rng_t_ref,
                    i_coefficients=coeffs,
                )
            )
        else:
            doppler_poly_list.append(
                meta.DopplerRate(
                    i_ref_az=az_t_ref,
                    i_ref_rg=rng_t_ref,
                    i_coefficients=coeffs,
                )
            )

    doppler_vector = (
        meta.DopplerCentroidVector(doppler_poly_list)
        if doppler_node_tag == "DopplerCentroid"
        else meta.DopplerRateVector(doppler_poly_list)
    )

    return create_sorted_poly_list(poly2d_vector=doppler_vector)


def general_sar_orbit_from_saocom_state_vectors(state_vectors: SAOCOMStateVectors) -> GeneralSarOrbit:
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


@dataclass
class SAOCOMCoordinateConversions:
    """SAOCOM coordinate conversion"""

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
    def _get_conversion_polynomial(node: etree._Element, tag: str) -> list[ConversionPolynomial]:
        """Composing a list of ConversionPolynomial object from main metadata node and selected poly tag.

        Parameters
        ----------
        node : etree._Element
            Channel metadata node
        tag : str
            selected conversion polynomial, "GroundToSlant" and "SlantToGround"

        Returns
        -------
        list[ConversionPolynomial]
            list of ConversionPolynomial, one for each occurrence in main metadata node
        """
        azimuth_ref_times, range_ref_times, coefficients = _extract_poly_info_from_node(node, tag)

        # removing cross azimuth terms from coefficients
        coefficients = [[c[0], c[1], c[4], c[5], c[6]] for c in coefficients]

        return [
            ConversionPolynomial(
                azimuth_reference_time=az_time, origin=range_ref_times[idx], polynomial=Polynomial(coefficients[idx])
            )
            for idx, az_time in enumerate(azimuth_ref_times)
        ]

    @staticmethod
    def from_metadata(node: etree._Element) -> SAOCOMCoordinateConversions:
        """Generating SAOCOMCoordinateConversions object directly from metadata xml node.

        Evaluated polynomial gives Slant Range in meters.

        Parameters
        ----------
        node : etree._Element
            Channel metadata node

        Returns
        -------
        SAOCOMCoordinateConversions
            polynomials for coordinates conversion dataclass
        """

        ground_to_slant_poly_list = SAOCOMCoordinateConversions._get_conversion_polynomial(
            node=node, tag="GroundToSlant"
        )
        slant_to_ground_poly_list = SAOCOMCoordinateConversions._get_conversion_polynomial(
            node=node, tag="SlantToGround"
        )

        azimuth_ref_times = [PreciseDateTime.from_utc_string(t.text) for t in node.findall("GroundToSlant/taz0_Utc")]

        return SAOCOMCoordinateConversions(
            azimuth_reference_times=np.array(azimuth_ref_times),
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
class SAOCOMStateVectors:
    """SAOCOM orbit's state vectors"""

    num: int  # attitude data numerosity
    orbit_direction: OrbitDirection  # orbit direction: ascending or descending
    positions: np.ndarray  # platform position data with respect to the Earth-fixed reference frame
    velocities: np.ndarray  # platform velocity data with respect to the Earth-fixed reference frame
    time_axis: np.ndarray  # PreciseDateTime axis at which orbit state vectors apply
    time_step: float  # time axis step

    @staticmethod
    def from_metadata(node: etree._Element) -> SAOCOMStateVectors:
        """Generating a SAOCOMStateVectors object directly from metadata xml node.

        Parameters
        ----------
        node : etree._Element
            StateVectorData xml node

        Returns
        -------
        SAOCOMStateVectors
            orbit's state vectors dataclass
        """

        numerosity = int(node.find("nSV_n").text)
        time_step = float(node.find("dtSV_s").text)
        time_start = PreciseDateTime.from_utc_string(node.find("t_ref_Utc").text)

        return SAOCOMStateVectors(
            num=numerosity,
            orbit_direction=OrbitDirection(node.find("OrbitDirection").text.lower()),
            positions=np.array([float(p.text) for p in node.findall("pSV_m/val")]).reshape(-1, 3),
            velocities=np.array([float(v.text) for v in node.findall("vSV_mOs/val")]).reshape(-1, 3),
            time_axis=np.arange(0, numerosity * time_step, time_step) + time_start,
            time_step=time_step,
        )


@dataclass
class SAOCOMBurstInfo:
    """SAOCOM burst info"""

    num: int  # number of bursts in this swath
    lines_per_burst: int  # number of azimuth lines within each burst
    samples_per_burst: int  # number of range samples within each burst
    azimuth_start_times: np.ndarray  # zero doppler azimuth time of the first line of this burst
    range_start_times: np.ndarray  # zero doppler range time of the first sample of this burst

    @staticmethod
    def from_metadata_node(node: etree._Element, raster_info: meta.RasterInfo) -> SAOCOMBurstInfo:
        """Generating SAOCOMBurstInfo object directly from metadata xml node.

        Parameters
        ----------
        node : etree._Element
            BurstInfo metadata node
        samples : int
            number of samples per burst

        Returns
        -------
        SAOCOMBurstInfo
            swath's burst info dataclass
        """
        if node is None:
            # GRD does not have burst
            return SAOCOMBurstInfo(
                num=1,
                lines_per_burst=raster_info.lines,
                samples_per_burst=raster_info.samples,
                azimuth_start_times=np.array([raster_info.lines_start]),
                range_start_times=np.array([raster_info.samples_start]),
            )

        return SAOCOMBurstInfo(
            num=int(node.find("NumberOfBursts").text),
            lines_per_burst=int(node.find("LinesPerBurst").text),
            samples_per_burst=raster_info.samples,
            azimuth_start_times=np.array(
                [PreciseDateTime.from_utc_string(t.text) for t in node.findall("Burst/AzimuthStartTime")]
            ),
            range_start_times=np.array([float(t.text) for t in node.findall("RangeStartTime")]),
        )


@dataclass
class SAOCOMGeneralChannelInfo:
    """SAOCOM general channel info dataclass"""

    channel_id: str
    swath: str
    product_type: SAOCOMProductType
    polarization: SARPolarization
    projection: SARProjection
    acquisition_mode: SAOCOMAcquisitionMode
    orbit_direction: OrbitDirection
    signal_frequency: float
    acq_start_time: PreciseDateTime

    @staticmethod
    def from_metadata(node: etree._Element, channel_id: str) -> SAOCOMGeneralChannelInfo:
        """Generating SAOCOMGeneralChannelInfo object directly from metadata.

        Parameters
        ----------
        root : etree._Element
            Channel metadata node
        channel_id : str
            channel id

        Returns
        -------
        SAOCOMGeneralChannelInfo
            general channel info dataclass
        """
        projection = SARProjection(node.find("DataSetInfo/Projection").text)

        return SAOCOMGeneralChannelInfo(
            channel_id=channel_id,
            swath=node.find("SwathInfo/Swath").text,
            product_type=SAOCOMProductType.SLC if projection == SARProjection.SLANT_RANGE else SAOCOMProductType.GRD,
            polarization=SARPolarization(node.find("SwathInfo/Polarization").text),
            projection=projection,
            acquisition_mode=SAOCOMAcquisitionMode(node.find("DataSetInfo/AcquisitionMode").text),
            orbit_direction=OrbitDirection(node.find("StateVectorData/OrbitDirection").text.lower()),
            signal_frequency=float(node.find("DataSetInfo/fc_hz").text),
            acq_start_time=PreciseDateTime.from_utc_string(node.find("SwathInfo/AcquisitionStartTime").text),
        )


@dataclass
class SAOCOMChannelMetadata:
    """SAOCOM channel metadata dataclass"""

    general_info: SAOCOMGeneralChannelInfo
    general_sar_orbit: GeneralSarOrbit
    image_radiometric_quantity: SARRadiometricQuantity
    burst_info: SAOCOMBurstInfo
    raster_info: meta.RasterInfo
    dataset_info: meta.DataSetInfo
    swath_info: meta.SwathInfo
    sampling_constants: SARSamplingFrequencies
    acquisition_timeline: meta.AcquisitionTimeLine | None
    doppler_centroid_poly: SortedPolyList
    doppler_rate_poly: SortedPolyList
    pulse: meta.Pulse | None
    coordinate_conversions: SAOCOMCoordinateConversions
    state_vectors: SAOCOMStateVectors


class SAOCOMProduct:
    """SAOCOM product object"""

    def __init__(self, path: Union[str, Path]) -> None:
        """SAOCOM Product init from directory path.

        Parameters
        ----------
        path : Union[str, Path]
            path to SAOCOM product
        """
        self._product_path = Path(path)
        self._product_name = self._product_path.name
        self._manifest_path = list(self._product_path.glob("*" + _MANIFEST_EXTENSION))
        assert len(self._manifest_path) == 1
        self._manifest = SAOCOMManifest.from_file(self._manifest_path[0])
        self._footprint = self._manifest.footprint

    @property
    def manifest_path(self) -> Path:
        """Manifest .xemt file path"""
        return self._manifest_path

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""
        return self._manifest.acquisition_start_time

    @property
    def channels_number(self) -> int:
        """Returning the number of channels of SAOCOM product"""
        return len(self._manifest.channels)

    @property
    def channels_list(self) -> list[str]:
        """Returning the list of channels"""
        return self._manifest.channels

    @property
    def footprint(self) -> tuple[float, float, float, float]:
        """Product footprint as tuple of (min lat, max lat, min lon, max lon)"""
        return self._footprint

    def get_files_from_channel_name(self, channel_name: str) -> list[Path]:
        """Get files associated to a given channel name.

        Parameters
        ----------
        channel_name : str
            channel id name

        Returns
        -------
        list[Path]
            path to .xml metadata file and binary file
        """
        return self._manifest.raster_files[channel_name]


@dataclass
class SAOCOMManifest:
    """SAOCOM .xemt manifest class"""

    manifest_path: Path
    channels: list[str]
    polarizations: list[SARPolarization]
    acquisition_start_time: PreciseDateTime
    acquisition_end_time: PreciseDateTime
    raster_files: dict[str, list[Path]]  # for each channel, a list of raster binary file and .xml metadata
    footprint: tuple[float, float, float, float]  # min lat, max lat, min lon, max lon of the scene

    @staticmethod
    def from_file(path: str | Path) -> SAOCOMManifest:
        """Generating a SAOCOMManifest object representing the content of the .xemt file.

        Parameters
        ----------
        path : str | Path
            path to the .xemt file

        Returns
        -------
        SAOCOMManifest
            SAOCOM manifest dataclass
        """
        path = Path(path)
        assert str(path.name).endswith(_MANIFEST_EXTENSION)

        # loading file
        root = etree.parse(path).getroot()
        image_attribute_node = root.find("product/features/imageAttributes")
        components_node = root.find("product/dataFile/components")

        channels = _generate_channels_from_manifest(image_attribute_node=image_attribute_node)
        raster_files_relative = _get_raster_file_paths_from_manifest(components_node=components_node, channels=channels)
        raster_files_full_path = {
            k: [path.parent.joinpath(path.parent.name, vv) for vv in v] for k, v in raster_files_relative.items()
        }

        # recovering scene footprint
        scene_vertices = root.find("product/features/scene/frame").findall("vertex")
        latitudes = [float(f.find("lat").text) for f in scene_vertices]
        longitudes = [float(f.find("lon").text) for f in scene_vertices]
        footprint = (min(latitudes), max(latitudes), min(longitudes), max(longitudes))

        return SAOCOMManifest(
            manifest_path=path,
            acquisition_start_time=PreciseDateTime.fromisoformat(
                root.find("product/features/acquisition/acquisitionTime/startTime").text
            ),
            acquisition_end_time=PreciseDateTime.fromisoformat(
                root.find("product/features/acquisition/acquisitionTime/endTime").text
            ),
            channels=channels,
            polarizations=[
                polarization_dict[p.lower()]
                for p in root.find("product/features/acquisition/parameters/acquiredPols").text.split("-")
            ],
            raster_files=raster_files_full_path,
            footprint=footprint,
        )


def _extract_poly_info_from_node(
    node: etree._Element, doppler_node_tag: str
) -> tuple[list[PreciseDateTime], list[float], list[list[float]]]:
    """Extracting main info from a polynomial metadata node.

    Parameters
    ----------
    node : etree._Element
        Channel metadata node
    doppler_node_tag : str
        polynomial tag to be searched for, it could be:
        "DopplerCentroid", "DopplerRate", "SlantToGround", "GroundToSlant"

    Returns
    -------
    tuple[list[PreciseDateTime], list[float], list[list[float]]]
        reference azimuth times (a list with a time for each polynomial node found),
        reference range times (same),
        polynomial coefficients (same)
    """
    azimuth_ref_times = [PreciseDateTime.from_utc_string(t.text) for t in node.findall(doppler_node_tag + "/taz0_Utc")]
    range_ref_times = [float(t.text) for t in node.findall(doppler_node_tag + "/trg0_s")]
    coefficients = [[float(c.text) for c in poly.findall("val")] for poly in node.findall(doppler_node_tag + "/pol")]

    assert len(azimuth_ref_times) == len(range_ref_times) == len(coefficients)

    return azimuth_ref_times, range_ref_times, coefficients


def _generate_channels_from_manifest(image_attribute_node: etree._Element) -> list[str]:
    """Composing channel names from manifest metadata.

    Channel names convention: swath + '_' + pol

    Parameters
    ----------
    image_attribute_node : etree._Element
        imageAttributes metadata node

    Returns
    -------
    list[str]
        list of channel names
    """
    swath_infos = image_attribute_node.findall("SwathInfo")
    return [s.find("Swath").text.lower() + "_" + s.find("Polarization").text.lower() for s in swath_infos]


def _get_raster_file_paths_from_manifest(components_node: etree._Element, channels: list[str]) -> dict[str, Path]:
    """Retrieve path to .xml and binary files from metadata.

    Parameters
    ----------
    components_node : etree._Element
        components metadata node
    channels : list[str]
        list of channels names

    Returns
    -------
    dict[str, list[Path]]
        each key is a channel name, each value a list of .xml and binary file
    """
    components = components_node.findall("component")
    channel_files_association_dict = dict.fromkeys(channels)

    tag = "Science samples"
    # get metadata files from manifest related to input tag
    metadata_files = [c.find("componentPath").text for c in components if c.find("componentTitle").text == tag]
    metadata_content = [
        c.find("componentContent").text.lower() for c in components if c.find("componentTitle").text == tag
    ]
    # associate each file with the corresponding channel
    for channel in channel_files_association_dict:
        channel_parts = channel.split("_")
        channel_files_association_dict[channel] = np.where(
            [all([c in r for c in channel_parts]) for r in metadata_content]
        )[0][0]
    # creating list of linked binary-metadata file for each metadata file found in manifest
    linked_files = list(
        map(list, zip(metadata_files, [b.replace(_RASTER_METADATA_EXTENSION, "") for b in metadata_files]))
    )

    return {k: linked_files[v] for k, v in channel_files_association_dict.items()}


def is_saocom_product(product: Union[str, Path]) -> bool:
    """Check if input path corresponds to a valid SAOCOM product, basic version.

    Conditions to be met for basic validity:
        - path exists
        - path is a directory
        - metadata file exist (.xemt)
        - folder with same name of metadata file exists
        - subfolder Data in the previous folder

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

    # check for metadata file existence
    manifest_file = list(product.glob("*" + _MANIFEST_EXTENSION))
    if len(manifest_file) != 1:
        return False
    prod_name = manifest_file[0].name.strip(_MANIFEST_EXTENSION)
    prod_data_folder = product.joinpath(prod_name)

    if not prod_data_folder.exists() or not prod_data_folder.is_dir():
        return False

    try:
        # loading manifest
        SAOCOMManifest.from_file(manifest_file[0])
    except Exception:
        return False

    return True
