# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Common Enum, dataclasses and other utilities
--------------------------------------------
"""

from dataclasses import dataclass
from enum import Enum, auto

from arepytools.timing.precisedatetime import PreciseDateTime
from numpy.polynomial import Polynomial


class StandardSARAcquisitionMode(Enum):
    """Standard cross-package SAR acquisition mode definition"""

    SCANSAR = auto()
    SPOTLIGHT = auto()
    STRIPMAP = auto()
    TOPSAR = auto()
    WAVE = auto()
    UNKNOWN = auto()


class SARRadiometricQuantity(Enum):
    """Enum class for radiometric analysis input/output quantity types"""

    BETA_NOUGHT = auto()
    SIGMA_NOUGHT = auto()
    GAMMA_NOUGHT = auto()


class SARPolarization(Enum):
    """Polarization enum class"""

    HH = "H/H"
    VV = "V/V"
    HV = "H/V"
    VH = "V/H"


class SARProjection(Enum):
    """Enum class for managing swath projection of product folder"""

    SLANT_RANGE = "SLANT RANGE"
    GROUND_RANGE = "GROUND RANGE"


class OrbitDirection(Enum):
    """Orbit direction: ascending or descending"""

    ASCENDING = "ascending"
    DESCENDING = "descending"


@dataclass
class SARSamplingFrequencies:
    """SAR signal sampling frequencies"""

    range_freq_hz: float
    range_bandwidth_freq_hz: float
    azimuth_freq_hz: float
    azimuth_bandwidth_freq_hz: float


@dataclass
class ConversionPolynomial:
    """Generic conversion polynomial wrapper"""

    azimuth_reference_time: PreciseDateTime
    origin: float
    polynomial: Polynomial
