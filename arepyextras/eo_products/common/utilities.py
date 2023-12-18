# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Common Enum, dataclasses and other utilities
--------------------------------------------
"""
from dataclasses import dataclass
from enum import Enum, auto


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


@dataclass
class SARSamplingFrequencies:
    """SAR signal sampling frequencies"""

    range_freq_hz: float
    range_bandwidth_freq_hz: float
    azimuth_freq_hz: float
    azimuth_bandwidth_freq_hz: float
