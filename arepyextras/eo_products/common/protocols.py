# SPDX-FileCopyrightText: Aresys S.r.l. <info@aresys.it>
# SPDX-License-Identifier: MIT

"""
Common Protocols to be matched for each implementation
------------------------------------------------------
"""

from pathlib import Path
from typing import Protocol, runtime_checkable

from arepytools.timing.precisedatetime import PreciseDateTime


@runtime_checkable
class EOL1ProductProtocol(Protocol):
    """Protocol to define the structure of a L1 main product object across different formats"""

    @property
    def acquisition_time(self) -> PreciseDateTime:
        """Acquisition start time for this product"""

    @property
    def data_list(self) -> list[Path]:
        """List of raster data files in product"""

    @property
    def metadata_list(self) -> list[Path]:
        """List of metadata files in product"""

    @property
    def channels_number(self) -> int:
        """Number of channels in product"""

    @property
    def channels_list(self) -> list[str]:
        """List of channels in terms of SwathID (swath-polarization), or unique identifier such as name/number"""

    def get_files_from_channel_name(self, channel_name: str) -> tuple[Path, Path, ...]:
        """Get metadata, raster and in case other path to files on disk corresponding to that channel identifier.

        Parameters
        ----------
        channel_name : str
            selected channel identifier

        Returns
        -------
        tuple[Path, Path, ...]
            metadata file path,
            raster file path,
            ...
        """
