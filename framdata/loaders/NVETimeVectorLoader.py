"""
Loader for NVE time vector data.

This module provides the NVETimeVectorLoader class, which extends FileLoader and TimeVectorLoader
to handle metadata and validation for time vector data from NVE parquet files.
"""

from datetime import datetime, timedelta, tzinfo
from pathlib import Path
from typing import Any

import numpy as np
from framcore.loaders import FileLoader, TimeVectorLoader
from framcore.timevectors import ReferencePeriod
from numpy.typing import NDArray

from framdata.database_names.TimeVectorMetadataNames import TimeVectorMetadataNames as TvMn


class NVETimeVectorLoader(FileLoader, TimeVectorLoader):
    """Common interface for metadata in NVE TimeVectorLoaders."""

    def __init__(self, source: Path | str, require_whole_years: bool, relative_loc: Path | str | None = None) -> None:
        """
        Initialize NVETimeVectorLoader with source and optional relative location.

        Args:
            source (Path | str): Path or string to the source file.
            require_whole_years (bool): Flag for validating that the time vectors in the source contain data for complete years.
            relative_loc (Path | str | None, optional): Relative location, defaults to None.

        """
        super().__init__(source, relative_loc)

        self._data: dict[str, NDArray] = None
        self._meta: dict[str, bool | int | str | datetime | timedelta | tzinfo] = None

        self._require_whole_years = require_whole_years

    def is_max_level(self, vector_id: str) -> bool | None:
        """
        Check if the time vector is classified as a max level vector.

        Args:
            vector_id (str): ID of the time vector.

        Returns:
            bool | None: True if max level, False otherwise, or None if not specified.

        """
        return self.get_metadata(vector_id)[TvMn.IS_MAX_LEVEL]

    def is_zero_one_profile(self, vector_id: str) -> bool | None:
        """
        Check if the time vector is classified as a zero-one profile vector.

        Args:
            vector_id (str): ID of the time vector.

        Returns:
            bool | None: True if zero-one profile, False otherwise, or None if not specified.

        """
        return self.get_metadata(vector_id)[TvMn.IS_ZERO_ONE_PROFILE]

    def get_unit(self, vector_id: str) -> str:
        """
        Get the unit of the given time vector.

        Args:
            vector_id (str): ID of a time vector. Not used since all time vectors in the NVE parquet files have the same
                             unit.

        Returns:
            str: Unit of the time vector.

        """
        return self.get_metadata(vector_id)[TvMn.UNIT]

    def get_reference_period(self, vector_id: str) -> ReferencePeriod | None:
        """
        Get Reference perod from metadata.

        Args:
            vector_id (str): Not used.

        Raises:
            ValueError: If only one of start year or number of years are set in metadata.

        Returns:
            ReferencePeriod | None

        """
        start_year = self.get_metadata(vector_id)[TvMn.REF_PERIOD_START_YEAR]
        num_years = self.get_metadata(vector_id)[TvMn.REF_PERIOD_NUM_YEARS]

        ref_period = None
        if start_year and num_years:
            ref_period = ReferencePeriod(start_year=start_year, num_years=num_years)
        elif start_year or num_years:
            message = (
                f"{self}: Both {TvMn.REF_PERIOD_START_YEAR} and {TvMn.REF_PERIOD_NUM_YEARS} must be provided for a valid reference period."
                "Alternatively, both must be None for undefined reference period."
            )
            raise ValueError(message)
        return ref_period

    def validate_vectors(self) -> None:
        """
        Validate data in all vectors contained in the Loader.

        Conditions validated:
            - If vector contains negative values.
            (- If vector is a zero one profile and contains values outside the unit interval.) * not in use currently

        Raises:
            ValueError: When conditions are violated.

        """
        errors = set()
        for vector_id in self.get_ids():
            errors |= self._validate_vector(vector_id)

        if errors:
            message = f"Found errors in {self}:"
            for e in errors:
                message += f"\n - {e}."

            raise ValueError(message)

    def _process_meta(self, raw_meta: dict[str | bytes, str | bytes | int | bool | None]) -> dict[str, Any]:
        processed_meta, missing_keys = TvMn.cast_meta(raw_meta)

        optional_keys = {TvMn.ID_COLUMN_NAME, TvMn.FREQUENCY, TvMn.NUM_POINTS, TvMn.START}
        missing_keys -= optional_keys

        if missing_keys:
            msg = f"{self} could not find keys: {missing_keys} in metadata of file {self.get_source()}. Metadata: {processed_meta}"
            raise KeyError(msg)

        return processed_meta

    def _validate_vector(self, vector_id: str) -> set[str]:
        index = self.get_index(vector_id)
        values = self.get_values(vector_id)

        errors = set()

        # validate index length
        if index.get_num_periods() not in range(values.size - 1, values.size + 1):  # Since ListTimeIndex objects' num_periods can vary.
            errors.add(f"{vector_id} - {type(index)} with {index.get_num_periods()} periods and vector with size ({values.size}) do not match.")

        # validate negative and missing values
        negatives = values < 0
        if np.any(negatives):
            errors.add(f"{vector_id} contains {negatives.sum()} negative values.")
        nans = np.isnan(values)
        if np.any(nans):
            errors.add(f"{vector_id} contains {nans.sum()} nan values.")

        # validate that index is whole years if required
        if self._require_whole_years and not index.is_whole_years():
            errors.add(f"{vector_id} is required to contain whole years but its index ({index}) is not classified as is_whole_years.")

        # outside_unit_interval = ((0 <= values) & (values <= 1))
        # if self.is_zero_one_profile(vector_id) and outside_unit_interval.any():
        #     num_outside_range = outside_unit_interval.sum()
        #     errors.add(f"{vector_id} is classified as a zero one vector but contains {num_outside_range} values outside the range 0, 1.")

        # if not self.is_zero_one_profile(vector_id):
        #     ref_period = self.get_reference_period(vector_id)
        #     ref_start_date = ref_period.get_start_year()

        #     index = self.get_index(vector_id)

        return errors
