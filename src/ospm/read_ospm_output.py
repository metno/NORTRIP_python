import logging
from pathlib import Path
import numpy as np

from input_classes import input_metadata

logger = logging.getLogger(__name__)


def _select_receptor_values(
    receptor_values: np.ndarray, choose_receptor: int, nodata: float
) -> np.ndarray:
    """Select or aggregate receptor columns based on choose_receptor."""
    if receptor_values.size == 0:
        return np.array([], dtype=float)

    cleaned = receptor_values.astype(float)
    cleaned[(cleaned < 0) | np.isnan(cleaned)] = nodata

    if choose_receptor == 3:
        result = np.full(cleaned.shape[0], nodata, dtype=float)
        for idx, row in enumerate(cleaned):
            valid = row[row != nodata]
            if valid.size > 0:
                result[idx] = float(np.mean(valid))
        return result

    column = max(0, min(cleaned.shape[1] - 1, choose_receptor - 1))
    return cleaned[:, column]


def read_ospm_output(*, output_file: Path, metadata: input_metadata) -> np.ndarray:
    """Read OSPM output file and return dispersion factors per time step."""
    if not output_file.exists():
        logger.warning("OSPM output file missing at %s", output_file)
        return np.array([], dtype=float)

    try:
        data = np.genfromtxt(output_file, skip_header=1)
    except Exception:
        logger.warning("Failed to parse OSPM output at %s", output_file)
        return np.array([], dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 8:
        logger.warning("Unexpected OSPM output format in %s", output_file)
        return np.array([], dtype=float)

    receptor_values = data[:, 6:8]
    selected = _select_receptor_values(
        receptor_values, metadata.choose_receptor_ospm, metadata.nodata
    )
    return selected
