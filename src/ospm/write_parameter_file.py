import logging
from pathlib import Path

from config_classes import model_parameters
from input_classes import input_metadata

logger = logging.getLogger(__name__)


def write_parameter_file(
    *,
    parameter_file: Path,
    metadata: input_metadata,
    model_parameters: model_parameters,
) -> None:
    """Create the OSPM parameter file based on metadata."""
    parameter_file.parent.mkdir(parents=True, exist_ok=True)

    h_canyon = metadata.h_canyon[0] if metadata.h_canyon else 0.0
    canyon_width = metadata.b_canyon if metadata.b_canyon > 0 else metadata.b_road
    f_roof_ospm = (
        model_parameters.f_roof_ospm_override
        if model_parameters.f_roof_ospm_override > 0
        else metadata.f_roof_ospm
    )
    f_turb_ospm = (
        model_parameters.f_turb_ospm_override
        if model_parameters.f_turb_ospm_override > 0
        else metadata.f_turb_ospm
    )

    lines = [
        f"{2:d}\n",  # isub_ospm fixed to canyon mode
        f"{h_canyon:6.1f}\t{canyon_width:6.1f}\t{metadata.SL1_ospm:6.1f}\t{metadata.SL2_ospm:6.1f}\t{metadata.ang_road:6.1f}\n",
        f"{1:d}\n",  # nexc_ospm
        f"{359.0:6.2f}\n",
        f"{360.0:6.2f}\n",
        f"{h_canyon:6.2f}\n",
        f"{metadata.RecHeight_ospm:6.2f}\n",
        f"{f_roof_ospm:6.2f}\n",
        f"{f_turb_ospm:6.2f}\n",
    ]

    with parameter_file.open("w", encoding="utf-8") as file_handle:
        file_handle.writelines(lines)

    logger.debug("Wrote OSPM parameter file to %s", parameter_file)
