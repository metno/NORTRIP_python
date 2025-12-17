import logging
from pathlib import Path
import numpy as np
from config_classes import model_file_paths, model_parameters
from initialise import time_config
from input_classes import (
    converted_data,
    input_airquality,
    input_metadata,
    model_variables,
)
from .read_ospm_output import read_ospm_output
from .run_ospm_exe import run_ospm_exe
from .write_input_file import write_input_file
from .write_parameter_file import write_parameter_file

logger = logging.getLogger(__name__)
PARAMETER_FILENAME = "nortrip_ospm_parameters.txt"
INPUT_FILENAME = "nortrip_ospm_input.txt"
OUTPUT_FILENAME = "nortrip_ospm_output.txt"


def OSPM_Main(
    *,
    paths: model_file_paths,
    model_variables: model_variables,
    model_parameters: model_parameters,
    metadata: input_metadata,
    airquality_data: input_airquality,
    time_config: time_config,
    converted: converted_data,
    ro: int = 0,
):
    """
    Run the OSPM dispersion model using use_ospm_flag == 1 pathway.
    """
    total_steps = time_config.max_time - time_config.min_time + 1

    if not paths.path_ospm:
        logger.error("OSPM path is not configured. Skipping OSPM run.")
        output = np.full(total_steps, metadata.nodata, dtype=float)
        for offset, ti in enumerate(
            range(time_config.min_time, time_config.max_time + 1)
        ):
            model_variables.f_conc[ti, ro] = output[offset]
        airquality_data.f_dis_input = output
        airquality_data.f_dis_available = (
            1 if output.size and np.any(output != metadata.nodata) else 0
        )
        return

    ospm_dir = Path(paths.path_ospm).resolve()
    input_dir = ospm_dir / "input"
    output_dir = ospm_dir / "output"
    parameter_file = input_dir / PARAMETER_FILENAME
    input_file = input_dir / INPUT_FILENAME
    output_file = output_dir / OUTPUT_FILENAME
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing OSPM input in %s", ospm_dir)
    write_parameter_file(
        parameter_file=parameter_file,
        metadata=metadata,
        model_parameters=model_parameters,
    )
    write_input_file(
        input_file=input_file,
        converted=converted,
        metadata=metadata,
        time_config=time_config,
        ro=ro,
    )

    status, message = run_ospm_exe(ospm_dir=ospm_dir)
    if status != 0:
        logger.warning("OSPM execution failed: %s", message.strip())
        output = np.full(total_steps, metadata.nodata, dtype=float)
        for offset, ti in enumerate(
            range(time_config.min_time, time_config.max_time + 1)
        ):
            model_variables.f_conc[ti, ro] = output[offset]
        airquality_data.f_dis_input = output
        airquality_data.f_dis_available = (
            1 if output.size and np.any(output != metadata.nodata) else 0
        )
        return

    logger.info("Reading OSPM output from %s", output_file)
    selected_values = read_ospm_output(output_file=output_file, metadata=metadata)
    output = np.full(total_steps, metadata.nodata, dtype=float)
    values_array = np.asarray(list(selected_values), dtype=float)
    fill_count = min(total_steps, len(values_array))
    if fill_count > 0:
        output[:fill_count] = values_array[:fill_count]

    for offset, ti in enumerate(range(time_config.min_time, time_config.max_time + 1)):
        model_variables.f_conc[ti, ro] = output[offset]

    airquality_data.f_dis_input = output
    airquality_data.f_dis_available = (
        1 if output.size and np.any(output != metadata.nodata) else 0
    )
