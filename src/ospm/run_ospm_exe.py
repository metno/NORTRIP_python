import subprocess
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

def run_ospm_exe(*, ospm_dir: Path) -> Tuple[int, str]:
    """
    Run the OSPM executable located in ospm_dir.

    Returns:
        (return_code, combined stdout/stderr)
    """
    resolved_dir = ospm_dir.resolve()
    exe_path = resolved_dir / "nortrip_ospm.exe"
    if not exe_path.exists():
        return 1, f"OSPM executable not found at {exe_path}"

    logger.info("Running OSPM executable")
    result = subprocess.run(
        [str(exe_path)],
        cwd=str(resolved_dir),
        capture_output=True,
        text=True,
        check=False,
    )
    combined_output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, combined_output
