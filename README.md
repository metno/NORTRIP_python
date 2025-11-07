# NORTRIP Python

Single road Python version of NORTRIP

Containing scripts, example data and documentation for running NORTRIP in Python

## Setup

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver, written in Rust.

1. **Install uv** (if not already installed):
   [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/)

2. **Clone the repository**:
   ```bash
   git clone https://github.com/metno/NORTRIP_python.git
   cd NORTRIP-python
   ```

3. **Install the packages**:
   ```bash
   uv sync
   ```

### Option 2: Using pip and venv

1. **Clone the repository**:
   ```bash
   git clone https://github.com/metno/NORTRIP_python.git
   cd NORTRIP-python
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Install the packages**:
   ```bash
   pip install -e .
   ```

## Running the model

After setup, you can run NORTRIP using:
```bash
uv run nortrip model_paths/model_paths_and_files.xlsx
```
or with venv active:
```bash
nortrip model_paths/model_paths_and_files.xlsx
```

### Full help:
```bash
uv run nortrip --help
```
or with venv active:
```bash
nortrip --help
```

All available arguments:

- `paths` (required): Path to a `.xlsx` or `.txt` file describing the file paths and filenames used in the model run.
- `-p`/`--print`: Print some plot results to the terminal.
- `-f`/`--fortran`: Run the Fortran model implementation (work in progress).
- `-l`/`--log`: Disable logging output except for errors.
- `-pl`/`--plot <plot type>`: Choose which plots to generate. Accepted values are `all`, `none`, `normal` (default), `summary`, `temperature`, or an 11-digit bitstring like `11110000010`. Pass `list` to see a numbered description of each plot.

## Reading and saving as text

The reading and saving of data is implied by the file type of each file in the model paths file. Supported file types are `txt` and `xlsx`.

### Reading input files
For input files with multiple sheets the txt files end with with the sheet name. 

Example:

If the `Model parameter filename` in the model paths file is `Road_dust_parameter_table_example.txt` then the model parameters will be read from the following files:  

- `Road_dust_parameter_table_example_params.txt`
- `Road_dust_parameter_table_example_flags.txt`
- `Road_dust_parameter_table_example_activities.txt`


### Saving output file
If the `Model output data filename` in the model paths file ends with `.txt` then the model output data will be saved in txt format.

**Important:** Saving the used model parameters for the model run is only supported when saving is done in xlsx format.
