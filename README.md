# NORTRIP Python

Single road python version of NORTRIP

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
   git clone <repository-url>
   cd NORTRIP-python
   ```

3. **Install the package**:
   ```bash
   uv sync
   ```

4. **Run the package**:
   ```bash
   uv run nortrip
   ```

### Option 2: Using pip and venv

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd NORTRIP-python
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment**:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate     # On Windows
   ```

4. **Install the package**:
   ```bash
   # Install in development mode with all dependencies
   pip install -e ".[dev]"
   
   # Or install in production mode
   pip install -e .
   ```

5. **Run the package**:
   ```bash
   nortrip
   ```


## Usage

After setup, you can run NORTRIP using:

```bash
# If installed with uv
uv run nortrip

# Or if venv is active
nortrip
```


## License

[Add your license information here]