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
   # Install in development mode with all dependencies
   uv sync --dev
   
   # Or install in production mode
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

## Development Setup

### Using uv (Recommended)

1. **Install development dependencies**:
   ```bash
   uv sync --dev
   ```

2. **Run tests**:
   ```bash
   # Run all tests
   uv run pytest
   # Run specific test file
   uv run pytest tests/functions/test_antoine_func.py
   ```

3. **Run linting**:

   Recomended to configure ruff in your editor to run linting and formatting automatically.
   ```bash
   # Run linting
   uv run ruff check src tests
   # Run formatting
   uv run ruff format src tests
   ```

### Using pip and venv

1. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

2. **Run tests**:
   ```bash
   pytest
   
   # Run tests with coverage
   pytest --cov=src
   
   # Run specific test file
   pytest tests/functions/test_antoine_func.py
   ```

3. **Run linting**:
   ```bash
   ruff check src tests
   ruff format src tests
   ```

## Project Structure


## Usage

After installation, you can run NORTRIP using:

```bash
# If installed with uv
uv run nortrip

# Or if venv is active
nortrip
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Run tests: `uv run pytest` or `pytest`
5. Run linting: `uv run ruff check src tests` or `ruff check src tests`
6. Commit your changes: `git commit -m 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Submit a pull request

## License

[Add your license information here]