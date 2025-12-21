# Spacial Home Assignment: Polygon Detection

A Python library and web application for detecting closed polygons from a set of lines.

## Requirements

- Python 3.11+

## Installation

Create a virtual environment and install the package in editable mode:

```bash
python -m venv .venv
# Activate:
#   Windows: .venv\Scripts\Activate
#   Linux/Mac: source .venv/bin/activate

pip install -e .[dev]
```

## Usage

### Command Line Interface (CLI)

Run the detection algorithm on a sample JSON input file:

```bash
# Using the installed console script
polygons-cli --input data/a_wall_layer.json --outdir out
```

Or using `python -m`:
```bash
python -m polygons.cli --input data/a_wall_layer.json --outdir out
```

**Outputs** (in `out/` directory):
- `polygons.json`: Detected polygons coordinates.
- `polygons.png`: Visualization of detected polygons.
- `all_polygons.png`: Visualization of all candidate polygons (including discarded ones).

### Web Application

Start the Flask-based validation UI:

```bash
# Using the installed console script
polygons-web
```
Access the UI at [http://127.0.0.1:5000](http://127.0.0.1:5000).

### Running Tests

Run the test suite using `pytest`:

```bash
pytest
```

## Project Structure

- `src/` - Source code (standard layout).
- `data/` - Sample input JSON files.
- `scripts/` - Extra utility scripts (e.g., `repro_compactness.py`).
- `tests/` - Unit tests.

## Extras

- `scripts/repro_compactness.py`: An optional script to verify compactness calculations.

## Assumptions and Limitations

- The input JSON is expected to contain a list of lines with start and end coordinates.
- The algorithm assumes lines form reasonably closed shapes but attempts to handle gaps via snapping and quantization.
- Performance is optimized for typical floor plan sizes.