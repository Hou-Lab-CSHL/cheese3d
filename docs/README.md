# cheese3d Documentation

This directory contains the documentation for the `cheese3d` package, built with [Sphinx](https://www.sphinx-doc.org/) and hosted on [ReadTheDocs](https://readthedocs.org/).

## Building Documentation Locally

### Using Pixi (Recommended)

The documentation is set up to use Pixi for dependency management. To build the documentation:

```bash
# From the project root
pixi run docs
```

Or manually:

```bash
# From the project root
pixi run -e docs sphinx-build -b html docs/source docs/_build/html
```

### Viewing Documentation

After building, you can start a local server to view the documentation:

```bash
# From the project root
pixi run docs-serve
```

Or manually:

```bash
# From the project root
cd docs/_build/html && python -m http.server
```

Then open your browser to http://localhost:8000

### Using Make (Alternative)

The documentation also includes a Makefile that's configured to use the Pixi docs environment:

```bash
# From the docs directory
make html
```

And to view the documentation:

```bash
# From the docs directory
make serve
```

## Documentation Structure

- `source/`: Contains the RST source files
  - `conf.py`: Sphinx configuration
  - `index.rst`: Documentation homepage
  - Other RST files: Main content pages
  - `api/`: API reference documentation

- `_build/`: Generated documentation (created when you build)
  - `html/`: HTML output
  - Other formats (PDF, EPUB) if enabled

## Adding New Pages

1. Create a new `.rst` file in the `source` directory
2. Add the file to the toctree in `index.rst` or another parent page
3. Rebuild the documentation

## Updating API Documentation

The API documentation is automatically generated from docstrings in the code. To update it:

1. Ensure your Python code has proper docstrings
2. Rebuild the documentation

## ReadTheDocs Integration

This documentation is set up for automatic building on ReadTheDocs when changes are pushed to the repository. The configuration is in the `.readthedocs.yaml` file in the project root.