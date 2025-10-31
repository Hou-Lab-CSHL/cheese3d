# cheese3d Documentation

This directory contains the documentation for the `cheese3d` package, built with [Sphinx](https://www.sphinx-doc.org/).

## Building Documentation Locally

The documentation is set up to use Pixi for dependency management. To build the documentation:

```bash
# From the project root
pixi run docs
```

### Viewing Documentation

After building, you can start a local server to view the documentation:

```bash
# From the project root
pixi run docs-serve
```

Then open your browser to link listed in your terminal output.

## Adding New Pages

1. Create a new `.rst` file in the `source` directory
2. Add the file to the toctree in `index.rst` or another parent page
3. Rebuild the documentation

## Updating API Documentation

The API documentation is automatically generated from docstrings in the code. To update it:

1. Ensure your Python code has proper docstrings
2. Rebuild the documentation
