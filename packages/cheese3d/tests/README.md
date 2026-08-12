# Testing

Run tests using:

```bash
# Run all tests
pixi run test

# Run unit tests only
pixi run test-unit

# Run integration tests only
pixi run test-integration

# Run tests with coverage report
pixi run test-cov

# Run tests manually from package directory (any backend env works for
# backend-agnostic tests; use -e lp / -e lp-cu13 / -e sleap for tests specific
# to those backends)
cd packages/cheese3d
pixi run -e dlc pytest

# Run specific test file
pixi run -e dlc pytest tests/test_config.py

# Run tests with specific marker
pixi run -e dlc pytest -m "not slow"
```

## Test Organization

- `tests/`: Test directory
- `tests/conftest.py`: Pytest configuration and shared fixtures
- `tests/data/`: Test data files (auto-created)
- `tests/projects/`: Temporary test projects (auto-created)

## Markers

- `slow`: Tests that take a long time to run (use `-m "not slow"` to skip)
- `integration`: Integration tests
- `unit`: Unit tests
- `gpu`: Tests requiring GPU
- `ephys`: Tests requiring ephys data
- `video`: Tests requiring video data
