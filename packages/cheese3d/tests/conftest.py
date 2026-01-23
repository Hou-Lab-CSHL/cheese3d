import pytest
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "ephys: marks tests that require ephys data"
    )
    config.addinivalue_line(
        "markers", "video: marks tests that require video data"
    )

@pytest.fixture(scope="session")
def project_root():
    """Get the root directory of the cheese3d package."""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Get the directory containing test data."""
    data_dir = project_root / "tests" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir

@pytest.fixture(scope="session")
def test_projects_dir(project_root):
    """Get the directory for temporary test projects."""
    projects_dir = project_root / "tests" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    return projects_dir

@pytest.fixture
def temp_project_path(test_projects_dir, tmp_path):
    """Create a temporary path for a test project."""
    project_path = test_projects_dir / f"test_project_{tmp_path.name}"
    project_path.mkdir(parents=True, exist_ok=True)
    return project_path

@pytest.fixture(autouse=True)
def cleanup_temp_files(tmp_path):
    """Automatically clean up temporary files after each test."""
    yield
    # Cleanup happens automatically with tmp_path fixture

@pytest.fixture
def sample_config():
    """Get a sample minimal configuration for testing."""
    return {
        "recordings": [],
        "views": {},
        "keypoints": {},
        "model": {
            "backend": "dlc",
            "dataset_type": "default"
        },
        "sync": {},
        "triangulation": {}
    }

@pytest.fixture
def mock_video_path(test_data_dir):
    """Get a path to a mock video file (does not need to exist)."""
    return test_data_dir / "mock_video.mp4"

@pytest.fixture
def mock_ephys_path(test_data_dir):
    """Get a path to a mock ephys file (does not need to exist)."""
    return test_data_dir / "mock_ephys.nex"
