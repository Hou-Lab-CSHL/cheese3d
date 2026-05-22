import pytest
import shutil
from omegaconf import OmegaConf
from cheese3d.project import Ch3DProject
from cheese3d.config import (ProjectConfig,
                             ModelConfig,
                             VideoConfig,
                             MultiViewConfig,
                             TriangulationConfig)
from cheese3d.synchronize.core import SyncConfig

@pytest.fixture
def minimal_project_config():
    """Create a minimal valid configuration for testing."""
    from cheese3d.config import (_DEFAULT_KEYPOINTS,
                                 _DEFAULT_KEYPOINT_GROUPS,
                                 _DEFAULT_VIDEO_REGEX,
                                 _DEFAULT_TRIANGULATION_AXES,
                                 _DEFAULT_TRIANGULATION_REF)

    cfg = OmegaConf.structured(ProjectConfig)
    cfg.name = "test_project"
    cfg.video_root = "videos"
    cfg.model_root = "model"
    cfg.video_regex = _DEFAULT_VIDEO_REGEX
    cfg.views = MultiViewConfig()
    cfg.views["TL"] = VideoConfig(view="TL")
    cfg.views["TR"] = VideoConfig(view="TR")
    cfg.calibration = {"type": "cal"}
    cfg.sessions = []
    cfg.triangulation = TriangulationConfig(
        axes=_DEFAULT_TRIANGULATION_AXES,
        ref_point=_DEFAULT_TRIANGULATION_REF
    )
    cfg.keypoints = _DEFAULT_KEYPOINTS
    cfg.keypoint_groups = _DEFAULT_KEYPOINT_GROUPS
    cfg.ignore_keypoint_labels = ["ref(head-post)"]
    cfg.sync = SyncConfig(["crosscorr", "regression", "samplerate"])
    cfg.model = ModelConfig()

    return cfg

@pytest.fixture
def project_with_dummy_files(tmp_path, minimal_project_config):
    """Create a project with minimal dummy files for testing."""
    root = tmp_path
    project_name = "test_project"
    project_path = root / project_name
    # Create project directory
    project_path.mkdir(parents=True)
    # Write config file
    config_path = project_path / "config.yaml"
    minimal_project_config.name = project_name
    OmegaConf.save(minimal_project_config, config_path)
    # Create dummy directories
    videos_path = project_path / "videos"
    videos_path.mkdir(parents=True)
    model_path = project_path / "model"
    model_path.mkdir(parents=True)
    triangulation_path = project_path / "triangulation"
    triangulation_path.mkdir(parents=True)
    checkpoints_path = project_path / "checkpoints"
    checkpoints_path.mkdir(parents=True)
    # Create dummy session directory with videos matching the regex pattern
    session_path = videos_path / "session1"
    session_path.mkdir(parents=True)
    # Create dummy video files matching _DEFAULT_VIDEO_REGEX pattern: .*_{{type}}_{{view}}.*\.avi
    # where type is [^_]+ and view is TL|TR|L|R|TC|BC
    for view in ["TL", "TR"]:
        video_file = session_path / f"recording_test_{view}.avi"
        video_file.touch()
    # Create calibration videos
    cal_video_tl = session_path / "cal_test_TL.avi"
    cal_video_tl.touch()
    cal_video_tr = session_path / "cal_test_TR.avi"
    cal_video_tr.touch()
    # Create dummy ephys data
    ephys_path = project_path / "ephys"
    ephys_path.mkdir(parents=True)
    ephys_file = ephys_path / "session1_ephys.nex"
    ephys_file.touch()
    # Create dummy triangulation files
    triang_session = triangulation_path / "recording_test"
    triang_session.mkdir(parents=True)
    pose_3d_file = triang_session / "pose-3d"
    pose_3d_file.mkdir(parents=True)
    (pose_3d_file / "recording_test.csv").touch()
    cheese3d_dir = triang_session / "cheese3d"
    cheese3d_dir.mkdir(parents=True)
    (cheese3d_dir / "cheese3d_features.csv").touch()
    # Create dummy checkpoints directory (should be excluded from restore)
    dummy_checkpoint = checkpoints_path / "old_checkpoint.tar.xz"
    dummy_checkpoint.touch()

    return project_path

def test_checkpoint_creates_archive(project_with_dummy_files):
    """Test that checkpoint creates an archive file."""
    project = Ch3DProject.from_path(project_with_dummy_files)
    project.checkpoint(skip_source=True)
    # Get only the newly created checkpoint (exclude dummy files)
    checkpoints = [p for p in project.checkpoint_path.glob("*.tar.xz")
                   if not p.name.startswith("old_")]
    assert len(checkpoints) == 1
    assert checkpoints[0].exists()

def test_restore_extracted_project(project_with_dummy_files, tmp_path):
    """Test that restore extracts project files correctly."""
    # Create a checkpoint from original project
    project = Ch3DProject.from_path(project_with_dummy_files)
    project.checkpoint(skip_source=True, portable=True)
    checkpoints = [p for p in project.checkpoint_path.glob("*.tar.xz")
                   if not p.name.startswith("old_")]
    checkpoint_file = checkpoints[0]
    # Delete old checkpoint dummy file to test it doesn't get restored
    old_checkpoint = project.path / "checkpoints" / "old_checkpoint.tar.xz"
    if old_checkpoint.exists():
        old_checkpoint.unlink()
    # Restore directly to the new location
    project.restore(checkpoint_file, skip_source=True)
    # Verify files were restored to original project (restore extracts to project.root)
    # Since we're using the original project instance, it restores to its own root
    # This tests that restore works correctly on the same project
    assert (project.path / "config.yaml").exists()
    assert (project.path / "triangulation").exists()
    assert (project.path / "triangulation" / "recording_test").exists()
    # Verify checkpoints directory was not restored
    # The old_checkpoint.tar.xz should NOT be restored since it was excluded from checkpoint
    assert not (project.path / "checkpoints" / "old_checkpoint.tar.xz").exists()

def test_restore_with_videos(project_with_dummy_files, tmp_path):
    """Test that restore extracts videos when skip_source=False."""
    # Create a checkpoint from original project
    project = Ch3DProject.from_path(project_with_dummy_files)
    project.checkpoint(skip_source=False, portable=True)
    checkpoints = [p for p in project.checkpoint_path.glob("*.tar.xz")
                   if not p.name.startswith("old_")]
    checkpoint_file = checkpoints[0]
    # Delete the original videos to test that they are restored
    videos_path = project.path / "videos"
    session_videos = list(videos_path.glob("*.avi"))
    for v in session_videos:
        v.unlink()
    # Restore the checkpoint
    project.restore(checkpoint_file, skip_source=False)
    # Verify videos were restored
    assert (project.path / "videos" / "session1" / "recording_test_TL.avi").exists()
    assert (project.path / "videos" / "session1" / "recording_test_TR.avi").exists()
    # Verify calibration videos were also restored
    assert (project.path / "videos" / "session1" / "cal_test_TL.avi").exists()
    assert (project.path / "videos" / "session1" / "cal_test_TR.avi").exists()

def test_checkpoint_roundtrip(project_with_dummy_files, tmp_path):
    """Test complete roundtrip: checkpoint -> restore -> verify."""
    # Create a checkpoint
    original_project = Ch3DProject.from_path(project_with_dummy_files)
    original_project.checkpoint(skip_source=True)
    checkpoints = [p for p in original_project.checkpoint_path.glob("*.tar.xz")
                   if not p.name.startswith("old_")]
    checkpoint_file = checkpoints[0]
    # Delete the triangulation data to test that it gets restored
    triangulation_path = original_project.path / "triangulation" / "recording_test"
    if triangulation_path.exists():
        shutil.rmtree(triangulation_path)
    # Delete the old checkpoint dummy file to test it doesn't get restored
    old_checkpoint = original_project.path / "checkpoints" / "old_checkpoint.tar.xz"
    if old_checkpoint.exists():
        old_checkpoint.unlink()
    # Restore the checkpoint
    original_project.restore(checkpoint_file, skip_source=True)
    # Verify config and triangulation files were restored
    assert (original_project.path / "config.yaml").exists()
    assert (original_project.path /
            "triangulation" /
            "recording_test" /
            "pose-3d" /
            "recording_test.csv").exists()
    assert (original_project.path /
            "triangulation" /
            "recording_test" /
            "cheese3d" /
            "cheese3d_features.csv").exists()
    # Verify checkpoints directory was not restored (prevents recursion)
    # The old_checkpoint.tar.xz should NOT be restored since it was excluded from checkpoint
    assert not (original_project.path / "checkpoints" / "old_checkpoint.tar.xz").exists()

def test_restore_checkpoint_file_not_found(project_with_dummy_files):
    """Test that restore raises error when checkpoint file doesn't exist."""
    project = Ch3DProject.from_path(project_with_dummy_files)
    with pytest.raises(FileNotFoundError, match="Checkpoint file not found"):
        project.restore("nonexistent_checkpoint.tar.xz")

def test_checkpoint_creates_multiple_checkpoints(project_with_dummy_files):
    """Test that multiple checkpoints can be created with timestamps."""
    import time
    project = Ch3DProject.from_path(project_with_dummy_files)
    project.checkpoint(skip_source=True)
    initial_checkpoints = sorted([p for p in project.checkpoint_path.glob("*.tar.xz")
                                  if not p.name.startswith("old_")])
    assert len(initial_checkpoints) == 1
    # Wait a moment to ensure different timestamp
    time.sleep(2)
    project.checkpoint(skip_source=True)
    all_checkpoints = sorted([p for p in project.checkpoint_path.glob("*.tar.xz")
                              if not p.name.startswith("old_")])
    assert len(all_checkpoints) == 2
    # Check that there are two unique checkpoint files with different names (timestamps)
    assert len(all_checkpoints) == 2
    assert all_checkpoints[0] != all_checkpoints[1]
    assert all_checkpoints[0].name != all_checkpoints[1].name

def test_checkpoint_uses_relative_paths(project_with_dummy_files):
    """Test that checkpoint stores relative paths, not absolute paths."""
    import tarfile
    project = Ch3DProject.from_path(project_with_dummy_files)
    project.checkpoint(skip_source=True)
    checkpoints = [p for p in project.checkpoint_path.glob("*.tar.xz")
                   if not p.name.startswith("old_")]
    assert len(checkpoints) == 1
    checkpoint_file = checkpoints[0]
    # Verify all paths in the archive are relative (don't start with /)
    with tarfile.open(checkpoint_file, "r:xz") as tar:
        for member in tar.getmembers():
            # All paths should be relative (start with project name, not /)
            assert not member.name.startswith("/"), f"Found absolute path in archive: {member.name}"
            # All paths should start with the project name
            assert member.name.startswith(project.name), f"Unexpected path: {member.name}"

def test_restore_reads_config_for_filtering(project_with_dummy_files):
    """Test that restore reads video_root/ephys_root from the checkpoint config."""
    # Initialize a project with custom video_root
    # Modify the config to have a custom video_root
    config_file = project_with_dummy_files / "config.yaml"
    config = OmegaConf.load(config_file)
    original_video_root = config.video_root
    config.video_root = "custom_video_dir"  # Use a non-standard name
    OmegaConf.save(config, config_file)
    project = Ch3DProject.from_path(project_with_dummy_files)
    # Create checkpoint with skip_source=True
    project.checkpoint(skip_source=True)
    checkpoints = [p for p in project.checkpoint_path.glob("*.tar.xz")
                   if not p.name.startswith("old_")]
    checkpoint_file = checkpoints[0]
    # Delete the custom_video_dir (which should be empty since we just renamed it)
    custom_video_path = project.path / "custom_video_dir"
    if custom_video_path.exists():
        shutil.rmtree(custom_video_path)
    # Restore should work because it reads the config and uses video_root from there
    project.restore(checkpoint_file, skip_source=True)
    # Restore config and verify
    config.video_root = original_video_root
    OmegaConf.save(config, config_file)
