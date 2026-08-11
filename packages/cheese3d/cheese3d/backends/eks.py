import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from eks.multicam_smoother import fit_eks_multicam
from eks.singlecam_smoother import fit_eks_singlecam

from cheese3d.backends.core import (Pose2dBackend,
                                    get_pose_backend_class,
                                    register_pose_backend)
from cheese3d.backends.lightning_pose import dlc_df_to_h5
from cheese3d.config import KeypointConfig, ModelConfig
from cheese3d.utils import BoundingBox

class EKSBackend(Pose2dBackend):
    def __init__(self,
                 name: str,
                 root_dir: Path,
                 videos: List[Path],
                 keypoints: List[KeypointConfig],
                 models: Dict[str, Any],
                 crops: Optional[List[BoundingBox]] = None,
                 train_model: Optional[str] = None,
                 smooth_param: Optional[float] = None,
                 s_frames: Optional[List[Tuple[int | None, int | None]]] = None,
                 calibration: Optional[str | Path] = None,
                 camera_names: Optional[List[str]] = None,
                 view_names: Optional[Dict[str, str]] = None):
        self.name = name
        self.root_dir = Path(root_dir).absolute()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.videos = videos
        self.keypoints = keypoints
        self.crops = crops
        self.train_model = train_model
        self.smooth_param = smooth_param
        self.s_frames = [tuple(s) for s in s_frames] if s_frames is not None else s_frames
        self.calibration = (Path(calibration).expanduser().absolute()
                            if calibration is not None else None)
        self.camera_names = camera_names
        self.view_names = view_names or {}
        self.model_configs = self._coerce_model_configs(models)
        self._validate_model_configs()
        self.models = self._build_models()

    @property
    def project_path(self):
        return self.root_dir

    @property
    def anipose_model_path(self):
        first_model = next(iter(self.models.values()))

        return first_model.project_path

    @property
    def ensemble_preds_path(self):
        return self.root_dir.parent / "ensemble_preds"

    @classmethod
    def from_existing(cls,
                      project_path: Path,
                      root_dir: Path,
                      *args,
                      **kwargs):
        raise NotImplementedError("Importing existing EKS ensembles is not implemented.")

    def import_c3d_labels(self, videos: Dict[str, Path]):
        for model in self.models.values():
            model.import_c3d_labels(videos)

    def export_c3d_labels(self, videos: Dict[str, Path]):
        for model in self.models.values():
            model.export_c3d_labels(videos)

    def extract_frames(self, videos: Optional[List[Path]] = None):
        for model in self.models.values():
            model.extract_frames(videos)

    def train(self, gpu, iterate_dataset: bool = True,
              training_settings: Optional[dict] = None):
        """Forward shared training settings to each selected EKS submodel."""
        for model_name, model in self._selected_train_models().items():
            print(f"Training EKS submodel: {model_name}")
            model.train(gpu, iterate_dataset, training_settings)

    def track(self,
              videos: Dict[str, Path],
              output_dir: Path,
              calibration_path: Optional[Path] = None,
              tracking_settings: Optional[dict] = None) -> bool:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_key = output_dir.parent.name
        expected_outputs = [output_dir / f"{Path(video).stem}.h5" for video in videos.values()]
        existing_outputs = [path for path in expected_outputs if path.exists()]
        if len(existing_outputs) == len(expected_outputs):
            return True
        if len(existing_outputs) > 0:
            raise RuntimeError("EKS tracking would overwrite existing output files. "
                               f"Existing outputs: {existing_outputs}")
        model_csvs = {}
        for model_name, model in self.models.items():
            sub_output_dir = self.ensemble_preds_path / model_name / output_key
            print(f"using model {model_name}")
            model.track(videos=videos,
                        output_dir=sub_output_dir,
                        calibration_path=calibration_path,
                        tracking_settings=tracking_settings)
            model_csvs[model_name] = self._normalize_outputs(model_name,
                                                             videos,
                                                             sub_output_dir)
        print(f"running eks on: {output_key}")
        self._run_eks(model_csvs=model_csvs,
                      videos=videos,
                      output_dir=output_dir,
                      calibration_path=calibration_path)

        return True

    def _validate_model_configs(self):
        if len(self.model_configs) < 2:
            raise ValueError("EKSBackend requires at least two submodels.")
        backend_types = []
        for model_name, cfg in self.model_configs.items():
            backend_type = cfg.backend_type
            if backend_type == "eks":
                raise ValueError("EKSBackend submodels must be primitive backends; "
                                 f"got nested backend_type='eks' for {model_name}.")
            backend_types.append(backend_type)
        if len(set(backend_types)) != 1:
            raise ValueError("EKSBackend currently requires all submodels to use the same "
                             f"backend_type; got {sorted(set(backend_types))}.")
        if self.train_model is not None and self.train_model not in self.model_configs:
            raise ValueError(f"EKSBackend train_model={self.train_model!r} does not match "
                             f"any configured submodel: {sorted(self.model_configs.keys())}")

    def _build_models(self):
        models = {}
        for model_name, cfg in self.model_configs.items():
            backend_cls = get_pose_backend_class(cfg.backend_type)
            models[model_name] = backend_cls(name=model_name,
                                             root_dir=self.root_dir / model_name / "backend",
                                             videos=self.videos,
                                             keypoints=self.keypoints,
                                             crops=self.crops,
                                             **dict(cfg.backend_options or {}))

        return models

    def _coerce_model_configs(self, models: Dict[str, Any]) -> Dict[str, ModelConfig]:
        coerced = {}
        for model_name, cfg in dict(models or {}).items():
            if isinstance(cfg, ModelConfig):
                coerced[model_name] = ModelConfig(name=model_name,
                                                  backend_type=cfg.backend_type,
                                                  backend_options=dict(cfg.backend_options or {}))
            elif hasattr(cfg, "get"):
                coerced[model_name] = ModelConfig(
                    name=model_name,
                    backend_type=cfg.get("backend_type", "dlc"),
                    backend_options=dict(cfg.get("backend_options", {}) or {})
                )
            else:
                raise TypeError("EKSBackend model configs must be ModelConfig-like or "
                                f"mapping objects; got {type(cfg).__name__} for {model_name}.")

        return coerced

    def _selected_train_models(self):
        if self.train_model is None:
            return self.models

        return {self.train_model: self.models[self.train_model]}

    def _normalize_outputs(self,
                           model_name: str,
                           videos: Dict[str, Path],
                           output_dir: Path) -> Dict[Path, Path]:
        csv_dir = output_dir / "eks-input"
        csv_dir.mkdir(parents=True, exist_ok=True)
        dfs = {}
        csvs = {}
        for video in videos.values():
            video = Path(video)
            h5_path = output_dir / f"{video.stem}.h5"
            if not h5_path.exists():
                raise FileNotFoundError(f"EKS submodel {model_name!r} did not produce "
                                        f"expected pose file: {h5_path}")
            csv_path = csv_dir / f"{video.stem}.csv"
            dfs[video.resolve()] = pd.read_hdf(h5_path)
            csvs[video.resolve()] = csv_path
        # trim to the shortest video cause EKS can't handle different lengths
        min_frames = min(len(df) for df in dfs.values())
        for video, df in dfs.items():
            df.iloc[:min_frames].to_csv(csvs[video])

        return csvs

    def _resolve_calibration(self, calibration_path: Optional[Path], multicam: bool) -> Optional[Path]:
        if not multicam:
            return None
        calibration = self.calibration or calibration_path
        if calibration is None or not Path(calibration).exists():
            raise FileNotFoundError("EKS multicam smoothing requires a calibration TOML. "
                                    "Run `cheese3d calibrate` first or set "
                                    "`model.backend_options.calibration` to a valid TOML path.")

        return Path(calibration)

    def _run_eks(self,
                 model_csvs: Dict[str, Dict[Path, Path]],
                 videos: Dict[str, Path],
                 output_dir: Path,
                 calibration_path: Optional[Path]):
        resolved_videos = {view: Path(video).resolve() for view, video in videos.items()}
        prediction_csvs = {self.view_names.get(view, view):
                           [str(model_csvs[model_name][video])
                            for model_name in self.models.keys()]
                           for view, video in resolved_videos.items()}
        bodyparts = [keypoint.label for keypoint in self.keypoints]
        if len(resolved_videos) == 1:
            video = list(resolved_videos.values())[0]
            save_file = self.ensemble_preds_path / "eks" / f"{video.stem}_eks.csv"
            save_file.parent.mkdir(parents=True, exist_ok=True)
            smoothed_df, *_ = fit_eks_singlecam(input_source=list(prediction_csvs.values()),
                                                save_file=str(save_file),
                                                bodypart_list=bodyparts,
                                                smooth_param=self.smooth_param,
                                                s_frames=self.s_frames)
            dlc_df_to_h5(smoothed_df, output_dir / f"{video.stem}.h5")
        else:
            calibration = self._resolve_calibration(calibration_path, multicam=True)
            camera_dfs, *_ = fit_eks_multicam(input_source=prediction_csvs,
                                              save_dir=str(self.ensemble_preds_path / "eks"),
                                              bodypart_list=bodyparts,
                                              smooth_param=self.smooth_param,
                                              s_frames=self.s_frames,
                                              camera_names=self.camera_names,
                                              calibration=str(calibration))
            for video, camera_df in zip(resolved_videos.values(), camera_dfs):
                dlc_df_to_h5(camera_df, output_dir / f"{video.stem}.h5")

register_pose_backend("eks", EKSBackend)
