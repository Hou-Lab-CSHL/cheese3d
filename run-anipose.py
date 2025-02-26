import hydra
from omegaconf import OmegaConf, DictConfig

from cheese3d.configutils import setup_hydra

@hydra.main(config_path="./configs", config_name="anipose-main", version_base=None)
def main(cfg: DictConfig):
    # get the Anipose config
    anipose_cfg = OmegaConf.to_object(cfg.anipose)
    # create the DLC project
    anipose_project = anipose_cfg.instantiate(root_dir=cfg.paths.anipose)

    # set up the DLC project if necessary
    if cfg.setupproject:
        anipose_project.setup_project()

    if cfg.clean:
        anipose_project.clean()
    else:
        if cfg.clean_data:
            anipose_project.clean_data()

        if cfg.clean_viz:
            anipose_project.clean_viz()

    if cfg.run_all:
        anipose_project.run_all()
    else:
        if cfg.run_data:
            anipose_project.run_data()
        else:
            if cfg.analyze:
                anipose_project.analyze()

            if cfg.filter:
                anipose_project.filter()

            if cfg.calibrate:
                anipose_project.calibrate()

            if cfg.triangulate:
                anipose_project.triangulate()

            if cfg.reproj_3d:
                anipose_project.reproject_3d()

        if cfg.run_viz:
            anipose_project.run_viz()
        else:
            if cfg.compare_viz:
                anipose_project.compare_viz()
            else:
                if cfg.label_2d:
                    anipose_project.label_2d()

                if cfg.label_3d:
                    anipose_project.label_3d()

            if cfg.label_combined:
                anipose_project.label_combined()

if __name__ == "__main__":
    setup_hydra()
    main()
