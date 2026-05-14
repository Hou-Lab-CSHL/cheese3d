# cheese3d install environment

A minimal [pixi](https://pixi.sh) workspace for end users who want to **run**
Cheese3D. It installs `cheese3d` and `cheese3d-annotator` from PyPI plus the
native dependencies (ffmpeg, CUDA, etc.) that those packages expect from conda.

## Maintainer notes

- `pixi.lock` is refreshed automatically by the `publish-cheese3d` and
  `publish-cheese3d-annotator` GitHub workflows. After each tag-triggered
  publish, the workflow re-runs `pixi update` here and pushes the resulting
  lockfile bump back to `main`.

- If you need to refresh it locally (e.g. to test before a release), run:

  ```sh
  cd public-env
  pixi update
  git add pixi.lock && git commit -m "public-env: refresh lockfile"
  ```

- This workspace is intentionally separate from the root workspace so that the
  shipped `pixi.lock` does not include solver output for the `dev` / `docs`
  features. Keep the conda-side dependency list here in sync with the
  corresponding `[dependencies]` / `[target.*.dependencies]` blocks in
  `../pixi.toml` when those change.
