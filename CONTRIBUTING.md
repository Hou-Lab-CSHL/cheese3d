# Contributing to Cheese3D

Thanks for your interest in contributing!

## Getting started

1. Fork and clone the repository.
2. Install [Pixi](https://pixi.sh), then set up the environment:
   ```bash
   pixi install
   ```
3. Create a branch for your changes:
   ```bash
   git checkout -b my-change
   ```

## Making changes

- If you change behavior or add a feature, update the docs under `docs/source/`.
- Write a clear commit message describing the *why* of your change.

## Building the docs

```bash
pixi run docs        # build once
pixi run docs-serve  # build and serve with live reload
```

## Submitting a pull request

1. Push your branch to your fork.
2. Open a pull request against `main`.
3. Describe what changed and why, and link any related issues.

## Reporting bugs

Open an issue on GitHub and include:

- A short description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- Your OS, Python version, and Cheese3D version
