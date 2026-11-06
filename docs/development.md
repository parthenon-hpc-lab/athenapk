<!-- This file was made in part with generative AI. -->

# AthenaPK developer guide

## Formatting code

AthenaPK uses `clang-format` for C++, `black` for Python, and `cpplint` for C++ linting.
Tool versions are pinned in `.pre-commit-config.yaml`, and CI runs all three on
pull requests, including those from forks.

Install the optional commit hook to check staged files automatically:
```bash
python -m pip install pre-commit
pre-commit install
```
Run all checks manually with `pre-commit run --all-files --show-diff-on-failure`.
The first run downloads the tools and requires internet access.

To format all tracked files, run `./scripts/format.sh` or build the
`format-athenapk` CMake target. Both use the pinned hooks when available.
`./scripts/format.sh --local` uses locally installed formatters instead, with
`CLANG_FORMAT` and `BLACK` overrides supported. See the
[contributor guide](../CONTRIBUTING.md#formatting-code) for details.

## Dev container

You can open a [GitHub Codespace](https://docs.github.com/en/codespaces) or use VSCode to automatically open local Docker container using the CUDA CI container image.

If you have the [VSCode Dev Container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed, on opening this repository in VSCode, it will automatically prompt to ask if you want to re-open this repository in a container.
