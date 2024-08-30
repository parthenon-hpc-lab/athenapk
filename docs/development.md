# AthenaPK developer guide

## Formatting code

AthenaPK uses `clang-format` for C++ and `black` for Python files to enforce code formatting.
All pull requests are checked automatically if the code is properly formatted.

The build target `format-athenapk` calls both formatters and automatically format all changes, i.e., before committing changes simply call `make format-athenapk` (or similar).
Alternatively, leave a comment with `@par-hermes format` in the open pull request to format the code directly on GitHub.

## Dev container

You can open a [GitHub Codespace](https://docs.github.com/en/codespaces) or use VSCode to automatically open local Docker container using the CUDA CI container image.

If you have the [VSCode Dev Container extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) installed, on opening this repository in VSCode, it will automatically prompt to ask if you want to re-open this repository in a container.
