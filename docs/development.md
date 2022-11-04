# AthenaPK developer guide

## Formatting code

AthenaPK uses `clang-format` for C++ and `black` for Python files to enforce code formatting.
All pull requests are checked automatically if the code is properly formatted.

The build target `format-athenapk` calls both formatters and automatically format all changes, i.e., before committing changes simply call `make format-athenapk` (or similar).
Alternatively, leave a comment with `@par-hermes format` in the open pull request to format the code directly on GitHub.
