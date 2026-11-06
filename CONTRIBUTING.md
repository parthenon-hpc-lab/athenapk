<!-- This file was made in part with generative AI. -->

# AthenaPK Contributor Guide

The following sections cover the current guidelines that all users and developers are
expected to try to follow.
As with all development these guidelines are not set in stone but subject to discussion
and revision any time as necessary.

1. [User/community interaction and getting help](#usercommunity-interaction-and-getting-help)
2. [General development workflow](#general-development-workflow)
    - [Planning/requesting/announcing changes](#planningrequestingannouncing-changes)
    - [Sumary of branching model and versioning](#summary-of-branching-model-and-versioning)
    - [Contributing code](#contributing-code)
    - [Merging code](#merging-code)
        * [Merging code from a fork](#merging-code-from-a-fork)
    - [Use of agentic coding](#use-of-agentic-coding)
    - [Pre-commit hooks](#pre-commit-hooks)
    - [Linting code](#linting-code)
    - [Formatting code](#formatting-code)

## User/community interaction and getting help

If you have general questions on installation, usage, etc., please use
- the [discussion board](https://github.com/parthenon-hpc-lab/athenapk/discussions), or
- get in contact via the AthenaPK channel on matrix.org [#AthenaPK:matrix.org](https://app.element.io/#/room/#AthenaPK:matrix.org)
and provide as many details as possible (such as AthenaPK version, potential modifications,
configuration arguments, system being run on, ...).

The discussion board is recommended as it is easier to search.
Be sure that other users have the same question(s) so everyone benefits from the information
available.
Once the question is answered and of general interest, feel free to open a PR to add the
corresponding information to the main documentation.

If you discover a bug, please create an [issue](https://github.com/parthenon-hpc-lab/athenapk/issues)
with as many details as possible.

## General development workflow

### Planning/requesting/announcing Changes

If you would like to see a new feature, would like to implement
a new feature, or have any other item/idea that you would like to share
and/or discuss open an issue.

This helps us to keep track of who is working on what and prevent duplicated efforts
and/or allows to pool resources early on.

Use GitHub labels as appropriate and feel free to directly add/tag people to the issue.

### Summary of branching model and versioning

Only a single main branch called `main` exists and all PRs should be merged
into that branch.
Individual versions/releases are tracked by tags.

We aim at creating a new release at least in sync with [Parthenon](https://github.com/parthenon-hpc-lab/parthenon) releases,
i.e., currently about every six months.
However, if a new Parthenon feature is required in AthenaPK, which introduces a bump in the Parthenon
submodule that comes with breaking changes inside AthenaPK, a new release should be created immediately.
The intention is to make breaking changes (that give you nice new features!) are as transparent as possible.

Following steps need to be done for a new release:

- Create a new tag for that version using a modified [calender versioning](https://calver.org/) scheme.
Releases will be tagged `vYY.0M` i.e., short years and zero-padded months.
- Update the version in the main `CMakeLists.txt`.
- Update the `CHANGELOG.md` (i.e., add a release header and create new empty
categories for the "Current develop" section.

### Contributing code

In order to keep the main repository in order, everyone is encouraged to create feature
branches starting with their username, followed by a "/", and ending with a brief
description, e.g., "username/add\_feature\_xyz".
Working on branches in private forks is also fine but not recommended (as the automated
testing infrastructure will then first work upon opening a pull request).

Once all changes are implemented or feedback by other developers is required/helpful
open a pull request again the `main` branch of the main repository.

In that pull request refer to the issue that you have addressed.

If your branch is not ready for a final merge (e.g., in order to discuss the
implementation), mark it as "work in progress" by prepending "WIP:" to the subject.

### Merging code

In order for code to be merged into `main` it must

- obey the style guide (test with CPPLINT)
- pass the linting test (test with CPPLINT)
- pass the formatting test (see "Formatting Code" below)
- pass the existing test suite
- have at least one approval
- include tests that cover the new feature (if applicable)
- include documentation in the `docs/` folder (feature or developer; if applicable)
- include a brief summary in `CHANGELOG.md`

The reviewers are expected to look out for the items above before approving a merge
request.

#### Merging code from a fork

PRs can opened as usual from forks.
Unfortunately, the CI will not automatically trigger for forks. This is for security
reasons. As a workaround, in order to trigger the CI, a local branch will need to be created
on AthenaPK first. The forked code can then be merged into the local branch on
AthenaPK. At this point when a new merge request is opened from the local branch
to the `main` branch it will trigger the CI.
Someone of the core team will take care of the work around once a PR from a fork.
No extra work is required from the contributor.

The workaround workflow for the AthenaPK core developer may look like
(from a local AthenaPK repository pulling in changes from a `feature-A` branch in a fork):

```bash
$ git remote add external-A https://github.com/CONTRIBUTOR/athenapk.git
$ git fetch external-A
$ git checkout external-A/feature-A
$ git push --set-upstream origin CONTRIBUTOR/feature-A
```

NOTE: Any subsequent updates made to the forked branch will need to be manually pulled into the local branch.

### Use of Agentic Coding

The following general guidelines follow the current Parthenon guidelines, which
were discussed in March 2026 during a
[regular call](https://github.com/parthenon-hpc-lab/parthenon/wiki/2026.03.26-Meeting-Notes#handling-llm-supportedassisted-prs).
They are subject to live evaluation and may change in the future.

In general, there are no limitations to the use of agentic coding/LLMs/... given that
- the usage is disclosed
  - by adding `// This file was made in part with generative AI` to the file, and
  - by clearly stating the extent of usage in the PR description
- the person submitting the PR keeps ownership/responsiblity of all changes (i.e., conducts a thorough review themself)

In addition, the following recommendations should be followed
- for large/feature-style PRs a `plan.md` document outlining the changes should be part of the PR. The file should be named by the PR number and placed in the `docs/plan_histories` folder.
  - if possible, large feature-style PRs are split into smaller, logically separated chunks (to ease reviewing)
- condense/consolidate generated code (to ease maintenance) as it is (still) often much more verbose than necessary (and typical in the existing codebase)
- reviewers are expected to be much more skeptical and thorough compared to code written by long-term contributors, and possibly more willing to reject

### Pre-commit hooks
Formatting (`clang-format` for C++, `black` for Python) and linting (`cpplint`) are
configured in `.pre-commit-config.yaml`, which pins the tool versions.
The "Check Formatting" CI runs these checks on all tracked files, including on PRs
from forks, without requiring a build or submodule checkout.

Installing the Git hook is optional but recommended. It checks staged files and
fixes their formatting whenever you commit:
```bash
python -m pip install pre-commit
pre-commit install
```
The first run requires internet access to install the tools. If a hook changes files,
review and stage those changes before committing again.
To run all formatting and lint checks manually:
```bash
pre-commit run --all-files --show-diff-on-failure
```
Update pinned tool revisions with `pre-commit autoupdate --freeze` and review any
resulting formatting changes.

### Linting code
`cpplint` checks AthenaPK's C++ sources in `src/` and `tst/`, using `CPPLINT.cfg`.
The configuration accommodates existing relative/umbrella includes, mixed guard
prefixes, anonymous TODOs, and implicit constructors. Spacing and namespace
indentation follow clang-format.
It runs in CI and in the optional commit hook. To run only linting manually:
```bash
pre-commit run cpplint --all-files
```

### Formatting code
Run `./scripts/format.sh` from the repository, or
`cmake --build . --target format-athenapk` from your build directory, to format all
tracked C++ and Python files. Both commands use the formatting hooks and versions
pinned in `.pre-commit-config.yaml` when `pre-commit` is available.

If `pre-commit` is unavailable, or when running `./scripts/format.sh --local`, the
script uses locally installed tools. Override them with the `CLANG_FORMAT` and
`BLACK` environment or CMake variables. The script warns if their major versions
differ from the pinned versions, since the result may differ from CI.
This formatting command only runs the formatters; use `pre-commit run --all-files`
to include linting.

As an alternative for branches in the main repository, our "Hermes" automation can
format the code and commit the result when you post this comment in your PR:
```
@par-hermes format
```
After Hermes formats your code, run `git pull` to update your local branch.
