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

### Formatting code
We use `clang-format` to automatically format the C++ code. If you have clang-format installed
locally, you can always execute `make format-athenapk` or `cmake --build . --target format-athenapk` from
your build directory to automatically format the code.

If you don't have `clang-format` installed locally, our "Hermes" automation can always
format the code for you. Just create the following comment in your PR, and Hermes will
format the code and automatically commit it:
```
@par-hermes format
```

After Hermes formats your code, remember to run `git pull` to update your local tracking
branch.

**WARNING:** Due to a limitation in GitHub Actions, the "Check Formatting" CI will not
run, which will block merging. If you don't plan on making any further commits, you or a
reviewer need to manually trigger the CI Action for the PR.

In addition to `clang-format`, `black` is used to enforce formatting on python scripts.
Running:
```
@par-hermes format
```

Will also format all the ".py" files found in the repository.
